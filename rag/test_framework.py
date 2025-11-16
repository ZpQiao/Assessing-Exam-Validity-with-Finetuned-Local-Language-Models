import json
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from typing import List, Dict, Tuple, Optional
import pandas as pd
from tqdm import tqdm
import time
from datetime import datetime
import os
import re
import pickle
from pathlib import Path

class ProbabilityTestFramework:
    def __init__(self, 
                 model_name: str, 
                 device: str = "cuda",
                 use_quantization: bool = False,
                 quantization_bits: int = 4,
                 local_model_path: str = None,
                 temperature: float = 0.0,
                 
                 use_rag: bool = False,
                 rag_system = None,
                 rag_k: int = 3):
        """
        初始化测试框架
        
        Args:
            model_name: 模型名称
            device: 运行设备
            use_quantization: 是否使用量化
            quantization_bits: 量化位数 (4 or 8)
            local_model_path: 本地模型路径（优先使用）
            temperature: 生成温度（0为确定性输出）
            use_rag: 是否使用RAG增强
            rag_system: RAG系统实例（ProbabilityRAG对象）
            rag_k: 检索的相关问题数量
        """
        self.temperature = temperature
        self.use_rag = use_rag
        self.rag_system = rag_system
        self.rag_k = rag_k
        self._last_rag_meta = None

        if use_rag and rag_system is None:
            raise ValueError("rag_system instance needed")
            
        # 决定实际加载路径
        actual_model_path = model_name
        if local_model_path and os.path.exists(local_model_path):
            actual_model_path = local_model_path
            print(f"✓ 从本地加载模型: {local_model_path}")
        else:
            if local_model_path:
                print(f"⚠️ 本地路径不存在: {local_model_path}")
            print(f"从 HuggingFace 加载模型: {model_name}")
        
        print(f"  量化: {'是 (' + str(quantization_bits) + '-bit)' if use_quantization else '否 (BF16)'}")
        print(f"  温度: {temperature} ({'确定性' if temperature == 0 else '随机性'})")
        
        self.model_name = model_name
        self.device = device
        self.use_quantization = use_quantization
        
        # 加载分词器
        self.tokenizer = AutoTokenizer.from_pretrained(
            actual_model_path,
            use_fast=False,
            trust_remote_code=True
        )
        
        # 配置量化参数
        if use_quantization:
            from transformers import BitsAndBytesConfig
            
            if quantization_bits == 4:
                quant_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_use_double_quant=True,
                    bnb_4bit_quant_type="nf4",
                    bnb_4bit_compute_dtype=torch.bfloat16,
                )
            elif quantization_bits == 8:
                quant_config = BitsAndBytesConfig(
                    load_in_8bit=True,
                )
            else:
                raise ValueError(f"不支持的量化位数: {quantization_bits}")
            
            # 加载量化模型
            self.model = AutoModelForCausalLM.from_pretrained(
                actual_model_path,
                device_map="auto",
                torch_dtype=torch.bfloat16,
                quantization_config=quant_config,
                trust_remote_code=True,
            )
        else:
            # 加载 BF16 模型
            self.model = AutoModelForCausalLM.from_pretrained(
                actual_model_path,
                device_map="auto",
                torch_dtype=torch.bfloat16,
                trust_remote_code=True
            )
        
        # 判断模型类型
        self.is_thinking = "thinking" in model_name.lower()
        
        print(f"✓ 模型加载完成! 类型: {'Thinking' if self.is_thinking else 'Instruct'}")

    def format_question(self, item: Dict, language: str = "english") -> str:
        """
        格式化问题为 prompt
        自动选择基础版本或RAG增强版本
        """
        self._last_rag_meta = None   # 避免上一题的缓存污染

        if self.use_rag:
            return self._format_question_with_rag(item, language)
        else:
            return self._format_question_basic(item, language)
    
    def _format_question_basic(self, item: Dict, language: str = "english") -> str:
        """格式化问题为 prompt（使用 JSON 格式输出，数字答案）"""
        lang_data = item[language]
        
        # 使用数字标记选项（1, 2, 3, ...）
        options_text = ""
        for i, option in enumerate(lang_data['options'], 1):
            options_text += f"{i}. {option}\n"
        
        # 根据语言调整 prompt
        if language == "english":
            prompt = f"""Please solve the following probability theory problem and select the correct answer.

Context: {lang_data['context']}

Question: {lang_data['question']}

Options:
{options_text}

Please respond with your reasoning followed by a JSON object in this exact format:
{{"answer": N}}

where N is the number of your chosen option (1, 2, 3, 4, 5, or 6).

Your response:"""
        else:  # danish
            prompt = f"""Løs følgende sandsynlighedsteori problem og vælg det rigtige svar.

Kontekst: {lang_data['context']}

Spørgsmål: {lang_data['question']}

Valgmuligheder:
{options_text}

Svar venligst med din ræsonnering efterfulgt af et JSON-objekt i dette nøjagtige format:
{{"answer": N}}

hvor N er nummeret på dit valgte svarmulighed (1, 2, 3, 4, 5 eller 6).

Dit svar:"""
        
        return prompt

    def _get_attr(self, obj, key, default=None):
        # 兼容 dict / 对象
        if isinstance(obj, dict):
            return obj.get(key, default)
        return getattr(obj, key, default)

    def _format_question_with_rag(self, item: Dict, language: str = "english") -> str:
        print("[format_question] entering RAG branch")
        lang_data = item[language]
        
        # 1. 构造查询文本
        query = f"{lang_data['context']} {lang_data['question']}".strip()
        
        # 2. 使用RAG检索相关问题
        try:
            raw = self.rag_system.retrieve_relevant(
                query, 
                k=self.rag_k,
                use_diversity=True,
                use_weights=True,
                verbose=False,
                exclude_base_key=item['base_key']
            )
            retrieved_questions, similarities, indices = raw
                    # 缓存：给 test_single_item 写入 CSV 用
            base_keys = []
            for q in retrieved_questions:
                bk = self._get_attr(q, 'base_key', '')
                base_keys.append(bk)
    
            def _to_pylist(x):
                if x is None:
                    return None
                try:
                    import numpy as np
                    return np.asarray(x).astype(float).tolist()
                except Exception:
                    return list(x)
    
            self._last_rag_meta = {
                'retrieved_base_keys': base_keys,
                'retrieved_similarities': _to_pylist(similarities),
                'retrieved_indices': _to_pylist(indices),
            }
            
        except Exception as e:
            print(f"RAG检索失败: {e}, 降级为基础prompt")
            return self._format_question_basic(item, language)
        
        # 3. 构建增强prompt
        prompt_parts = []
        
        # 添加系统指令

        if language == "english":
            prompt_parts.append(
                "You are an expert in probability and statistics. "
                "Below are some related problems from past exams that MAY provide useful context. "
                "However, they may not be directly applicable to this specific problem. "
                "VERY IMPORTANT:\n"
                "- DO NOT copy hidden assumptions (like independence, a parameter p, or a specific distribution)\n"
                "  from the references into the current problem.\n"
                "- Use ONLY assumptions that are explicitly stated in the CURRENT problem.\n"
                "- If a reference problem introduces a parameter p, that does NOT mean p exists here.\n"
                "- If there is any conflict, ALWAYS follow the CURRENT problem.\n\n"
                "Always derive your solution independently and verify your calculations.\n\n"
                "COIN-TOSS CONVENTION FOR THESE EXAMS:\n"
                "- If a problem just says \"a coin\" and does NOT define p, assume a fair coin: p = 1/2.\n"
                "- Only treat p as unknown if the current problem explicitly introduces p.\n\n"
            )
        else:  # danish
            prompt_parts.append(
                "Du er ekspert i sandsynlighedsteori og statistik. "
                "Nedenfor er nogle relaterede problemer fra tidligere eksamener, der MÅSKE kan give nyttig kontekst. "
                "De er dog ikke nødvendigvis direkte anvendelige på dette specifikke problem. "
                "MEGET VIGTIGT:\n"
                "- KOPIÉR IKKE skjulte antagelser (som uafhængighed, en parameter p eller en særlig fordeling)\n"
                "  fra referenceopgaverne over i den aktuelle opgave.\n"
                "- Brug KUN antagelser, der er eksplicit givet i DEN AKTUELLE opgave.\n"
                "- Hvis en referenceopgave indfører en parameter p, betyder det IKKE, at p findes her.\n"
                "- Ved konflikt skal du ALTID følge DEN AKTUELLE opgave.\n\n"
                "Udled altid din løsning selvstændigt og verificer dine beregninger.\n\n"
                "KONVENTION FOR MØNTKAST I DISSE EKSAMENER:\n"
                "- Hvis en opgave blot siger \"en mønt\" og IKKE definerer p, antag en fair mønt: p = 1/2.\n"
                "- Behandl kun p som ukendt, hvis den aktuelle opgave eksplicit indfører p.\n\n"
            )
        # 添加参考问题
        if retrieved_questions:
            prompt_parts.append("=== REFERENCE PROBLEMS ===\n\n")
            
            for i, q in enumerate(retrieved_questions, 1):
                prompt_parts.append(f"Reference {i}:\n")
                
                # Context
                if q.context:
                    prompt_parts.append(f"Context: {q.context}\n")
                
                # Question
                prompt_parts.append(f"Question: {q.question}\n")
                
                # Correct answer (1-indexed)
                prompt_parts.append(f"Correct Answer: Option {q.answer_index}\n")
                
                # Key steps
                if q.explanation_key_steps:
                    prompt_parts.append("\nKey steps:\n")
                    for step in q.explanation_key_steps:
                        prompt_parts.append(f"  - {step}\n")
                
                # Formulae
                if q.explanation_formulae:
                    prompt_parts.append("\nKey formulae:\n")
                    for formula in q.explanation_formulae:
                        prompt_parts.append(f"  - {formula}\n")
                
                # Pitfalls
                if q.explanation_pitfalls:
                    prompt_parts.append("\nCommon pitfalls to avoid:\n")
                    for pitfall in q.explanation_pitfalls:
                        prompt_parts.append(f" {pitfall}\n")
                
                prompt_parts.append("\n")
        
        # 添加分隔符
        prompt_parts.append("=== PROBLEM TO SOLVE ===\n\n")
        
        # 添加当前问题的context
        if lang_data['context']:
            prompt_parts.append(f"Context: {lang_data['context']}\n")
        
        # 添加问题
        prompt_parts.append(f"Question: {lang_data['question']}\n\n")
        
        # 添加选项
        prompt_parts.append("Options:\n")
        for i, opt in enumerate(lang_data['options'], 1):
            prompt_parts.append(f"{i}. {opt}\n")
        
        # 添加输出格式要求
        if language == "english":
            prompt_parts.append(
                """IMPORTANT NOTES:
- Mathematical expressions may be written in different equivalent forms
- For example: sqrt(a^2 * b) equals a * sqrt(b)
- Expression order doesn't matter: e^(-A) * Σ(...) equals Σ(...) * e^(-A)
- Consider algebraic equivalence when comparing your answer to the options
- If your derived expression matches an option algebraically, select that option

Please respond with your reasoning followed by a JSON object in this exact format:
{{"answer": N}}

where N is the number of your chosen option (1, 2, 3, 4, 5, or 6).
Option 6 ("Ved ikke" / "Do not know") should ONLY be selected if:
- The problem is genuinely unsolvable with given information, OR
- All options are clearly incorrect, OR  
- You cannot derive any reasonable answer

Your response:"""
            )
        else:
            prompt_parts.append(
                """VIGTIGE NOTER:
- Matematiske udtryk kan være skrevet i forskellige ækvivalente former
- For eksempel: sqrt(a^2 * b) er lig med a * sqrt(b)
- Udtrykrækkefølge betyder ikke noget: e^(-A) * Σ(...) er lig med Σ(...) * e^(-A)
- Overvej algebraisk ækvivalens når du sammenligner dit svar med valgmulighederne

Svar venligst med din ræsonnering efterfulgt af et JSON-objekt i dette nøjagtige format:
{{"answer": N}}

hvor N er nummeret på dit valgte svarmulighed (1, 2, 3, 4, 5 eller 6).
Valgmulighed 6 ("Ved ikke") skal KUN vælges hvis problemet er ægte uløseligt.

Dit svar:"""
            )
        
        return ''.join(prompt_parts)        
    
    def generate_answer(self, prompt: str, max_tokens: int = 2000) -> Dict:
        """
        生成答案（推荐实现：先完整/短片段生成，再检测重复并决定是否跳过）。
        说明：此实现避免试图强制中断 model.generate，从而不会留下占用 GPU 的后台线程。
        """
        print(f"\nprompt长度: {len(prompt)} 字符")
        messages = [{"role": "user", "content": prompt}]
        
        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=self.is_thinking
        )
        
        inputs = self.tokenizer([text], return_tensors="pt").to(self.model.device)
        # 生成配置（可按需微调）
        gen_config = {
            "max_new_tokens": max_tokens,
            "pad_token_id": self.tokenizer.pad_token_id,
            "eos_token_id": self.tokenizer.eos_token_id,
            "temperature": 0,
            "do_sample": True if (self.temperature and self.temperature > 0) else False,
        }
        
        start_time = time.time()
        try:
            with torch.no_grad():
                outputs = self.model.generate(**inputs, **gen_config)
        except Exception as e:
            generation_time = time.time() - start_time
            return {
                'full_response': f'ERROR_GENERATION: {e}',
                'thinking_process': '',
                'predicted_answer': None,
                'generation_time': generation_time,
                'has_thinking': False,
                'has_json': False,
                'skip_reason': None
            }
        
        generation_time = time.time() - start_time
        # 解码完整生成（去掉输入部分）
        try:
            full_resp = self.tokenizer.decode(
                outputs[0][inputs["input_ids"].shape[1]:],
                skip_special_tokens=True
            )
        except Exception:
            full_resp = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # 标准化用于重复检测：collapse whitespace, lowercase, 去掉不可见字符
        norm = re.sub(r'\s+', ' ', full_resp).strip().lower()
        
        # 简单稳健的重复检测：寻找某个子串在末尾连续重复若干次
        # 我们尝试识别长度在 8..64 的片段是否连续重复 >= threshold 次
        repetition_threshold = 5
        skipped = False
        repeat_info = None
        
        # 只在长文本时做检测
        if len(norm) >= 50:
            for length in range(8, min(64, len(norm)//2 + 1)):
                # 获取最近的 length * (threshold) 长度片段
                tail = norm[-(length * (repetition_threshold + 1)):] if len(norm) >= length * (repetition_threshold + 1) else norm
                # 如果尾部太短，跳过该 length
                if len(tail) < length * 2:
                    continue
                # 检查是否存在连续重复
                chunks = [tail[i:i+length] for i in range(0, len(tail), length)]
                # 找到最后连续重复片段次数
                # 从尾部往前数同一片段重复次数
                last_chunk = chunks[-1]
                cnt = 1
                for ch in reversed(chunks[:-1]):
                    if ch == last_chunk:
                        cnt += 1
                    else:
                        break
                if cnt >= repetition_threshold:
                    skipped = True
                    repeat_info = {
                        'repeat_chunk': last_chunk[:100],
                        'repeat_length': length,
                        'repeat_count': cnt
                    }
                    break
        
        if skipped:
            return {
                'full_response': 'SKIPPED_DUE_TO_REPETITION',
                'thinking_process': '',
                'predicted_answer': None,
                'generation_time': generation_time,
                'has_thinking': False,
                'has_json': False,
                'skip_reason': f"detected_repetition: chunk_len={repeat_info['repeat_length']}, count={repeat_info['repeat_count']}"
            }
        
        # 否则交由原有解析器解析
        return self._parse_response(full_resp, generation_time)

    def _parse_response(self, response: str, gen_time: float) -> Dict:
        """解析模型响应（优先 JSON，备选其他明确格式）"""
        result = {
            'full_response': response,
            'thinking_process': '',
            'predicted_answer': None,
            'generation_time': gen_time,
            'has_thinking': False,
            'has_json': False
        }
        
        # 提取思考过程（Thinking 模型）
        if self.is_thinking and '</think>' in response:
            result['has_thinking'] = True
            parts = response.split('</think>')
            if len(parts) >= 2:
                think_start = response.find('<think>')
                if think_start != -1:
                    result['thinking_process'] = response[think_start:response.find('</think>') + 8]
                else:
                    result['thinking_process'] = parts[0]
                response = parts[-1].strip()
        
        # 方法1: JSON 格式（最优先，最可靠）
        json_patterns = [
            r'\{["\']?answer["\']?\s*:\s*(\d+)\s*\}',           # {"answer": 5}
            r'\{["\']?answer["\']?\s*:\s*["\'](\d+)["\']\s*\}', # {"answer": "5"}
        ]
        
        for pattern in json_patterns:
            matches = re.findall(pattern, response, re.IGNORECASE)
            if matches:
                try:
                    answer = int(matches[-1])
                    if 1 <= answer <= 6:  # 验证答案在有效范围
                        result['predicted_answer'] = answer
                        result['has_json'] = True
                        return result  # 找到 JSON 格式就立即返回
                except:
                    continue
        
        # 方法2: 明确的答案声明（备选）
        answer_patterns = [
            r'(?:final answer|my answer|the answer|answer)["\']?\s*(?:is|:)\s*["\']?(\d+)["\']?',
            r'(?:option|choice)["\']?\s*["\']?(\d+)["\']?(?:\s+is correct|\s+is the answer)',
            r'(?:I choose|I select|select|choose)["\']?\s*(?:option|choice)?\s*["\']?(\d+)["\']?',
            r'correct answer.*?(?:is|:)\s*["\']?(\d+)["\']?',
        ]
        
        # 优先在最后100字符中查找（答案通常在结尾）
        search_text = response[-100:] if len(response) > 100 else response
        
        for pattern in answer_patterns:
            matches = re.findall(pattern, search_text, re.IGNORECASE)
            if matches:
                try:
                    answer = int(matches[-1])
                    if 1 <= answer <= 6:  # 验证答案在有效范围
                        result['predicted_answer'] = answer
                        return result
                except:
                    continue
        
        return result
    
    def test_single_item(self, item: Dict, language: str = "english") -> Dict:
        """测试单个题目"""
        # 格式化问题
        prompt = self.format_question(item, language)
        
        # 生成答案
        result = self.generate_answer(prompt)
        
        # 获取正确答案
        correct_answer = int(item['answer_index'])
        predicted_answer = result['predicted_answer']
        
        # 判断正确性
        is_correct = predicted_answer == correct_answer if predicted_answer else False

        rag_meta = getattr(self, '_last_rag_meta', None) or {}
        rag_base_keys = rag_meta.get('retrieved_base_keys')
        rag_sims = rag_meta.get('retrieved_similarities')
        rag_idxs = rag_meta.get('retrieved_indices')
        
        return {
            'base_key': item['base_key'],
            'language': language,
            'context': item[language]['context'],
            'question': item[language]['question'],
            'correct_answer': correct_answer,
            'predicted_answer': predicted_answer,
            'is_correct': is_correct,
            'full_response': result['full_response'],
            'thinking_process': result['thinking_process'],
            'has_thinking': result['has_thinking'],
            'has_json': result['has_json'],
            'generation_time': result['generation_time'],
            'knowledge_points': ', '.join(item.get('knowledge_points', [])),
            'options': str(item[language].get('options', [])),
            'used_rag': self.use_rag, 
            'rag_k': self.rag_k if self.use_rag else None, 
            'rag_retrieved_base_keys': json.dumps(rag_base_keys) if rag_base_keys is not None else None,
            'rag_retrieved_similarities': json.dumps(rag_sims) if rag_sims is not None else None,
            'rag_retrieved_indices': json.dumps(rag_idxs) if rag_idxs is not None else None,
        }
    
    def test_dataset(self, 
                    data: List[Dict], 
                    languages: List[str] = ['english'],
                    limit: int = None,
                    checkpoint_dir: str = './checkpoints',
                    checkpoint_interval: int = 10,
                    resume: bool = False,
                    realtime_csv: bool = True,  # 新增：是否实时写CSV
                    output_dir: str = './results') -> pd.DataFrame:
        """
        在数据集上运行测试（支持checkpoint、续传和实时CSV）
        """
        if limit:
            data = data[:limit]
        
        # 创建目录
        os.makedirs(checkpoint_dir, exist_ok=True)
        os.makedirs(output_dir, exist_ok=True)
        
        # CSV文件路径
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_short_name = self.model_name.split('/')[-1]
        rag_suffix = f"_rag_k{self.rag_k}" if self.use_rag else "_baseline"
        realtime_csv_file = None
        
        if realtime_csv:
            realtime_csv_file = f"{output_dir}/{model_short_name}_realtime_{timestamp}.csv"
            print(f"实时CSV文件: {realtime_csv_file}")
        
        # Checkpoint文件
        checkpoint_file = os.path.join(
            checkpoint_dir, 
            f"{model_short_name}_checkpoint.pkl"
        )
        
        # 恢复checkpoint
        results = []
        completed_items = set()
        csv_exists = False
        
        if resume and os.path.exists(checkpoint_file):
            try:
                with open(checkpoint_file, 'rb') as f:
                    checkpoint_data = pickle.load(f)
                    results = checkpoint_data['results']
                    completed_items = checkpoint_data['completed_items']
                    print(f"✓ 从checkpoint恢复，已完成: {len(completed_items)} 个测试")
                    
                    # 如果恢复，将已有结果写入新的CSV
                    if realtime_csv and results:
                        pd.DataFrame(results).to_csv(
                            realtime_csv_file, 
                            index=False, 
                            encoding='utf-8-sig'
                        )
                        csv_exists = True
                        
            except Exception as e:
                print(f"加载checkpoint失败: {e}")
        
        # 测试主循环
        total_tests = len(data) * len(languages)
        remaining_tests = total_tests - len(completed_items)
        
        print(f"\n开始测试 (共 {total_tests} 个，剩余 {remaining_tests} 个)")
        
        try:
            with tqdm(total=remaining_tests, desc="测试进度") as pbar:
                for item in data:
                    for lang in languages:
                        test_id = f"{item['base_key']}_{lang}"
                        if test_id in completed_items:
                            continue
                        
                        try:
                            # 执行测试
                            result = self.test_single_item(item, lang)
                            results.append(result)
                            completed_items.add(test_id)
                            
                            # 实时写入CSV（追加模式）
                            if realtime_csv and realtime_csv_file:
                                result_df = pd.DataFrame([result])
                                result_df.to_csv(
                                    realtime_csv_file,
                                    mode='a',  # 追加
                                    header=not csv_exists,  # 只在首次写header
                                    index=False,
                                    encoding='utf-8-sig'
                                )
                                csv_exists = True
                            
                            # 定期保存checkpoint
                            if len(results) % checkpoint_interval == 0:
                                self._save_checkpoint(
                                    checkpoint_file, 
                                    results, 
                                    completed_items
                                )
                                if realtime_csv:
                                    print(f"\nCheckpoint保存 + CSV已更新 ({len(results)}/{total_tests})")
                                else:
                                    print(f"\nCheckpoint保存 ({len(results)}/{total_tests})")
                        
                        except KeyboardInterrupt:
                            print("\n\n检测到中断...")
                            self._save_checkpoint(checkpoint_file, results, completed_items)
                            print(f"✓ 进度已保存")
                            if realtime_csv:
                                print(f"✓ CSV文件: {realtime_csv_file}")
                            
                            df = pd.DataFrame(results)
                            return df
                        
                        except Exception as e:
                            print(f"\n错误: {test_id}: {str(e)}")
                            # 记录错误结果
                            error_result = {
                                'base_key': item['base_key'],
                                'language': lang,
                                'is_correct': False,
                                'full_response': f"ERROR: {str(e)}",
                                # ... 其他字段
                            }
                            results.append(error_result)
                            
                            # 错误也要写入CSV
                            if realtime_csv and realtime_csv_file:
                                pd.DataFrame([error_result]).to_csv(
                                    realtime_csv_file,
                                    mode='a',
                                    header=not csv_exists,
                                    index=False,
                                    encoding='utf-8-sig'
                                )
                                csv_exists = True
                        
                        pbar.update(1)
            
            # 测试完成
            print(f"\nTest Done！")
            if realtime_csv:
                print(f"实时结果已保存在: {realtime_csv_file}")
            
            # 删除checkpoint
            if len(completed_items) == total_tests and os.path.exists(checkpoint_file):
                os.remove(checkpoint_file)
                print(f"✓ Checkpoint已清理")
        
        except Exception as e:
            print(f"\nTest error: {e}")
            self._save_checkpoint(checkpoint_file, results, completed_items)
        
        df = pd.DataFrame(results)
        return df
    
    def _save_checkpoint(self, checkpoint_file: str, results: List, completed_items: set):
        """保存checkpoint"""
        try:
            with open(checkpoint_file, 'wb') as f:
                pickle.dump({
                    'results': results,
                    'completed_items': completed_items,
                    'timestamp': datetime.now().isoformat()
                }, f)
        except Exception as e:
            print(f"Failed to save checkpoint: {e}")
    
    def _print_statistics(self, df: pd.DataFrame):
        """打印统计信息"""
        print("\n" + "=" * 80)
        print("测试统计:")
        print("=" * 80)
        
        # 总体统计
        print(f"\n总测试数: {len(df)}")
        print(f"正确数: {df['is_correct'].sum()}")
        print(f"准确率: {df['is_correct'].mean()*100:.2f}%")
        print(f"平均耗时: {df['generation_time'].mean():.2f} 秒")
        print(f"总耗时: {df['generation_time'].sum():.1f} 秒 ({df['generation_time'].sum()/60:.1f} 分钟)")
        
        if self.is_thinking:
            has_thinking = df['has_thinking'].sum()
            print(f"包含思考过程: {has_thinking} ({has_thinking/len(df)*100:.1f}%)")
        
        # JSON 格式使用统计
        has_json = df['has_json'].sum()
        print(f"使用 JSON 格式: {has_json} ({has_json/len(df)*100:.1f}%)")
        
        # 成功预测统计
        predicted = df['predicted_answer'].notna().sum()
        print(f"成功提取答案: {predicted} ({predicted/len(df)*100:.1f}%)")
        
        # 按语言统计
        if len(df['language'].unique()) > 1:
            print("\n按语言统计:")
            lang_stats = df.groupby('language').agg({
                'is_correct': ['count', 'sum', 'mean'],
                'generation_time': 'mean',
                'has_json': 'sum'
            }).round(3)
            lang_stats.columns = ['题目数', '正确数', '准确率', '平均耗时(秒)', 'JSON格式数']
            lang_stats['准确率'] = (lang_stats['准确率'] * 100).round(2)
            print(lang_stats)
    
    def _analyze_errors(self, df: pd.DataFrame):
        """分析错误样本"""
        errors = df[df['is_correct'] == False].copy()
        
        
        print(f"\n{'='*80}")
        print(f"错误分析 (共 {len(errors)} 个错误，错误率 {len(errors)/len(df)*100:.2f}%)")
        print(f"{'='*80}")
        
        # 错误原因分类
        no_answer = errors['predicted_answer'].isna().sum()
        wrong_answer = len(errors) - no_answer
        
        print(f"\n错误类型:")
        print(f"  ❌ 未能提取答案: {no_answer} ({no_answer/len(errors)*100:.1f}%)")
        print(f"  ❌ 答案错误: {wrong_answer} ({wrong_answer/len(errors)*100:.1f}%)")


def load_jsonl_data(file_path: str) -> List[Dict]:
    """加载 JSONL 格式的数据"""
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                data.append(json.loads(line))
    
    print(f"✓ 成功加载 {len(data)} 道题目")
    return data


def compare_models(data: List[Dict],
                   model_configs: List[Dict],
                   languages: List[str] = ['english'],
                   output_dir: str = './results',
                   limit: int = None,
                   checkpoint_dir: str = './checkpoints',
                   resume: bool = False):
    """
    对比多个模型（支持checkpoint）
    
    Args:
        data: 测试数据
        model_configs: 模型配置列表
        languages: 测试语言列表
        output_dir: 结果保存目录
        limit: 限制测试数量
        checkpoint_dir: checkpoint保存目录
        resume: 是否从checkpoint恢复
    """
    os.makedirs(output_dir, exist_ok=True)
    
    all_results = {}
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    for config in model_configs:
        model_name = config['name']
        use_quant = config.get('use_quantization', False)
        quant_bits = config.get('quantization_bits', 4)
        local_path = config.get('local_path', None)
        temperature = config.get('temperature', 0.0)
        use_rag = config.get('use_rag', False)
        rag_system = config.get('rag_system', None)
        rag_k = config.get('rag_k', 3)
        
        print(f"\n{'='*80}")
        print(f"测试模型: {model_name}")
        if use_quant:
            print(f"量化配置: {quant_bits}-bit")
        if local_path:
            print(f"本地路径: {local_path}")
        print(f"温度: {temperature}")
        print(f"{'='*80}\n")
        
        try:
            # 创建测试器
            tester = ProbabilityTestFramework(
                model_name,
                use_quantization=use_quant,
                quantization_bits=quant_bits,
                local_model_path=local_path,
                temperature=temperature,
                use_rag=use_rag,
                rag_system=rag_system,
                rag_k=rag_k
            )
            
            # 运行测试（支持checkpoint）
            results_df = tester.test_dataset(
                data, 
                languages, 
                limit,
                checkpoint_dir=checkpoint_dir,
                checkpoint_interval=10,  # 每10题保存一次
                resume=resume
            )
            
            # 保存详细结果
            model_short_name = model_name.split('/')[-1]
            quant_suffix = f"_q{quant_bits}" if use_quant else ""
            rag_suffix = f"_rag_k{rag_k}" if use_rag else "_baseline"
            temp_suffix = f"_t{temperature}".replace('.', '')
            output_file = f"{output_dir}/{model_short_name}{quant_suffix}{temp_suffix}_{timestamp}.csv"
            results_df.to_csv(output_file, index=False, encoding='utf-8-sig')
            print(f"\n✓ 结果已保存到: {output_file}")
            
            all_results[f"{model_name}{quant_suffix}"] = results_df
            
            # 清理显存
            del tester
            torch.cuda.empty_cache()
            
        except Exception as e:
            print(f"\n✗ 模型 {model_name} 测试失败: {str(e)}")
            import traceback
            traceback.print_exc()
    
    # 生成对比报告
    if len(all_results) > 1:
        generate_comparison_report(all_results, output_dir, timestamp)
    
    return all_results


def generate_comparison_report(all_results: Dict[str, pd.DataFrame],
                               output_dir: str,
                               timestamp: str):
    """生成对比报告"""
    print("\n" + "="*80)
    print("模型对比报告")
    print("="*80)
    
    comparison_data = []
    
    for model_name, df in all_results.items():
        comparison_data.append({
            '模型': model_name.split('/')[-1],
            '题目数': len(df),
            '准确率(%)': f"{df['is_correct'].mean()*100:.2f}",
            '平均耗时(秒)': f"{df['generation_time'].mean():.2f}",
            '总耗时(分钟)': f"{df['generation_time'].sum()/60:.1f}",
            '成功预测数': df['predicted_answer'].notna().sum(),
            'JSON格式数': df['has_json'].sum(),
        })
    
    comparison_df = pd.DataFrame(comparison_data)
    print("\n", comparison_df.to_string(index=False))
    
    # 保存对比报告
    report_file = f"{output_dir}/comparison_report_{timestamp}.csv"
    comparison_df.to_csv(report_file, index=False, encoding='utf-8-sig')
    print(f"\n✓ 对比报告已保存到: {report_file}")


# ============================================================================
# 主程序
# ============================================================================

if __name__ == "__main__":
    print("="*80)
    print("概率论题库测试框架 v3.0")
    print("✅ 支持checkpoint和续传")
    print("✅ 支持本地模型加载")
    print("✅ 支持温度0（确定性输出）")
    print("="*80)
    
    # 1. 加载测试数据
    data = load_jsonl_data('probability_test_set.jsonl')
    
    # 2. 定义测试配置
    QUICK_TEST = False  # 快速测试模式
    test_limit = 10 if QUICK_TEST else None
    
    # 要测试的语言
    test_languages = ['english', 'danish']
    
    # 要测试的模型配置
    model_configs = [
        # 本地量化模型（优先使用）
        {
            'name': 'Qwen/Qwen3-14B',
            'local_path': '/root/models/qwen3-14b-q4',  # 本地路径
            'use_quantization': True,
            'quantization_bits': 4,
            'timeout': 120,  # 每题最多120秒
            'temperature': 0.0  # 确定性输出
        },
        
        # 如果需要测试其他模型，添加在这里
        # {
        #     'name': 'Qwen/Qwen3-4B-Thinking-2507',
        #     'local_path': '/root/models/qwen3-4b-thinking',
        #     'use_quantization': False,
        #     'temperature': 0.0
        # },
    ]
    
    # 3. 运行对比测试
    print(f"\n配置:")
    print(f"  测试模式: {'快速测试 (前' + str(test_limit) + '题)' if QUICK_TEST else '完整测试'}")
    print(f"  测试语言: {', '.join(test_languages)}")
    print(f"  模型数量: {len(model_configs)}")
    print(f"  Checkpoint: 启用（每10题保存）")
    print(f"  支持中断: Ctrl+C 安全中断并保存进度")
    
    results = compare_models(
        data=data,
        model_configs=model_configs,
        languages=test_languages,
        output_dir='./probability_test_results',
        checkpoint_dir='./checkpoints',
        limit=test_limit,
        resume=False  # 启用续传
    )
    
    print("\n" + "="*80)
    print("所有测试完成！")
    print("="*80)