"""
支持微调模型（LoRA）的测试框架
在原有基础上增加了 LoRA 适配器加载功能
已更新 prompt 格式为第二份代码的简洁版本
"""

import json
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel  # 🔧 新增：用于加载 LoRA 模型
from typing import List, Dict
import pandas as pd
from tqdm import tqdm
import time
from datetime import datetime
import os
import re

class ProbabilityTestFramework:
    def __init__(self, 
                 model_name: str,
                 device: str = "cuda",
                 use_quantization: bool = False,
                 quantization_bits: int = 4,
                 local_model_path: str = None,
                 lora_adapter_path: str = None,  # 🔧 新增：LoRA 适配器路径
                 temperature: float = 0.0):
        """
        初始化测试框架
        
        Args:
            model_name: 基础模型名称
            device: 运行设备
            use_quantization: 是否使用量化
            quantization_bits: 量化位数 (4 or 8)
            local_model_path: 本地基础模型路径
            lora_adapter_path: LoRA 适配器路径（微调输出目录）🆕
            temperature: 生成温度
        """
        self.temperature = temperature
        self.model_name = model_name
        self.device = device
        self.use_quantization = use_quantization
        self.lora_adapter_path = lora_adapter_path
        
        # 决定基础模型路径
        actual_model_path = model_name
        if local_model_path and os.path.exists(local_model_path):
            actual_model_path = local_model_path
            print(f"✓ 基础模型: {local_model_path}")
        else:
            print(f"✓ 基础模型: {model_name} (从 HF)")
        
        # 🔧 显示 LoRA 信息
        if lora_adapter_path:
            if os.path.exists(lora_adapter_path):
                print(f"✓ LoRA 适配器: {lora_adapter_path}")
            else:
                print(f"⚠️ LoRA 路径不存在: {lora_adapter_path}")
                print(f"   将只加载基础模型")
                self.lora_adapter_path = None
        
        print(f"  量化: {'是 (' + str(quantization_bits) + '-bit)' if use_quantization else '否 (BF16)'}")
        print(f"  温度: {temperature} ({'确定性' if temperature == 0 else '随机性'})")
        
        # 加载分词器
        self.tokenizer = AutoTokenizer.from_pretrained(
            actual_model_path,
            use_fast=False,
            trust_remote_code=True
        )
        
        # 配置量化参数
        if use_quantization:
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
            
            # 加载量化的基础模型
            base_model = AutoModelForCausalLM.from_pretrained(
                actual_model_path,
                device_map="auto",
                torch_dtype=torch.bfloat16,
                quantization_config=quant_config,
                trust_remote_code=True,
            )
        else:
            # 加载 BF16 基础模型
            base_model = AutoModelForCausalLM.from_pretrained(
                actual_model_path,
                device_map="auto",
                torch_dtype=torch.bfloat16,
                trust_remote_code=True
            )
        
        # 🔧 加载 LoRA 适配器（如果有）
        if self.lora_adapter_path and os.path.exists(self.lora_adapter_path):
            print(f"🔧 加载 LoRA 适配器...")
            try:
                self.model = PeftModel.from_pretrained(
                    base_model,
                    self.lora_adapter_path,
                    is_trainable=False  # 推理模式
                )
                print(f"✓ LoRA 适配器加载成功（微调模型）")
                self.is_finetuned = True
            except Exception as e:
                print(f"⚠️ LoRA 加载失败: {e}")
                print(f"   使用基础模型")
                self.model = base_model
                self.is_finetuned = False
        else:
            self.model = base_model
            self.is_finetuned = False
        
        # 判断模型类型
        self.is_thinking = "thinking" in model_name.lower()
        
        model_type = "微调模型" if self.is_finetuned else "基础模型"
        print(f"✓ 模型加载完成! 类型: {model_type}")

    def format_question(self, item: Dict, language: str = "english") -> str:
        """
        🔧 更新：使用第二份代码中的简洁 prompt 格式
        格式化问题为 prompt（使用 JSON 格式输出，数字答案）
        """
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

    def generate_answer(self, prompt: str, max_tokens: int = 16000) -> Dict:
        """生成答案"""
        messages = [{"role": "user", "content": prompt}]
        
        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=self.is_thinking
        )
        
        inputs = self.tokenizer([text], return_tensors="pt").to(self.model.device)
        
        gen_config = {
            "max_new_tokens": max_tokens,
            "pad_token_id": self.tokenizer.pad_token_id,
            "eos_token_id": self.tokenizer.eos_token_id,
            "temperature": self.temperature,
            "do_sample": True if self.temperature > 0 else False,
        }
        
        start_time = time.time()
        try:
            with torch.no_grad():
                outputs = self.model.generate(**inputs, **gen_config)
        except Exception as e:
            generation_time = time.time() - start_time
            return {
                'full_response': f'ERROR: {e}',
                'predicted_answer': None,
                'generation_time': generation_time,
                'has_json': False
            }
        
        generation_time = time.time() - start_time
        
        try:
            full_resp = self.tokenizer.decode(
                outputs[0][inputs["input_ids"].shape[1]:],
                skip_special_tokens=True
            )
        except:
            full_resp = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        return self._parse_response(full_resp, generation_time)

    def _parse_response(self, response: str, gen_time: float) -> Dict:
        """解析响应，提取答案（优先 JSON 格式）"""
        result = {
            'full_response': response,
            'predicted_answer': None,
            'generation_time': gen_time,
            'has_json': False
        }
        
        # 方法1: JSON 格式（最优先，最可靠）
        json_patterns = [
            r'\{["\']?answer["\']?\s*:\s*(\d+)\s*\}',           # {"answer": 5}
            r'\{["\']?answer["\']?\s*:\s*["\'](\d+)["\']\s*\}', # {"answer": "5"}
            r'["\']?answer["\']?\s*[:=]\s*(\d+)',                # answer: 5
        ]
        
        for pattern in json_patterns:
            matches = re.findall(pattern, response, re.IGNORECASE)
            if matches:
                try:
                    answer = int(matches[-1])
                    if 1 <= answer <= 6:
                        result['predicted_answer'] = answer
                        result['has_json'] = True
                        return result
                except:
                    continue
        
        # 方法2: 明确答案声明
        answer_patterns = [
            r'(?:final answer|my answer|the answer|answer)["\']?\s*(?:is|:)\s*["\']?(?:option\s*)?(\d+)["\']?',
            r'(?:option|choice)["\']?\s*["\']?(\d+)["\']?(?:\s+is correct|\s+is the answer)',
            r'(?:I choose|I select|select|choose)["\']?\s*(?:option|choice)?\s*["\']?(\d+)["\']?',
            r'correct answer.*?(?:is|:)\s*["\']?(?:option\s*)?(\d+)["\']?',
            r'therefore.*?answer.*?(?:is|:)\s*["\']?(\d+)["\']?',
        ]
        
        search_text = response[-200:] if len(response) > 200 else response
        
        for pattern in answer_patterns:
            matches = re.findall(pattern, search_text, re.IGNORECASE)
            if matches:
                try:
                    answer = int(matches[-1])
                    if 1 <= answer <= 6:
                        result['predicted_answer'] = answer
                        return result
                except:
                    continue
        
        # 方法3: 最后尝试查找数字
        last_number = re.findall(r'\b([1-6])\b', response[-100:] if len(response) > 100 else response)
        if last_number:
            try:
                result['predicted_answer'] = int(last_number[-1])
            except:
                pass
        
        return result

    def test_single_item(self, item: Dict, language: str = "english") -> Dict:
        """测试单个题目"""
        prompt = self.format_question(item, language)
        result = self.generate_answer(prompt)
        
        correct_answer = int(item['answer_index'])
        predicted_answer = result['predicted_answer']
        is_correct = predicted_answer == correct_answer if predicted_answer else False
        
        return {
            'base_key': item['base_key'],
            'language': language,
            'correct_answer': correct_answer,
            'predicted_answer': predicted_answer,
            'is_correct': is_correct,
            'full_response': result['full_response'][:500],  # 截断保存
            'has_json': result['has_json'],
            'generation_time': result['generation_time'],
            'model_type': 'finetuned' if self.is_finetuned else 'baseline'  # 🔧 标记模型类型
        }

    def test_dataset(self, 
                    data: List[Dict], 
                    languages: List[str] = ['english'],
                    limit: int = None) -> pd.DataFrame:
        """在数据集上运行测试"""
        if limit:
            data = data[:limit]
        
        results = []
        total = len(data) * len(languages)
        
        print(f"\n开始测试 (共 {total} 题)")
        
        with tqdm(total=total, desc="测试进度") as pbar:
            for item in data:
                for lang in languages:
                    try:
                        result = self.test_single_item(item, lang)
                        results.append(result)
                    except Exception as e:
                        print(f"\n错误: {item['base_key']}_{lang}: {e}")
                        results.append({
                            'base_key': item['base_key'],
                            'language': lang,
                            'is_correct': False,
                            'full_response': f"ERROR: {e}",
                            'model_type': 'finetuned' if self.is_finetuned else 'baseline'
                        })
                    pbar.update(1)
        
        df = pd.DataFrame(results)
        
        # 打印统计
        self._print_statistics(df)
        
        return df

    def _print_statistics(self, df: pd.DataFrame):
        """打印统计信息"""
        print("\n" + "=" * 80)
        print(f"测试统计 ({'微调模型' if self.is_finetuned else '基础模型'})")
        print("=" * 80)
        print(f"总题数: {len(df)}")
        print(f"正确数: {df['is_correct'].sum()}")
        print(f"准确率: {df['is_correct'].mean()*100:.2f}%")
        print(f"成功提取答案: {df['predicted_answer'].notna().sum()}/{len(df)}")
        print(f"使用 JSON 格式: {df['has_json'].sum()}/{len(df)}")
        print(f"平均耗时: {df['generation_time'].mean():.2f} 秒")
        print("=" * 80)


def load_jsonl_data(file_path: str) -> List[Dict]:
    """加载 JSONL 数据"""
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                data.append(json.loads(line))
    print(f"✓ 加载 {len(data)} 道题目")
    return data


def compare_baseline_vs_finetuned(
    test_data_path: str,
    base_model_path: str,
    lora_adapter_path: str,
    languages: List[str] = ['english'],
    limit: int = None,
    output_dir: str = './comparison_results'
):
    """
    对比基础模型和微调模型
    
    Args:
        test_data_path: 测试数据路径
        base_model_path: 基础模型路径
        lora_adapter_path: LoRA 适配器路径
        languages: 测试语言
        limit: 限制测试数量
        output_dir: 结果保存目录
    """
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # 加载测试数据
    print("="*80)
    print("加载测试数据")
    print("="*80)
    test_data = load_jsonl_data(test_data_path)
    
    # ========================================
    # 1. 测试基础模型
    # ========================================
    # print("\n" + "="*80)
    # print("测试基础模型")
    # print("="*80)
    
    # baseline_tester = ProbabilityTestFramework(
    #     model_name="Qwen/Qwen3-14B",
    #     local_model_path=base_model_path,
    #     use_quantization=True,
    #     quantization_bits=4,
    #     lora_adapter_path=None,  # 不加载 LoRA
    #     temperature=0.0
    # )
    
    # baseline_results = baseline_tester.test_dataset(
    #     test_data,
    #     languages=languages,
    #     limit=limit
    # )
    
    # # 保存基础模型结果
    # baseline_file = f"{output_dir}/baseline_{timestamp}.csv"
    # baseline_results.to_csv(baseline_file, index=False, encoding='utf-8-sig')
    # print(f"\n✓ 基础模型结果: {baseline_file}")
    
    # # 清理显存
    # del baseline_tester
    # torch.cuda.empty_cache()
    
    # ========================================
    # 2. 测试微调模型
    # ========================================
    print("\n" + "="*80)
    print("测试微调模型")
    print("="*80)
    
    finetuned_tester = ProbabilityTestFramework(
        model_name="Qwen/Qwen3-14B",
        local_model_path=base_model_path,
        use_quantization=True,
        quantization_bits=4,
        lora_adapter_path=lora_adapter_path,  # 加载 LoRA
        temperature=0.0
    )
    
    finetuned_results = finetuned_tester.test_dataset(
        test_data,
        languages=languages,
        limit=limit
    )
    
    # 保存微调模型结果
    finetuned_file = f"{output_dir}/finetuned_{timestamp}.csv"
    finetuned_results.to_csv(finetuned_file, index=False, encoding='utf-8-sig')
    print(f"\n✓ 微调模型结果: {finetuned_file}")
    
    del finetuned_tester
    torch.cuda.empty_cache()
    
    # ========================================
    # 3. 生成对比报告
    # ========================================
    print("\n" + "="*80)
    print("对比报告")
    print("="*80)
    
    comparison = pd.DataFrame([
        {
            '模型': '基础模型 (Qwen3-14B)',
            '准确率': f"{baseline_results['is_correct'].mean()*100:.2f}%",
            '正确数': baseline_results['is_correct'].sum(),
            '总题数': len(baseline_results),
            '提取成功率': f"{baseline_results['predicted_answer'].notna().mean()*100:.2f}%",
            'JSON 格式率': f"{baseline_results['has_json'].mean()*100:.2f}%"
        },
        {
            '模型': '微调模型 (LoRA)',
            '准确率': f"{finetuned_results['is_correct'].mean()*100:.2f}%",
            '正确数': finetuned_results['is_correct'].sum(),
            '总题数': len(finetuned_results),
            '提取成功率': f"{finetuned_results['predicted_answer'].notna().mean()*100:.2f}%",
            'JSON 格式率': f"{finetuned_results['has_json'].mean()*100:.2f}%"
        }
    ])
    
    print("\n", comparison.to_string(index=False))
    
    # 计算提升
    baseline_acc = baseline_results['is_correct'].mean()
    finetuned_acc = finetuned_results['is_correct'].mean()
    improvement = (finetuned_acc - baseline_acc) * 100
    
    print(f"\n📈 准确率提升: {improvement:+.2f} 个百分点")
    
    if improvement > 0:
        print(f"✅ 微调有效！准确率从 {baseline_acc*100:.2f}% 提升到 {finetuned_acc*100:.2f}%")
    elif improvement < 0:
        print(f"⚠️ 微调后准确率下降 {abs(improvement):.2f} 个百分点")
    else:
        print("准确率无变化")
    
    # 保存对比报告
    report_file = f"{output_dir}/comparison_report_{timestamp}.csv"
    comparison.to_csv(report_file, index=False, encoding='utf-8-sig')
    print(f"\n✓ 对比报告: {report_file}")
    
    print("\n" + "="*80)
    print("对比完成！")
    print("="*80)
    
    return baseline_results, finetuned_results


# ============================================================================
# 主程序
# ============================================================================

if __name__ == "__main__":
    print("="*80)
    print("概率论模型测试 - 支持微调模型对比")
    print("🔧 已更新：使用简洁 prompt 格式（JSON 输出）")
    print("="*80)
    
    # 配置
    TEST_DATA = "probability_test_set.jsonl"  # 测试集
    BASE_MODEL = "/root/models/qwen3-14b-q4"  # 基础模型
    LORA_ADAPTER = "./qwen3-qlora-output/final_model"  # 微调后的 LoRA
    
    TEST_LANGUAGES = ['english', 'danish']  # 或 ['english', 'danish']
    TEST_LIMIT = None  # None = 全部，10 = 快速测试
    
    # 运行对比
    baseline_df, finetuned_df = compare_baseline_vs_finetuned(
        test_data_path=TEST_DATA,
        base_model_path=BASE_MODEL,
        lora_adapter_path=LORA_ADAPTER,
        languages=TEST_LANGUAGES,
        limit=TEST_LIMIT,
        output_dir='./comparison_results'
    )
