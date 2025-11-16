

import json
import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import List, Dict, Any, Tuple, Optional
from sentence_transformers import SentenceTransformer
import faiss


# ============================================================
# 数据结构定义
# ============================================================

@dataclass
class ExamQuestion:
    """考试问题数据结构"""
    base_key: str
    context: str
    question: str
    options: List[str]
    answer_index: int
    knowledge_points: List[str]
    explanation_key_steps: List[str]
    explanation_formulae: List[str]
    explanation_pitfalls: List[str]
    language: str = 'english'


# ============================================================
# 数据加载
# ============================================================

class DataLoader:
    """加载JSONL格式的考试数据"""
    
    def __init__(self, data_file: str):
        self.data_file = data_file
        self.questions = []
        
    def load_data(self) -> List[ExamQuestion]:
        """加载JSONL格式的数据"""
        questions = []
        
        try:
            with open(self.data_file, 'r', encoding='utf-8') as f:
                for line_num, line in enumerate(f, 1):
                    try:
                        item = json.loads(line.strip())
                        
                        question = ExamQuestion(
                            base_key=item['base_key'],
                            context=item['english']['context'],
                            question=item['english']['question'],
                            options=item['english']['options'],
                            answer_index=item['answer_index'],
                            knowledge_points=item['knowledge_points'],
                            explanation_key_steps=item['explanation_key_steps'],
                            explanation_formulae=item['explanation_formulae'],
                            explanation_pitfalls=item['explanation_pitfalls']
                        )
                        questions.append(question)
                        
                    except json.JSONDecodeError as e:
                        print(f"Line {line_num} JSON error: {e}")
                    except KeyError as e:
                        print(f"Line {line_num} missing key: {e}")
                        
        except FileNotFoundError:
            print(f"Error: File '{self.data_file}' not found")
            return []
        
        self.questions = questions
        print(f"✓ Loaded {len(questions)} questions")
        return questions


# ============================================================
# 文本嵌入 - 升级版
# ============================================================

class TextEmbedder:
    """文本嵌入生成器 - 支持长文本 (8192 tokens)"""
    
    def __init__(self, model_name: str = 'nomic-ai/nomic-embed-text-v1.5'):
        """
        初始化嵌入模型
        """
        print(f"Loading embedding model: {model_name}...")
        
        # nomic模型需要trust_remote_code=True
        try:
            self.model = SentenceTransformer(model_name, trust_remote_code=True)
        except:
            self.model = SentenceTransformer(model_name)
        
        self.embedding_dim = self.model.get_sentence_embedding_dimension()
        self.max_seq_length = self.model.max_seq_length
          # 添加tokenizer用于计数
        self.tokenizer = self.model.tokenizer
        
        print(f"✓ Embedding model loaded")
        print(f"  Dimension: {self.embedding_dim}")
        print(f"  Max tokens: {self.max_seq_length}")
    def count_tokens(self, text: str) -> int:
        """统计文本的token数量"""
        try:
            return len(self.tokenizer.encode(text))
        except:
            # 如果tokenizer不可用，用粗略估计
            return len(text.split()) * 1.3  # 英文大约1.3 tokens per word
        
    def create_search_text(self, question: ExamQuestion) -> str:
        components = [
            question.context,
            question.question,
            ' '.join(question.knowledge_points)
        ]
        
        if question.explanation_formulae:
            components.append(' '.join(question.explanation_formulae))
        
        if question.explanation_key_steps:
            components.append(' '.join(question.explanation_key_steps[:5]))
        
        if question.explanation_pitfalls:
            components.append(' '.join(question.explanation_pitfalls[:3]))
        
        search_text = ' '.join(filter(None, components))
        
        token_count = self.count_tokens(search_text)
        if token_count > self.max_seq_length:
            print(f"   Warning: Text will be truncated!")
            print(f"   Question: {question.base_key}")
            print(f"   Tokens: {token_count} -> {self.max_seq_length} (truncated)")
                
        return search_text
    
    def embed_texts(self, texts: List[str], batch_size: int = 32) -> np.ndarray:
        """批量生成文本嵌入 - 添加统计信息"""
        print(f"Generating embeddings for {len(texts)} texts...")
        
        # 统计token长度
        token_counts = [self.count_tokens(text) for text in texts]
        max_tokens = max(token_counts)
        avg_tokens = sum(token_counts) / len(token_counts)
        truncated_count = sum(1 for t in token_counts if t > self.max_seq_length)
        
        print(f"  Token statistics:")
        print(f"    Max: {max_tokens} tokens")
        print(f"    Avg: {avg_tokens:.0f} tokens")
        print(f"    Truncated: {truncated_count}/{len(texts)} questions ({truncated_count/len(texts)*100:.1f}%)")
        
        if truncated_count > 0:
            print(f"  ⚠️ {truncated_count} questions will be truncated!")
        
        embeddings = self.model.encode(
            texts,
            batch_size=batch_size,
            normalize_embeddings=True,
            show_progress_bar=True
        )
        
        print(f"✓ Embeddings generated (shape: {embeddings.shape})")
        return embeddings
        
    def embed_single(self, text: str) -> np.ndarray:
        """生成单个文本的嵌入"""
        return self.model.encode(
            [text],
            normalize_embeddings=True
        )[0]


# ============================================================
# 向量数据库
# ============================================================

class VectorDatabase:
    """基于FAISS的向量数据库"""
    
    def __init__(self, embedding_dim: int = 768):
        self.embedding_dim = embedding_dim
        self.embeddings = None
        self.metadata = []
        self.index = None
        
    def add_questions(self, questions: List[ExamQuestion], embedder: TextEmbedder):
        """添加问题到向量数据库"""
        if not questions:
            print("Warning: No questions to add")
            return
            
        search_texts = []
        for q in questions:
            search_text = embedder.create_search_text(q)
            search_texts.append(search_text)
            
            self.metadata.append({
                'question': q,
                'search_text': search_text
            })
        
        self.embeddings = embedder.embed_texts(search_texts)
        self.embedding_dim = self.embeddings.shape[1]
        self._build_faiss_index()
        
        print(f"✓ Added {len(questions)} questions to vector database")
    
    def _build_faiss_index(self):
        """构建FAISS索引"""
        if self.embeddings is None or len(self.embeddings) == 0:
            raise ValueError("No embeddings available for building index")
            
        self.index = faiss.IndexFlatIP(self.embedding_dim)
        self.index.add(self.embeddings.astype('float32'))
        
        print(f"✓ FAISS index built with {self.index.ntotal} vectors")
    
    def search(self, query_embedding: np.ndarray, k: int = 5) -> Tuple[np.ndarray, List[int]]:
        """搜索最相似的k个向量"""
        if self.index is None:
            raise ValueError("Index not built. Call add_questions() first.")
            
        if query_embedding.ndim == 1:
            query_embedding = query_embedding.reshape(1, -1)
        
        k = min(k, self.index.ntotal)
        similarities, indices = self.index.search(
            query_embedding.astype('float32'), 
            k
        )
        
        return similarities[0], indices[0].tolist()


# ============================================================
# 多样性过滤
# ============================================================

class DiversityFilter:
    """确保检索结果的多样性"""
    
    @staticmethod
    def extract_base_key(key: str) -> str:
        """提取基础问题ID"""
        for suffix in ['_rephrase_', '_backward_']:
            if suffix in key:
                return key.split(suffix)[0]
        return key
    
    @staticmethod
    def filter_diverse_results(
        indices: List[int], 
        questions: List[ExamQuestion], 
        k: int = 3
    ) -> List[int]:
        """过滤结果以确保多样性"""
        selected_indices = []
        seen_base_keys = set()
        
        for idx in indices:
            if idx >= len(questions):
                continue
                
            question = questions[idx]
            base_key = DiversityFilter.extract_base_key(question.base_key)
            
            if base_key not in seen_base_keys:
                selected_indices.append(idx)
                seen_base_keys.add(base_key)
                
                if len(selected_indices) >= k:
                    break
        
        return selected_indices


# ============================================================
# 知识点权重
# ============================================================

class KnowledgePointWeighting:
    """基于训练准确率的知识点权重系统"""
    
    def __init__(self):
        self.weights = {}
        
    def compute_weights(self, accuracy_dict: Dict[str, float], alpha: float = 0.7):
        """根据准确率计算知识点权重"""
        for kp, accuracy in accuracy_dict.items():
            if accuracy > 1:
                accuracy = accuracy / 100.0
            
            accuracy = max(0.0, min(1.0, accuracy))
            self.weights[kp] = 1.0 + alpha * (1 - accuracy)
        
        print(f"✓ Computed weights for {len(self.weights)} knowledge points")
        
        if self.weights:
            sorted_kps = sorted(self.weights.items(), key=lambda x: x[1], reverse=True)
            print("\nTop-5 hardest knowledge points:")
            for kp, weight in sorted_kps[:5]:
                acc = accuracy_dict[kp]
                print(f"  • {kp}: weight={weight:.3f} (accuracy={acc:.1%})")
    
    def apply_weights(
        self, 
        similarities: np.ndarray, 
        questions: List[ExamQuestion],
        indices: List[int]
    ) -> List[Tuple[float, int]]:
        """应用知识点权重到搜索结果"""
        weighted_results = []
        
        for sim, idx in zip(similarities, indices):
            if idx >= len(questions):
                continue
                
            question = questions[idx]
            
            kp_weights = [self.weights.get(kp, 1.0) for kp in question.knowledge_points]
            weight = sum(kp_weights) / len(kp_weights) if kp_weights else 1.0

            
            weighted_score = sim * weight
            weighted_results.append((weighted_score, idx))
        
        weighted_results.sort(key=lambda x: x[0], reverse=True)
        return weighted_results
    
    def load_from_csv(self, csv_path: str, alpha: float = 0.7):
        """从CSV文件加载知识点准确率"""
        try:
            df = pd.read_csv(csv_path)
            
            if "accuracy" in df.columns:
                acc = df["accuracy"].astype(float)
            elif "accuracy_pct" in df.columns:
                acc = df["accuracy_pct"].astype(float) / 100.0
            else:
                raise ValueError("CSV must contain 'accuracy' or 'accuracy_pct' column")
            
            accuracy_dict = dict(zip(
                df["knowledge_point"].astype(str).str.strip(),
                acc.clip(0, 1).fillna(0.0)
            ))
            
            self.compute_weights(accuracy_dict, alpha)
            
        except FileNotFoundError:
            print(f"Warning: CSV file '{csv_path}' not found")
        except Exception as e:
            print(f"Error loading CSV: {e}")


# ============================================================
# 提示构建器
# ============================================================

class PromptBuilder:
    """构建增强的提示信息"""
    
    def build_prompt(
        self, 
        test_question: Dict[str, Any], 
        retrieved_questions: List[ExamQuestion]
    ) -> str:
        """构建包含检索到的相关问题的提示"""
        
        prompt_parts = [
            "You are an expert in probability and statistics. ",
            "Use the following similar problems as reference to solve the given problem.\n\n"
        ]
        
        if retrieved_questions:
            prompt_parts.append("=== REFERENCE PROBLEMS ===\n\n")
            
            for i, q in enumerate(retrieved_questions, 1):
                prompt_parts.append(f"Reference {i}:\n")
                if q.context:
                    prompt_parts.append(f"Context: {q.context}\n")
                prompt_parts.append(f"Question: {q.question}\n")
                prompt_parts.append(f"Answer: Option {q.answer_index + 1}\n")
                
                if q.explanation_key_steps:
                    prompt_parts.append("Key steps:\n")
                    for step in q.explanation_key_steps:
                        prompt_parts.append(f"  - {step}\n")
                
                if q.explanation_formulae:
                    prompt_parts.append("Key formulae:\n")
                    for formula in q.explanation_formulae:
                        prompt_parts.append(f"  - {formula}\n")
                
                prompt_parts.append("\n")
        
        prompt_parts.append("=== PROBLEM TO SOLVE ===\n\n")
        if test_question.get('context'):
            prompt_parts.append(f"Context: {test_question['context']}\n")
        prompt_parts.append(f"Question: {test_question['question']}\n\n")
        
        if 'options' in test_question:
            prompt_parts.append("Options:\n")
            for i, opt in enumerate(test_question['options'], 1):
                prompt_parts.append(f"{i}. {opt}\n")
        
        prompt_parts.append("\nProvide your answer and explanation.")
        
        return ''.join(prompt_parts)


# ============================================================
# 主RAG系统
# ============================================================

class ProbabilityRAG:
    """完整的概率统计RAG系统 - 升级版"""
    
    def __init__(
        self, 
        knowledge_base_path: str, 
        embedding_model: str = 'nomic-ai/nomic-embed-text-v1.5'
    ):
        """
        初始化RAG系统
        
        Args:
            knowledge_base_path: 知识库文件路径
            embedding_model: embedding模型名称
                推荐：
                - 'nomic-ai/nomic-embed-text-v1.5' (8192 tokens, 首选)
                - 'Alibaba-NLP/gte-large-en-v1.5' (8192 tokens)
                - 'BAAI/bge-large-en-v1.5' (512 tokens, 备选)
        """
        print("=" * 60)
        print("Initializing Probability RAG System (Long Context Edition)")
        print("=" * 60)
        
        self.data_loader = DataLoader(knowledge_base_path)
        self.questions = self.data_loader.load_data()
        
        if not self.questions:
            raise ValueError("No questions loaded. Please check the data file.")
        
        self.embedder = TextEmbedder(embedding_model)
        self.vector_db = VectorDatabase(self.embedder.embedding_dim)
        self.diversity_filter = DiversityFilter()
        self.kp_weighter = KnowledgePointWeighting()
        self.prompt_builder = PromptBuilder()
        
        self.vector_db.add_questions(self.questions, self.embedder)
        
        print("=" * 60)
        print("✓ RAG System Initialized Successfully")
        print("=" * 60)
    
    def set_knowledge_weights(
        self, 
        accuracy_dict: Optional[Dict[str, float]] = None, 
        csv_path: Optional[str] = None,
        alpha: float = 0.7
    ):
        """设置知识点权重"""
        if csv_path:
            self.kp_weighter.load_from_csv(csv_path, alpha)
        elif accuracy_dict:
            self.kp_weighter.compute_weights(accuracy_dict, alpha)
        else:
            print("Warning: No accuracy data provided")
    
    def retrieve_relevant(
        self, 
        query: str, 
        k: int = 3,
        use_diversity: bool = True,
        use_weights: bool = True
    ) -> List[ExamQuestion]:
        """检索相关问题"""
        
        query_embedding = self.embedder.embed_single(query)
        initial_k = min(k * 5, len(self.questions)) if use_diversity else k
        similarities, indices = self.vector_db.search(query_embedding, k=initial_k)
        
        if use_weights and self.kp_weighter.weights:
            weighted_results = self.kp_weighter.apply_weights(
                similarities, self.questions, indices
            )
            weighted_scores = [score for score, _ in weighted_results]
            indices = [idx for _, idx in weighted_results]
            similarities = np.array(weighted_scores)
        
        if use_diversity:
            indices = self.diversity_filter.filter_diverse_results(
                indices, self.questions, k
            )
        else:
            indices = indices[:k]
        
        return [self.questions[idx] for idx in indices]
    
    def solve_question(
        self, 
        test_question: Dict[str, Any],
        k: int = 3,
        verbose: bool = True
    ) -> Dict[str, Any]:
        """使用RAG解决测试问题"""
        
        if verbose:
            print("\n" + "=" * 60)
            print(f"Solving: {test_question.get('question', '')[:60]}...")
            print("=" * 60)
        
        context = test_question.get('context', '')
        question = test_question.get('question', '')
        query = f"{context} {question}".strip()
        
        retrieved = self.retrieve_relevant(query, k=k)
        
        if verbose:
            print(f"\n✓ Retrieved {len(retrieved)} relevant questions:")
            for i, q in enumerate(retrieved, 1):
                print(f"  {i}. {q.question[:60]}...")
                print(f"     Knowledge: {', '.join(q.knowledge_points[:3])}")
        
        enhanced_prompt = self.prompt_builder.build_prompt(
            test_question, retrieved
        )
        
        result = {
            'query': query,
            'retrieved_questions': retrieved,
            'enhanced_prompt': enhanced_prompt,
            'prompt_length': len(enhanced_prompt),
            'num_retrieved': len(retrieved)
        }
        
        if verbose:
            print(f"\n✓ Generated prompt: {len(enhanced_prompt)} characters")
        
        return result


# ============================================================
# 测试代码
# ============================================================

def main():
    """主测试函数"""
    
    print("\n🚀 Testing RAG System\n")
    
    try:
        # 使用新的长上下文模型
        rag = ProbabilityRAG(
            'augmented_train.jsonl',
            embedding_model='nomic-ai/nomic-embed-text-v1.5'  # 🎯 8192 tokens!
        )
    except Exception as e:
        print(f"Error initializing RAG: {e}")
        print("\nTip: Make sure sentence-transformers is installed:")
        print("  pip install sentence-transformers")
        return
    
    # 设置知识点权重
    rag.set_knowledge_weights(csv_path="Qwen3-14B_trainset_kp_accuracy.csv", alpha=0.7)
    
    # 测试问题
    test_question = {
        'context': 'Three independent normal distributions are combined',
        'question': 'What is the probability that their sum exceeds 50?',
        'options': [
            'Φ((50-μ)/σ)',
            '1-Φ((50-μ)/σ)', 
            '0.5',
            'Cannot determine'
        ]
    }
    
    result = rag.solve_question(test_question, k=3, verbose=True)
    
    # 保存结果
    output_file = 'rag_output_upgraded.txt'
    try:
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write("="*60 + "\n")
            f.write("RAG Output \n")
            f.write("="*60 + "\n\n")
            f.write(result['enhanced_prompt'])
        print(f"\n✓ Results saved to {output_file}")
    except Exception as e:
        print(f"Error saving results: {e}")
    
    print("\n" + "=" * 60)
    print("RAG Statistics:")
    print("=" * 60)
    print(f"Total questions in database: {len(rag.questions)}")
    print(f"Questions retrieved: {result['num_retrieved']}")
    print(f"Prompt length: {result['prompt_length']} characters")
    print(f"Embedding model: {rag.embedder.model.get_sentence_embedding_dimension()}D")
    print(f"Max tokens: {rag.embedder.max_seq_length}")
    print("=" * 60)


if __name__ == "__main__":
    main()