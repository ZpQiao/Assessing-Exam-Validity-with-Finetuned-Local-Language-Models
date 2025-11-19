import json
import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import List, Dict, Any, Tuple, Optional
from sentence_transformers import SentenceTransformer
import faiss


# ============================================================
# Data structures
# ============================================================

@dataclass
class ExamQuestion:
    """Data structure for a single exam question."""
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
# Data loading
# ============================================================

class DataLoader:
    """Load exam data from a JSONL file."""

    def __init__(self, data_file: str, language: str = 'english'):
        self.data_file = data_file
        self.language = language  # 'english' or 'danish'
        self.questions: List[ExamQuestion] = []

    def load_data(self) -> List[ExamQuestion]:
        """Load questions from a JSONL file."""
        questions: List[ExamQuestion] = []

        try:
            with open(self.data_file, 'r', encoding='utf-8') as f:
                for line_num, line in enumerate(f, 1):
                    try:
                        item = json.loads(line.strip())

                        # Choose language dynamically
                        lang_data = item.get(self.language)
                        if not lang_data:
                            print(f"Warning: Line {line_num} - {self.language} block not found for {item.get('base_key', 'unknown')}")
                            continue

                        question = ExamQuestion(
                            base_key=item['base_key'],
                            context=lang_data['context'],
                            question=lang_data['question'],
                            options=lang_data['options'],
                            answer_index=item['answer_index'],
                            knowledge_points=item['knowledge_points'],
                            explanation_key_steps=item['explanation_key_steps'],
                            explanation_formulae=item['explanation_formulae'],
                            explanation_pitfalls=item['explanation_pitfalls'],
                            language=self.language
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
        print(f"[INFO] Loaded {len(questions)} questions ({self.language})")
        return questions


# ============================================================
# Text embedding (with GTE/E5 prefix support)
# ============================================================

class TextEmbedder:
    """Text embedder with support for long texts and GTE-style prefixes."""

    def __init__(self, model_name: str = 'Alibaba-NLP/gte-multilingual-base'):
        print(f"[INFO] Loading embedding model: {model_name}...")

        self.model_name = model_name

        # Load model
        try:
            self.model = SentenceTransformer(model_name, trust_remote_code=True)
        except Exception:
            self.model = SentenceTransformer(model_name)

        self.embedding_dim = self.model.get_sentence_embedding_dimension()
        self.max_seq_length = self.model.max_seq_length
        self.tokenizer = self.model.tokenizer

        # Check whether we need prefixes (GTE/E5 style)
        self.needs_prefix = any(x in model_name.lower() for x in ['gte', 'e5'])

        print(f"[INFO] Embedding model loaded")
        print(f"  Dimension: {self.embedding_dim}")
        print(f"  Max tokens: {self.max_seq_length}")
        if self.needs_prefix:
            print("  [INFO] Using instruction prefixes (query: / passage:)")

    def _add_prefix(self, text: str, is_query: bool = False) -> str:
        """Add GTE/E5 prefix to text if required."""
        if not self.needs_prefix:
            return text

        prefix = "query: " if is_query else "passage: "
        return prefix + text

    def count_tokens(self, text: str) -> int:
        """Estimate token count for a given text."""
        try:
            return len(self.tokenizer.encode(text))
        except Exception:
            # Fallback rough estimate
            return int(len(text.split()) * 1.3)

    def create_search_text(self, question: ExamQuestion) -> str:
        """Build the text used for indexing/search for a question."""
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
            print("   Warning: Search text will be truncated to model max length.")
            print(f"   Question: {question.base_key}")
            print(f"   Tokens: {token_count} -> {self.max_seq_length} (truncated)")

        return search_text

    def embed_texts(
        self,
        texts: List[str],
        batch_size: int = 32,
        is_query: bool = False
    ) -> np.ndarray:
        """
        Generate embeddings for a list of texts.

        Args:
            texts: List of strings.
            batch_size: Batch size for encoding.
            is_query: Whether these are query embeddings (True) or passage embeddings (False).
        """
        print(f"[INFO] Generating embeddings for {len(texts)} texts...")

        # Add GTE/E5 prefixes if needed
        processed_texts = texts
        if self.needs_prefix:
            prefix_type = "query" if is_query else "passage"
            print(f"  [INFO] Adding '{prefix_type}:' prefix to texts...")
            processed_texts = [self._add_prefix(text, is_query) for text in texts]

        # Token length statistics (on the original, prefix-free text)
        token_counts = [self.count_tokens(text) for text in texts]
        max_tokens = max(token_counts)
        avg_tokens = sum(token_counts) / len(token_counts)
        truncated_count = sum(1 for t in token_counts if t > self.max_seq_length)

        print("  Token statistics:")
        print(f"    Max: {max_tokens} tokens")
        print(f"    Avg: {avg_tokens:.0f} tokens")
        print(f"    Truncated: {truncated_count}/{len(texts)} texts ({truncated_count/len(texts)*100:.1f}%)")

        if truncated_count > 0:
            print(f"  [WARN] {truncated_count} texts will be truncated to model max length.")

        embeddings = self.model.encode(
            processed_texts,
            batch_size=batch_size,
            normalize_embeddings=True,
            show_progress_bar=True
        )

        print(f"[INFO] Embeddings generated (shape: {embeddings.shape})")
        return embeddings

    def embed_single(self, text: str, is_query: bool = True) -> np.ndarray:
        """
        Generate an embedding for a single text.

        Args:
            text: Input string.
            is_query: Whether this is a query embedding (True) or passage (False).
        """
        processed_text = self._add_prefix(text, is_query) if self.needs_prefix else text

        return self.model.encode(
            [processed_text],
            normalize_embeddings=True
        )[0]


# ============================================================
# Vector database
# ============================================================

class VectorDatabase:
    """FAISS-based vector database."""

    def __init__(self, embedding_dim: int = 768):
        self.embedding_dim = embedding_dim
        self.embeddings: Optional[np.ndarray] = None
        self.metadata: List[Dict[str, Any]] = []
        self.index: Optional[faiss.IndexFlatIP] = None

    def add_questions(self, questions: List[ExamQuestion], embedder: TextEmbedder):
        """Add questions to the vector database and build index."""
        if not questions:
            print("[WARN] No questions to add to vector database.")
            return

        search_texts = []
        for q in questions:
            search_text = embedder.create_search_text(q)
            search_texts.append(search_text)

            self.metadata.append({
                'question': q,
                'search_text': search_text
            })

        # Knowledge base uses passage embeddings (is_query=False)
        self.embeddings = embedder.embed_texts(search_texts, is_query=False)
        self.embedding_dim = self.embeddings.shape[1]
        self._build_faiss_index()

        print(f"[INFO] Added {len(questions)} questions to vector database.")

    def _build_faiss_index(self):
        """Build a FAISS index over the current embeddings."""
        if self.embeddings is None or len(self.embeddings) == 0:
            raise ValueError("No embeddings available for building index.")

        self.index = faiss.IndexFlatIP(self.embedding_dim)
        self.index.add(self.embeddings.astype('float32'))

        print(f"[INFO] FAISS index built with {self.index.ntotal} vectors.")

    def search(self, query_embedding: np.ndarray, k: int = 5) -> Tuple[np.ndarray, List[int]]:
        """Search for the top-k most similar vectors given a query embedding."""
        if self.index is None:
            raise ValueError("Index not built. Call add_questions() first.")

        if query_embedding.ndim == 1:
            query_embedding = query_embedding.reshape(1, -1)

        k = min(k, self.index.ntotal)
        similarities, indices = self.index.search(
            query_embedding.astype('float32'),
            k
        )

        # similarities: (1, k) ndarray, indices: (1, k) ndarray
        return similarities[0], indices[0].tolist()


# ============================================================
# Diversity filtering
# ============================================================

class DiversityFilter:
    """Ensure diversity among retrieved questions."""

    @staticmethod
    def extract_base_key(key: str) -> str:
        """Extract the base question ID, stripping suffixes like _rephrase_ / _backward_."""
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
        """Filter retrieval results to keep diverse base questions."""
        selected_indices: List[int] = []
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
# Knowledge point weighting
# ============================================================

class KnowledgePointWeighting:
    """Knowledge point weighting system based on training accuracy."""

    def __init__(self):
        self.weights: Dict[str, float] = {}

    def compute_weights(self, accuracy_dict: Dict[str, float], alpha: float = 0.7):
        """Compute knowledge point weights from accuracy values."""
        for kp, accuracy in accuracy_dict.items():
            if accuracy > 1:
                accuracy = accuracy / 100.0

            accuracy = max(0.0, min(1.0, accuracy))
            self.weights[kp] = 1.0 + alpha * (1 - accuracy)

        print(f"[INFO] Computed weights for {len(self.weights)} knowledge points.")

        if self.weights:
            sorted_kps = sorted(self.weights.items(), key=lambda x: x[1], reverse=True)
            print("\nTop-5 hardest knowledge points:")
            for kp, weight in sorted_kps[:5]:
                acc = accuracy_dict[kp]
                print(f"  - {kp}: weight={weight:.3f} (accuracy={acc:.1%})")

    def apply_weights(
        self,
        similarities: np.ndarray,
        questions: List[ExamQuestion],
        indices: List[int]
    ) -> List[Tuple[float, int]]:
        """Apply knowledge point weights to similarity scores."""
        weighted_results: List[Tuple[float, int]] = []

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
        """Load knowledge point accuracies from a CSV file and compute weights."""
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
# Prompt builder (optional, commented)
# ============================================================

# class PromptBuilder:
#     """Build RAG-augmented prompts that include retrieved reference problems."""
#
#     def build_prompt(
#         self,
#         test_question: Dict[str, Any],
#         retrieved_questions: List[ExamQuestion]
#     ) -> str:
#         """Construct a prompt that includes retrieved related questions."""
#
#         prompt_parts: List[str] = [
#             "You are an expert in probability and statistics. ",
#             "Use the following similar problems as reference to solve the given problem.\n\n"
#         ]
#
#         if retrieved_questions:
#             prompt_parts.append("=== REFERENCE PROBLEMS ===\n\n")
#
#             for i, q in enumerate(retrieved_questions, 1):
#                 prompt_parts.append(f"Reference {i}:\n")
#
#                 if q.context:
#                     prompt_parts.append(f"Context: {q.context}\n")
#
#                 prompt_parts.append(f"Question: {q.question}\n")
#                 prompt_parts.append(f"Answer: Option {q.answer_index}\n")
#
#                 if q.explanation_key_steps:
#                     prompt_parts.append("\nKey steps:\n")
#                     for step in q.explanation_key_steps:
#                         prompt_parts.append(f"  - {step}\n")
#
#                 if q.explanation_formulae:
#                     prompt_parts.append("\nKey formulae:\n")
#                     for formula in q.explanation_formulae:
#                         prompt_parts.append(f"  - {formula}\n")
#
#                 if q.explanation_pitfalls:
#                     prompt_parts.append("\nCommon pitfalls to avoid:\n")
#                     for pitfall in q.explanation_pitfalls:
#                         prompt_parts.append(f"  - {pitfall}\n")
#
#                 prompt_parts.append("\n")
#
#         prompt_parts.append("=== PROBLEM TO SOLVE ===\n\n")
#
#         if test_question.get('context'):
#             prompt_parts.append(f"Context: {test_question['context']}\n")
#
#         prompt_parts.append(f"Question: {test_question['question']}\n\n")
#
#         if 'options' in test_question:
#             prompt_parts.append("Options:\n")
#             for i, opt in enumerate(test_question['options'], 1):
#                 prompt_parts.append(f"{i}. {opt}\n")
#
#         prompt_parts.append("\nProvide your answer and explanation.")
#
#         return ''.join(prompt_parts)


# ============================================================
# Main RAG system
# ============================================================

class ProbabilityRAG:
    """Multilingual probability/statistics RAG system."""

    def __init__(
        self,
        knowledge_base_path: str,
        language: str = 'english',
        embedding_model: Optional[str] = None
    ):
        """
        Initialize the RAG system.

        Args:
            knowledge_base_path: Path to knowledge base JSONL file.
            language: 'english' or 'danish'.
            embedding_model: Name of the embedding model. If None, a default is used.
        """
        print("=" * 60)
        print("Initializing Probability RAG System (Multilingual Edition)")
        print("=" * 60)

        # Always use this embedding model here
        embedding_model = 'Alibaba-NLP/gte-multilingual-base'

        print(f"Language: {language}")
        print(f"Embedding model: {embedding_model}")
        print()

        self.language = language
        self.data_loader = DataLoader(knowledge_base_path, language=language)
        self.questions = self.data_loader.load_data()

        if not self.questions:
            raise ValueError("No questions loaded. Please check the data file.")

        self.embedder = TextEmbedder(embedding_model)
        self.vector_db = VectorDatabase(self.embedder.embedding_dim)
        self.diversity_filter = DiversityFilter()
        self.kp_weighter = KnowledgePointWeighting()
        # self.prompt_builder = PromptBuilder()

        self.vector_db.add_questions(self.questions, self.embedder)

        print("=" * 60)
        print("[INFO] RAG System Initialized Successfully")
        print("=" * 60)

    def set_knowledge_weights(
        self,
        accuracy_dict: Optional[Dict[str, float]] = None,
        csv_path: Optional[str] = None,
        alpha: float = 0.7
    ):
        """Configure knowledge point weights from a dict or CSV."""
        if csv_path:
            self.kp_weighter.load_from_csv(csv_path, alpha)
        elif accuracy_dict:
            self.kp_weighter.compute_weights(accuracy_dict, alpha)
        else:
            print("Warning: No accuracy data provided; weights not updated.")

    def retrieve_relevant(
        self,
        query: str,
        k: int = 3,
        use_diversity: bool = True,
        use_weights: bool = True,
        verbose: bool = False,
        exclude_base_key: Optional[str] = None,
    ) -> Tuple[List[ExamQuestion], np.ndarray, List[int]]:
        """
        Retrieve relevant questions from the knowledge base.

        Returns:
            retrieved_questions: list of ExamQuestion
            similarities: similarity scores
            indices: indices into self.questions
        """
        # Query uses query embeddings (is_query=True)
        query_embedding = self.embedder.embed_single(query, is_query=True)
        initial_k = min(k * 5, len(self.questions)) if use_diversity else k
        similarities, indices = self.vector_db.search(query_embedding, k=initial_k)

        # Save original results in case we need to fall back
        original_similarities = similarities.copy()
        original_indices = indices.copy()

        # Apply knowledge point weights
        if use_weights and self.kp_weighter.weights:
            weighted_results = self.kp_weighter.apply_weights(
                similarities, self.questions, indices
            )
            weighted_scores = [score for score, _ in weighted_results]
            indices = [idx for _, idx in weighted_results]
            similarities = np.array(weighted_scores)

        # Exclude items with the same base_key as the current test question
        if exclude_base_key:
            excluded_base = DiversityFilter.extract_base_key(exclude_base_key)
            filtered_indices: List[int] = []
            filtered_similarities: List[float] = []

            for sim, idx in zip(similarities, indices):
                if idx >= len(self.questions):
                    continue

                q = self.questions[idx]
                q_base = DiversityFilter.extract_base_key(q.base_key)

                if q_base == excluded_base:
                    if verbose:
                        print(f"  Skipping {q.base_key} (same base_key as test question)")
                    continue

                filtered_indices.append(idx)
                filtered_similarities.append(float(sim))

            # Fall back to the original list if everything got filtered
            if filtered_indices:
                indices = filtered_indices
                similarities = np.array(filtered_similarities, dtype=np.float32)
            else:
                if verbose:
                    print("  Warning: all retrieved questions had the same base_key; using original list.")
                indices = original_indices
                similarities = original_similarities

        # Diversity filtering
        if use_diversity:
            selected_indices = self.diversity_filter.filter_diverse_results(
                indices, self.questions, k
            )
        else:
            selected_indices = indices[:k]

        # Extract similarity scores for the selected indices
        final_similarities: List[float] = []
        for idx in selected_indices:
            try:
                pos = list(indices).index(idx)
                final_similarities.append(float(similarities[pos]))
            except ValueError:
                final_similarities.append(0.0)

        final_similarities_arr = np.array(final_similarities, dtype=np.float32)

        # Optional verbose logging
        if verbose:
            print("\n[INFO] Retrieval details:")
            for rank, (idx, sim) in enumerate(zip(selected_indices, final_similarities_arr), 1):
                q = self.questions[idx]
                print(f"  Rank {rank}: {q.base_key} (similarity: {sim:.4f})")

        return [self.questions[idx] for idx in selected_indices], final_similarities_arr, selected_indices

    def solve_question(
        self,
        test_question: Dict[str, Any],
        k: int = 3,
        verbose: bool = True
    ) -> Dict[str, Any]:
        """Use RAG to retrieve related questions and build an augmented prompt for a test question."""
        if verbose:
            print("\n" + "=" * 60)
            print(f"Solving: {test_question.get('question', '')[:60]}...")

        context = test_question.get('context', '')
        question = test_question.get('question', '')
        query = f"{context} {question}".strip()
        test_base_key = test_question.get('base_key', '')

        # Retrieve related questions, similarities, and indices (excluding same base_key)
        retrieved, similarities, indices = self.retrieve_relevant(
            query,
            k=k,
            verbose=verbose,
            exclude_base_key=test_base_key,
        )

        if verbose:
            print(f"\n[INFO] Retrieved {len(retrieved)} relevant questions:\n")
            for i, (q, sim) in enumerate(zip(retrieved, similarities), 1):
                print(f"  {i}. [{q.base_key}] (similarity: {sim:.4f})")
                print(f"     Question: {q.question[:70]}...")
                print(f"     Knowledge: {', '.join(q.knowledge_points[:3])}")
                if len(q.knowledge_points) > 3:
                    print(f"                {', '.join(q.knowledge_points[3:6])}")
                print()

        # NOTE: prompt_builder is commented out above; this call will fail unless you enable it.
        enhanced_prompt = self.prompt_builder.build_prompt(
            test_question, retrieved
        )

        result = {
            'query': query,
            'retrieved_questions': retrieved,
            'retrieved_base_keys': [q.base_key for q in retrieved],
            'similarities': similarities.tolist(),
            'indices': indices,
            'enhanced_prompt': enhanced_prompt,
            'prompt_length': len(enhanced_prompt),
            'num_retrieved': len(retrieved)
        }

        if verbose:
            print(f"[INFO] Generated prompt length: {len(enhanced_prompt)} characters")
            print(f"\nRetrieved question IDs: {', '.join(result['retrieved_base_keys'])}")
            print("Similarity scores: " + ", ".join([f"{s:.4f}" for s in similarities]))

        return result


# ============================================================
# Test / demo
# ============================================================

def main():
    """Main test function for the multilingual RAG system."""
    print("\n[INFO] Testing Multilingual RAG System\n")

    print("\n" + "=" * 30 + " ENGLISH RAG " + "=" * 30)

    try:
        rag_en = ProbabilityRAG(
            'probability_augmented_train_set.jsonl',
            language='english',
            embedding_model='Alibaba-NLP/gte-multilingual-base'
        )

        # Set knowledge point weights
        rag_en.set_knowledge_weights(
            csv_path="Qwen3-14B_trainset_kp_accuracy.csv",
            alpha=0.7
        )

        # Example test question
        test_question = {
            'base_key': 'exam_2014_05_28-11',
            'context': 'A lotto player buys a row every week. The probability of getting a pay-out on the row in a given week is 1/5.',
            'question': 'The probability of getting a pay-out in at least 10 out of 100 weeks is',
            'options': [
                r'Φ\left(\frac{10-20+\frac{1}{2}}{\sqrt{100 \cdot \frac{4}{25}}}\right)',
                r'\sum_{i=10}^{100} \frac{20^{i}}{i!} e^{-20}',
                r'1-5^{-100} \sum_{i=0}^{9}\binom{100}{i} 4^{100-i}',
                r'\frac{\binom{20}{10}\binom{80}{10}}{\binom{100}{20}}',
                r'\sum_{i=10}^{100}\binom{100}{i} \frac{1}{2^{100}}',
                'Do not know'
            ],
            'answer_index': 0,
            'knowledge_points': ['Binomial distribution', 'Complement rule (at least one)', 'Normal approximation']
        }

        result_en = rag_en.solve_question(test_question, k=3, verbose=True)

        # Save prompt and retrieval info to a text file
        output_file = 'rag_output_english.txt'
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write("=" * 60 + "\n")
            f.write("RAG Output (English)\n")
            f.write("=" * 60 + "\n\n")
            f.write(f"Retrieved Questions: {', '.join(result_en['retrieved_base_keys'])}\n")
            f.write("Similarities: " + ", ".join([f"{s:.4f}" for s in result_en['similarities']]) + "\n\n")
            f.write("=" * 60 + "\n\n")
            f.write(result_en['enhanced_prompt'])
        print(f"\n[INFO] English results saved to {output_file}")

        # Summary
        print("\n" + "=" * 60)
        print("RAG System Summary:")
        print("=" * 60)
        print(f"Questions loaded: {len(rag_en.questions)}")
        print(f"Questions retrieved: {result_en['num_retrieved']}")
        print(f"Prompt length: {result_en['prompt_length']} characters")
        print(f"Embedding dimension: {rag_en.embedder.embedding_dim}D")
        print(f"Max tokens: {rag_en.embedder.max_seq_length}")
        print(f"Prefix support: {rag_en.embedder.needs_prefix}")
        print("=" * 60)

    except Exception as e:
        print(f"[ERROR] English RAG error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
