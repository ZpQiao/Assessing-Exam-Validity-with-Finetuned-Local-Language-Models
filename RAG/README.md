# Probability RAG System

RAG-enhanced evaluation framework for probability theory MCQs with FAISS indexing and knowledge point weighting.

---

## `rag.py` (RAG System Core)

- **Input:** Knowledge base JSONL + embedding model + knowledge point weights CSV.
- **Output:** Retrieved similar questions with similarity scores and weighted rankings.
- Provides **multilingual RAG** with FAISS indexing and diversity filtering.

Initialize RAG:
```python
from rag import ProbabilityRAG

rag_system = ProbabilityRAG(
    knowledge_base_path='probability_augmented_train_set.jsonl',
    language='english',  # or 'danish'
    embedding_model='Alibaba-NLP/gte-multilingual-base'
)

# Set knowledge point weights
rag_system.set_knowledge_weights(
    csv_path="Qwen3-14B_trainset_kp_accuracy.csv",
    alpha=0.7
)
```

Retrieve similar questions:
```python
retrieved_questions, similarities, indices = rag_system.retrieve_relevant(
    query="Your probability question here",
    k=3,
    use_diversity=True,
    use_weights=True,
    exclude_base_key='2014_1_1'  # Exclude questions with same base_key
)
```

Features:
- **FAISS indexing** for fast similarity search
- **Diversity filtering** to avoid redundant retrievals
- **Knowledge point weighting** based on model performance
- **GTE/E5 prefix support** for better embeddings
- **Automatic exclusion** of same base_key questions

---

## `test_framework.py` (Evaluation Framework)

- **Input:** Test data JSONL + model config + RAG system.
- **Output:** CSV results with accuracy, timing, and RAG metadata.
- Full-featured testing with **checkpointing**, **RAG support**, and **multi-language** evaluation.

Key class: `ProbabilityTestFramework`
```python
from test_framework import ProbabilityTestFramework

tester = ProbabilityTestFramework(
    model_name="Qwen/Qwen3-14B",
    local_model_path="/root/models/qwen3-14b-q4",
    use_quantization=True,
    quantization_bits=4,
    temperature=0.0,
    use_rag=True,           # Enable RAG
    rag_system=rag_system,  # RAG instance from rag.py
    rag_k=3                 # Retrieve top-3 similar questions
)

results_df = tester.test_dataset(
    data=test_data,
    languages=['english', 'danish'],
    checkpoint_dir='./checkpoints',
    resume=True  # Resume from checkpoint if exists
)
```

Features:
- **RAG-augmented prompting** with retrieved similar questions
- **Checkpoint/resume** every 10 questions (Ctrl+C safe)
- **Multi-model comparison** with automatic report generation
- **Detailed logging** of RAG retrieval metadata (base_keys, similarities)
- **JSON answer extraction** with fallback parsing

---

## `test_main.py` (Main Entry Point)

- **Input:** Test data JSONL + knowledge base JSONL + model paths.
- **Output:** Comparison results between baseline and RAG-enhanced models.
- Unified script to **run baseline vs RAG comparison** with automatic configuration.

Configure in script:
```python
# Enable/disable RAG
USE_RAG = True

# RAG configuration
rag_system = ProbabilityRAG(
    knowledge_base_path='probability_augmented_train_set.jsonl',
    language='english',
    embedding_model='Alibaba-NLP/gte-multilingual-base'
)

# Model configurations
model_configs = [
    # Baseline (no RAG)
    {
        'name': 'Qwen/Qwen3-14B',
        'local_path': '/root/models/qwen3-14b-q4',
        'use_quantization': True,
        'quantization_bits': 4,
        'temperature': 0.0,
        'use_rag': False,
    },
    
    # RAG-enhanced
    {
        'name': 'Qwen/Qwen3-14B',
        'local_path': '/root/models/qwen3-14b-q4',
        'use_quantization': True,
        'quantization_bits': 4,
        'temperature': 0.0,
        'use_rag': True,
        'rag_system': rag_system,
        'rag_k': 3,  # Retrieve 3 similar questions
    },
]
```

Run:
```bash
python test_main.py
```

Output files:
```
./probability_test_results/
  ├── Qwen3-14B_q4_t00_baseline_<timestamp>.csv
  ├── Qwen3-14B_q4_t00_rag_k3_<timestamp>.csv
  └── comparison_report_<timestamp>.csv
```

---

## Quick Workflow

1. **Prepare data**:
   - `probability_augmented_train_set.jsonl` (knowledge base with explanations)
   - `probability_test_set.jsonl` (test questions)
   - `Qwen3-14B_trainset_kp_accuracy.csv` (knowledge point weights)

2. **Test RAG system**:
   ```bash
   python rag.py  # Test retrieval quality
   ```

3. **Run evaluation**:
   ```bash
   python test_main.py  # Compare baseline vs RAG
   ```

4. **Analyze results**:
   - Check CSV files in `./probability_test_results/`
   - View comparison report for accuracy improvements

---

## Data Format

### Knowledge Base (for RAG)
```json
{
  "base_key": "2004_1_1",
  "english": {
    "context": "...",
    "question": "...",
    "options": ["...", "..."]
  },
  "answer_index": 2,
  "knowledge_points": ["Binomial distribution", "Normal approximation"],
  "explanation_key_steps": ["Step 1: ...", "Step 2: ..."],
  "explanation_formulae": ["P(X=k) = ...", "μ = np"],
  "explanation_pitfalls": ["Don't confuse with Poisson", "..."]
}
```

### Test Data
```json
{
  "base_key": "2014_1_1",
  "english": {
    "context": "...",
    "question": "...",
    "options": ["...", "..."],
    "correct_answer": 2
  },
  "danish": { ... }
}
```

---

## RAG Components

### Text Embedder
- **Model:** GTE-multilingual-base (768D embeddings)
- **Prefix support:** Automatic `query:` / `passage:` for GTE/E5 models
- **Context:** Combines question + knowledge points + formulae + key steps

### FAISS Index
- **Type:** Inner product (normalized embeddings = cosine similarity)
- **Retrieval:** Returns top-K most similar questions with scores

### Diversity Filter
- Groups questions by base_key (e.g., `2004_1` for all sub-questions)
- Ensures retrieved questions come from different exam problems
- Prevents redundant context from same question variants

### Knowledge Point Weighter
- Boosts/penalizes retrieved questions based on model's past performance
- Formula: `adjusted_score = similarity * (1 + alpha * (accuracy - 0.5))`
- Higher `alpha` = stronger knowledge point influence

---

## Requirements

```bash
# Core dependencies
pip install torch transformers sentence-transformers
pip install faiss-cpu  # or faiss-gpu for GPU acceleration
pip install pandas numpy tqdm