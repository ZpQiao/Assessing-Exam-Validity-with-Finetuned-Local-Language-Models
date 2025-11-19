"""
Test framework for probability MCQs with (optional) LoRA fine-tuned models.

Features:
- Load a base model (Qwen3-14B) with optional quantization.
- Optionally load a LoRA adapter on top of the base model for inference.
- Evaluate on a JSONL test set of MCQs (English / Danish).
- Extract answers from model outputs, compute accuracy and basic statistics.
"""

import json
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel  # For loading LoRA adapters
from typing import List, Dict
import pandas as pd
from tqdm import tqdm
import time
from datetime import datetime
import os
import re


class ProbabilityTestFramework:
    def __init__(
        self,
        model_name: str,
        device: str = "cuda",
        use_quantization: bool = False,
        quantization_bits: int = 4,
        local_model_path: str = None,
        lora_adapter_path: str = None,
        temperature: float = 0.0,
    ):
        """
        Initialize the test framework.

        Args:
            model_name: Base model name (e.g., "Qwen/Qwen3-14B").
            device: Device to run on ("cuda", "cpu", etc.).
            use_quantization: Whether to enable 4-bit/8-bit quantization.
            quantization_bits: Number of bits for quantization (4 or 8).
            local_model_path: Local path to the base model (if available).
            lora_adapter_path: Path to the LoRA adapter (fine-tuned checkpoint).
            temperature: Sampling temperature for generation.
        """
        self.temperature = temperature
        self.model_name = model_name
        self.device = device
        self.use_quantization = use_quantization
        self.lora_adapter_path = lora_adapter_path

        # Determine base model path
        actual_model_path = model_name
        if local_model_path and os.path.exists(local_model_path):
            actual_model_path = local_model_path
            print(f"Base model: {local_model_path}")
        else:
            print(f"Base model: {model_name} (from HF hub)")

        # LoRA info
        if lora_adapter_path:
            if os.path.exists(lora_adapter_path):
                print(f"LoRA adapter: {lora_adapter_path}")
            else:
                print(f"WARNING: LoRA adapter path does not exist: {lora_adapter_path}")
                print("         Falling back to base model only.")
                self.lora_adapter_path = None

        print(
            f"Quantization: "
            f"{str(quantization_bits) + '-bit' if use_quantization else 'disabled (bf16)'}"
        )
        print(
            f"Temperature: {temperature} ({'deterministic' if temperature == 0 else 'stochastic'})"
        )

        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            actual_model_path,
            use_fast=False,
            trust_remote_code=True,
        )

        # Configure quantization
        if use_quantization:
            if quantization_bits == 4:
                quant_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_use_double_quant=True,
                    bnb_4bit_quant_type="nf4",
                    bnb_4bit_compute_dtype=torch.bfloat16,
                )
            elif quantization_bits == 8:
                quant_config = BitsAndBytesConfig(load_in_8bit=True)
            else:
                raise ValueError(f"Unsupported quantization bits: {quantization_bits}")

            # Load quantized base model
            base_model = AutoModelForCausalLM.from_pretrained(
                actual_model_path,
                device_map="auto",
                torch_dtype=torch.bfloat16,
                quantization_config=quant_config,
                trust_remote_code=True,
            )
        else:
            # Load base model in bf16
            base_model = AutoModelForCausalLM.from_pretrained(
                actual_model_path,
                device_map="auto",
                torch_dtype=torch.bfloat16,
                trust_remote_code=True,
            )

        # Load LoRA adapter (if provided)
        if self.lora_adapter_path and os.path.exists(self.lora_adapter_path):
            print("Loading LoRA adapter...")
            try:
                self.model = PeftModel.from_pretrained(
                    base_model,
                    self.lora_adapter_path,
                    is_trainable=False,  # inference mode
                )
                print("LoRA adapter loaded successfully (fine-tuned model).")
                self.is_finetuned = True
            except Exception as e:
                print(f"WARNING: Failed to load LoRA adapter: {e}")
                print("         Using base model only.")
                self.model = base_model
                self.is_finetuned = False
        else:
            self.model = base_model
            self.is_finetuned = False

        # Mark "thinking" models if needed
        self.is_thinking = "thinking" in model_name.lower()

        model_type = "fine-tuned (LoRA)" if self.is_finetuned else "base"
        print(f"Model ready. Type: {model_type}")

    def format_question(self, item: Dict, language: str = "english") -> str:
        """
        Format a multiple-choice question into a prompt.

        Uses a simple prompt style that:
        - shows context, question, and numbered options
        - asks for reasoning + a JSON answer: {"answer": N}
        """
        lang_data = item[language]

        # Numbered options: 1, 2, 3, ...
        options_text = ""
        for i, option in enumerate(lang_data["options"], 1):
            options_text += f"{i}. {option}\n"

        if language == "english":
            prompt = f"""Please solve the following probability problem and select the correct answer.

Context: {lang_data['context']}

Question: {lang_data['question']}

Options:
{options_text}
Please respond with your reasoning followed by a JSON object in this exact format:
{{"answer": N}}

where N is the number of your chosen option (1, 2, 3, 4, 5, or 6).

Your response:"""
        else:  # danish
            prompt = f"""Løs følgende sandsynlighedsopgave, og vælg det rigtige svar.

Kontekst: {lang_data['context']}

Spørgsmål: {lang_data['question']}

Valgmuligheder:
{options_text}
Svar venligst med din ræsonnering efterfulgt af et JSON-objekt i dette nøjagtige format:
{{"answer": N}}

hvor N er nummeret på dit valgte svar (1, 2, 3, 4, 5 eller 6).

Dit svar:"""

        return prompt

    def generate_answer(self, prompt: str, max_tokens: int = 16000) -> Dict:
        """Run the model on a single prompt and extract the raw response."""
        messages = [{"role": "user", "content": prompt}]

        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=self.is_thinking,
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
                "full_response": f"ERROR: {e}",
                "predicted_answer": None,
                "generation_time": generation_time,
                "has_json": False,
            }

        generation_time = time.time() - start_time

        try:
            full_resp = self.tokenizer.decode(
                outputs[0][inputs["input_ids"].shape[1] :],
                skip_special_tokens=True,
            )
        except Exception:
            full_resp = self.tokenizer.decode(outputs[0], skip_special_tokens=True)

        return self._parse_response(full_resp, generation_time)

    def _parse_response(self, response: str, gen_time: float) -> Dict:
        """
        Parse the model's response and extract the predicted answer.

        Priority:
        1) JSON-like pattern {"answer": N}
        2) Explicit textual patterns ("final answer is 3", etc.)
        3) Last single digit 1–6 in the tail of the response.
        """
        result = {
            "full_response": response,
            "predicted_answer": None,
            "generation_time": gen_time,
            "has_json": False,
        }

        # 1) JSON-like patterns
        json_patterns = [
            r'\{["\']?answer["\']?\s*:\s*(\d+)\s*\}',  # {"answer": 5}
            r'\{["\']?answer["\']?\s*:\s*["\'](\d+)["\']\s*\}',  # {"answer": "5"}
            r'["\']?answer["\']?\s*[:=]\s*(\d+)',  # answer: 5
        ]

        for pattern in json_patterns:
            matches = re.findall(pattern, response, re.IGNORECASE)
            if matches:
                try:
                    answer = int(matches[-1])
                    if 1 <= answer <= 6:
                        result["predicted_answer"] = answer
                        result["has_json"] = True
                        return result
                except Exception:
                    continue

        # 2) Explicit answer statements
        answer_patterns = [
            r'(?:final answer|my answer|the answer|answer)["\']?\s*(?:is|:)\s*["\']?(?:option\s*)?(\d+)["\']?',
            r'(?:option|choice)["\']?\s*["\']?(\d+)["\']?(?:\s+is correct|\s+is the answer)',
            r'(?:I choose|I select|select|choose)["\']?\s*(?:option|choice)?\s*["\']?(\d+)["\']?',
            r'(?:correct answer).*?(?:is|:)\s*["\']?(?:option\s*)?(\d+)["\']?',
            r'(?:therefore).*?answer.*?(?:is|:)\s*["\']?(\d+)["\']?',
        ]

        search_text = response[-200:] if len(response) > 200 else response

        for pattern in answer_patterns:
            matches = re.findall(pattern, search_text, re.IGNORECASE)
            if matches:
                try:
                    answer = int(matches[-1])
                    if 1 <= answer <= 6:
                        result["predicted_answer"] = answer
                        return result
                except Exception:
                    continue

        # 3) Last standalone digit 1–6 near the end
        last_number = re.findall(
            r"\b([1-6])\b", response[-100:] if len(response) > 100 else response
        )
        if last_number:
            try:
                result["predicted_answer"] = int(last_number[-1])
            except Exception:
                pass

        return result

    def test_single_item(self, item: Dict, language: str = "english") -> Dict:
        """Run a single MCQ and compute correctness."""
        prompt = self.format_question(item, language)
        result = self.generate_answer(prompt)

        correct_answer = int(item["answer_index"])
        predicted_answer = result["predicted_answer"]
        is_correct = predicted_answer == correct_answer if predicted_answer else False

        return {
            "base_key": item["base_key"],
            "language": language,
            "correct_answer": correct_answer,
            "predicted_answer": predicted_answer,
            "is_correct": is_correct,
            "full_response": result["full_response"][:500],  # truncate for logging
            "has_json": result["has_json"],
            "generation_time": result["generation_time"],
            "model_type": "finetuned" if self.is_finetuned else "baseline",
        }

    def test_dataset(
        self,
        data: List[Dict],
        languages: List[str] = ["english"],
        limit: int = None,
    ) -> pd.DataFrame:
        """Run evaluation on a dataset."""
        if limit:
            data = data[:limit]

        results = []
        total = len(data) * len(languages)

        print(f"\nStarting evaluation (total {total} questions)...")

        with tqdm(total=total, desc="Progress") as pbar:
            for item in data:
                for lang in languages:
                    try:
                        result = self.test_single_item(item, lang)
                        results.append(result)
                    except Exception as e:
                        print(f"\nERROR: {item['base_key']}_{lang}: {e}")
                        results.append(
                            {
                                "base_key": item["base_key"],
                                "language": lang,
                                "is_correct": False,
                                "full_response": f"ERROR: {e}",
                                "model_type": "finetuned"
                                if self.is_finetuned
                                else "baseline",
                            }
                        )
                    pbar.update(1)

        df = pd.DataFrame(results)

        # Print statistics
        self._print_statistics(df)

        return df

    def _print_statistics(self, df: pd.DataFrame):
        """Print evaluation statistics."""
        print("\n" + "=" * 80)
        print(f"Evaluation summary ({'fine-tuned' if self.is_finetuned else 'base'} model)")
        print("=" * 80)
        print(f"Total questions: {len(df)}")
        print(f"Correct: {df['is_correct'].sum()}")
        print(f"Accuracy: {df['is_correct'].mean() * 100:.2f}%")
        print(
            f"Successful answer extraction: "
            f"{df['predicted_answer'].notna().sum()}/{len(df)}"
        )
        print(f"JSON answer detected: {df['has_json'].sum()}/{len(df)}")
        print(f"Average generation time: {df['generation_time'].mean():.2f} seconds")
        print("=" * 80)


def load_jsonl_data(file_path: str) -> List[Dict]:
    """Load JSONL data from file_path."""
    data = []
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                data.append(json.loads(line))
    print(f"Loaded {len(data)} questions from {file_path}")
    return data


def compare_baseline_vs_finetuned(
    test_data_path: str,
    base_model_path: str,
    lora_adapter_path: str,
    languages: List[str] = ["english"],
    limit: int = None,
    output_dir: str = "./comparison_results",
):
    """
    Compare base model vs LoRA fine-tuned model on the same test set.

    Args:
        test_data_path: Path to test data (JSONL).
        base_model_path: Local path to the base model.
        lora_adapter_path: LoRA adapter path.
        languages: Languages to evaluate.
        limit: Limit the number of test items (None = all).
        output_dir: Directory to save results and reports.
    """
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Load test data
    print("=" * 80)
    print("Loading test data")
    print("=" * 80)
    test_data = load_jsonl_data(test_data_path)

    # ========================================
    # 1. Base model (currently commented out in this script)
    # ========================================
    # print("\n" + "="*80)
    # print("Evaluating base model")
    # print("="*80)
    #
    # baseline_tester = ProbabilityTestFramework(
    #     model_name="Qwen/Qwen3-14B",
    #     local_model_path=base_model_path,
    #     use_quantization=True,
    #     quantization_bits=4,
    #     lora_adapter_path=None,
    #     temperature=0.0,
    # )
    #
    # baseline_results = baseline_tester.test_dataset(
    #     test_data,
    #     languages=languages,
    #     limit=limit,
    # )
    #
    # baseline_file = f"{output_dir}/baseline_{timestamp}.csv"
    # baseline_results.to_csv(baseline_file, index=False, encoding="utf-8-sig")
    # print(f"\nBase model results saved to: {baseline_file}")
    #
    # del baseline_tester
    # torch.cuda.empty_cache()

    # ========================================
    # 2. Fine-tuned model (LoRA)
    # ========================================
    print("\n" + "=" * 80)
    print("Evaluating fine-tuned (LoRA) model")
    print("=" * 80)

    finetuned_tester = ProbabilityTestFramework(
        model_name="Qwen/Qwen3-14B",
        local_model_path=base_model_path,
        use_quantization=True,
        quantization_bits=4,
        lora_adapter_path=lora_adapter_path,
        temperature=0.0,
    )

    finetuned_results = finetuned_tester.test_dataset(
        test_data,
        languages=languages,
        limit=limit,
    )

    finetuned_file = f"{output_dir}/finetuned_{timestamp}.csv"
    finetuned_results.to_csv(finetuned_file, index=False, encoding="utf-8-sig")
    print(f"\nFine-tuned model results saved to: {finetuned_file}")

    del finetuned_tester
    torch.cuda.empty_cache()

    # ========================================
    # 3. Comparison report
    # ========================================
    print("\n" + "=" * 80)
    print("Comparison report")
    print("=" * 80)

    # NOTE: The base-model block above is commented out in this file.
    # If you want a true comparison, uncomment it and ensure baseline_results exists.
    comparison = pd.DataFrame(
        [
            {
                "model": "fine-tuned (LoRA)",
                "accuracy": f"{finetuned_results['is_correct'].mean()*100:.2f}%",
                "correct": finetuned_results["is_correct"].sum(),
                "total": len(finetuned_results),
                "extraction_success_rate": f"{finetuned_results['predicted_answer'].notna().mean()*100:.2f}%",
                "json_rate": f"{finetuned_results['has_json'].mean()*100:.2f}%",
            }
        ]
    )

    print("\n", comparison.to_string(index=False))

    report_file = f"{output_dir}/comparison_report_{timestamp}.csv"
    comparison.to_csv(report_file, index=False, encoding="utf-8-sig")
    print(f"\nComparison report saved to: {report_file}")

    print("\n" + "=" * 80)
    print("Evaluation finished.")
    print("=" * 80)

    # Return only fine-tuned results here (baseline_results would require enabling the base-block)
    return None, finetuned_results


# ============================================================================
# Entry point
# ============================================================================

if __name__ == "__main__":
    print("=" * 80)
    print("Probability model evaluation - supports fine-tuned (LoRA) comparison")
    print("Prompt format: simple text + JSON answer extraction")
    print("=" * 80)

    # Configuration
    TEST_DATA = "probability_test_set.jsonl"  # test set
    BASE_MODEL = "/root/models/qwen3-14b-q4"  # base model path
    LORA_ADAPTER = "./qwen3-qlora-output/final_model"  # fine-tuned LoRA path

    TEST_LANGUAGES = ["english", "danish"]
    TEST_LIMIT = None  # None = all, or set e.g. 10 for quick sanity check

    # Run comparison (currently only fine-tuned results are returned)
    baseline_df, finetuned_df = compare_baseline_vs_finetuned(
        test_data_path=TEST_DATA,
        base_model_path=BASE_MODEL,
        lora_adapter_path=LORA_ADAPTER,
        languages=TEST_LANGUAGES,
        limit=TEST_LIMIT,
        output_dir="./comparison_results",
    )
