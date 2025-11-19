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
    def __init__(
        self,
        model_name: str,
        device: str = "cuda",
        use_quantization: bool = False,
        quantization_bits: int = 4,
        local_model_path: str = None,
        temperature: float = 0.0,
        use_rag: bool = False,
        rag_system=None,
        rag_k: int = 3,
    ):
        """
        Initialize the test framework.

        Args:
            model_name: Base model name.
            device: Device to run on.
            use_quantization: Whether to use quantization.
            quantization_bits: Number of bits for quantization (4 or 8).
            local_model_path: Local model path (preferred if exists).
            temperature: Generation temperature (0 = deterministic).
            use_rag: Whether to use RAG-augmented prompting.
            rag_system: RAG system instance (e.g., a ProbabilityRAG object).
            rag_k: Number of retrieved similar questions.
        """
        self.temperature = temperature
        self.use_rag = use_rag
        self.rag_system = rag_system
        self.rag_k = rag_k
        self._last_rag_meta = None

        if use_rag and rag_system is None:
            raise ValueError("rag_system instance needed")

        # Determine actual model path
        actual_model_path = model_name
        if local_model_path and os.path.exists(local_model_path):
            actual_model_path = local_model_path
            print(f"[INFO] Loading model from local path: {local_model_path}")
        else:
            if local_model_path:
                print(f"[WARN] Local path not found: {local_model_path}")
            print(f"[INFO] Loading model from Hugging Face: {model_name}")

        print(
            f"  Quantization: "
            f"{str(quantization_bits) + '-bit' if use_quantization else 'disabled (BF16)'}"
        )
        print(
            f"  Temperature: {temperature} "
            f"({'deterministic' if temperature == 0 else 'stochastic'})"
        )

        self.model_name = model_name
        self.device = device
        self.use_quantization = use_quantization

        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            actual_model_path,
            use_fast=False,
            trust_remote_code=True,
        )

        # Configure and load model
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
                raise ValueError(f"Unsupported quantization bits: {quantization_bits}")

            self.model = AutoModelForCausalLM.from_pretrained(
                actual_model_path,
                device_map="auto",
                torch_dtype=torch.bfloat16,
                quantization_config=quant_config,
                trust_remote_code=True,
            )
        else:
            self.model = AutoModelForCausalLM.from_pretrained(
                actual_model_path,
                device_map="auto",
                torch_dtype=torch.bfloat16,
                trust_remote_code=True,
            )

        # Mark "thinking" models
        self.is_thinking = "thinking" in model_name.lower()

        print(
            f"[INFO] Model loaded. Type: "
            f"{'Thinking' if self.is_thinking else 'Instruct'}"
        )

    def format_question(self, item: Dict, language: str = "english") -> str:
        """
        Format a question into a prompt.
        Automatically chooses basic or RAG-enhanced version.
        """
        self._last_rag_meta = None  # Clear previous RAG metadata

        if self.use_rag:
            return self._format_question_with_rag(item, language)
        else:
            return self._format_question_basic(item, language)

    def _format_question_basic(self, item: Dict, language: str = "english") -> str:
        """Format question into a simple prompt (JSON answer, numeric index)."""
        lang_data = item[language]

        # Numbered options 1, 2, 3, ...
        options_text = ""
        for i, option in enumerate(lang_data["options"], 1):
            options_text += f"{i}. {option}\n"

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

hvor N er nummeret på dit valgte svar (1, 2, 3, 4, 5 eller 6).

Dit svar:"""

        return prompt

    def _get_attr(self, obj, key, default=None):
        """Helper: support both dict and object attributes."""
        if isinstance(obj, dict):
            return obj.get(key, default)
        return getattr(obj, key, default)

    def _format_question_with_rag(self, item: Dict, language: str = "english") -> str:
        print("[DEBUG] format_question: using RAG-augmented prompt")
        lang_data = item[language]

        # 1. Build query text
        query = f"{lang_data['context']} {lang_data['question']}".strip()

        # 2. Use RAG to retrieve similar questions
        try:
            raw = self.rag_system.retrieve_relevant(
                query,
                k=self.rag_k,
                use_diversity=True,
                use_weights=True,
                verbose=False,
                exclude_base_key=item["base_key"],
            )
            retrieved_questions, similarities, indices = raw

            # Cache RAG metadata for later logging
            base_keys = []
            for q in retrieved_questions:
                bk = self._get_attr(q, "base_key", "")
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
                "retrieved_base_keys": base_keys,
                "retrieved_similarities": _to_pylist(similarities),
                "retrieved_indices": _to_pylist(indices),
            }

        except Exception as e:
            print(f"[WARN] RAG retrieval failed: {e}, falling back to basic prompt.")
            return self._format_question_basic(item, language)

        # 3. Build augmented prompt
        prompt_parts = []

        # System-level guidance
        if language == "english":
            prompt_parts.append(
                "You are an expert in probability and statistics. "
                "Below are some related problems from past exams that MAY provide useful context. "
                "However, they may not be directly applicable to this specific problem.\n\n"
                "VERY IMPORTANT:\n"
                "- DO NOT copy hidden assumptions (such as independence, a parameter p, "
                "or a specific distribution) from the reference problems into the current problem.\n"
                "- Use ONLY assumptions explicitly stated in the CURRENT problem.\n"
                "- If a reference problem introduces a parameter p, that does NOT mean p exists here.\n"
                "- In case of any conflict, ALWAYS follow the CURRENT problem.\n\n"
                "Always derive your solution independently and verify your calculations.\n\n"
                "COIN-TOSS CONVENTION FOR THESE EXAMS:\n"
                "- If a problem just says \"a coin\" and does not define p, assume a fair coin: p = 1/2.\n"
                "- Only treat p as unknown if the current problem explicitly introduces p.\n\n"
            )
        else:  # danish
            prompt_parts.append(
                "Du er ekspert i sandsynlighedsteori og statistik. "
                "Nedenfor er nogle relaterede problemer fra tidligere eksamener, som MÅSKE kan give nyttig kontekst. "
                "De er dog ikke nødvendigvis direkte anvendelige på denne specifikke opgave.\n\n"
                "MEGET VIGTIGT:\n"
                "- KOPIER IKKE skjulte antagelser (som uafhængighed, en parameter p eller en bestemt fordeling)\n"
                "  fra referenceopgaverne over i den aktuelle opgave.\n"
                "- Brug KUN antagelser, der er eksplicit givet i DEN AKTUELLE opgave.\n"
                "- Hvis en referenceopgave indfører en parameter p, betyder det IKKE, at p findes her.\n"
                "- Ved konflikt skal du ALTID følge DEN AKTUELLE opgave.\n\n"
                "Udled altid din løsning selvstændigt og verificer dine beregninger.\n\n"
                "KONVENTION FOR MØNTKAST I DISSE EKSAMENER:\n"
                "- Hvis en opgave blot siger \"en mønt\" og IKKE definerer p, antag en fair mønt: p = 1/2.\n"
                "- Behandl kun p som ukendt, hvis den aktuelle opgave eksplicit indfører p.\n\n"
            )

        # Reference problems section
        if retrieved_questions:
            prompt_parts.append("=== REFERENCE PROBLEMS ===\n\n")

            for i, q in enumerate(retrieved_questions, 1):
                prompt_parts.append(f"Reference {i}:\n")

                if q.context:
                    prompt_parts.append(f"Context: {q.context}\n")

                prompt_parts.append(f"Question: {q.question}\n")
                prompt_parts.append(f"Correct Answer: Option {q.answer_index}\n")

                if q.explanation_key_steps:
                    prompt_parts.append("\nKey steps:\n")
                    for step in q.explanation_key_steps:
                        prompt_parts.append(f"  - {step}\n")

                if q.explanation_formulae:
                    prompt_parts.append("\nKey formulae:\n")
                    for formula in q.explanation_formulae:
                        prompt_parts.append(f"  - {formula}\n")

                if q.explanation_pitfalls:
                    prompt_parts.append("\nCommon pitfalls to avoid:\n")
                    for pitfall in q.explanation_pitfalls:
                        prompt_parts.append(f"  - {pitfall}\n")

                prompt_parts.append("\n")

        # Now add the current target problem
        prompt_parts.append("=== PROBLEM TO SOLVE ===\n\n")

        if lang_data["context"]:
            prompt_parts.append(f"Context: {lang_data['context']}\n")

        prompt_parts.append(f"Question: {lang_data['question']}\n\n")
        prompt_parts.append("Options:\n")
        for i, opt in enumerate(lang_data["options"], 1):
            prompt_parts.append(f"{i}. {opt}\n")

        if language == "english":
            prompt_parts.append(
                """IMPORTANT NOTES:
- Mathematical expressions may be written in different equivalent forms.
- For example: sqrt(a^2 * b) equals a * sqrt(b).
- Expression order does not matter: e^(-A) * Σ(...) equals Σ(...) * e^(-A).
- Consider algebraic equivalence when comparing your answer to the options.
- If your derived expression matches an option algebraically, select that option.

Please respond with your reasoning followed by a JSON object in this exact format:
{"answer": N}

where N is the number of your chosen option (1, 2, 3, 4, 5, or 6).
Option 6 ("Ved ikke" / "Do not know") should ONLY be selected if:
- The problem is genuinely unsolvable with the given information, OR
- All options are clearly incorrect, OR
- You cannot derive any reasonable answer.

Your response:"""
            )
        else:
            prompt_parts.append(
                """VIGTIGE NOTER:
- Matematiske udtryk kan være skrevet i forskellige ækvivalente former.
- For eksempel: sqrt(a^2 * b) er lig med a * sqrt(b).
- Rækkefølgen af faktorer er ligegyldig: e^(-A) * Σ(...) er lig med Σ(...) * e^(-A).
- Overvej algebraisk ækvivalens, når du sammenligner dit svar med valgmulighederne.

Svar venligst med din ræsonnering efterfulgt af et JSON-objekt i dette nøjagtige format:
{"answer": N}

hvor N er nummeret på dit valgte svar (1, 2, 3, 4, 5 eller 6).
Valgmulighed 6 ("Ved ikke") skal KUN vælges, hvis opgaven reelt er uløselig.

Dit svar:"""
            )

        return "".join(prompt_parts)

    def generate_answer(self, prompt: str, max_tokens: int = 2000) -> Dict:
        """
        Generate an answer for a single prompt.

        Strategy:
        - Run a single generate() call.
        - Detect pathological repetition in the tail of the output.
        - If repeated patterns are detected, skip this sample.
        """
        print(f"\n[DEBUG] Prompt length: {len(prompt)} characters")
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
            "temperature": 0,
            "do_sample": True
            if (self.temperature is not None and self.temperature > 0)
            else False,
        }

        start_time = time.time()
        try:
            with torch.no_grad():
                outputs = self.model.generate(**inputs, **gen_config)
        except Exception as e:
            generation_time = time.time() - start_time
            return {
                "full_response": f"ERROR_GENERATION: {e}",
                "thinking_process": "",
                "predicted_answer": None,
                "generation_time": generation_time,
                "has_thinking": False,
                "has_json": False,
                "skip_reason": None,
            }

        generation_time = time.time() - start_time

        try:
            full_resp = self.tokenizer.decode(
                outputs[0][inputs["input_ids"].shape[1] :],
                skip_special_tokens=True,
            )
        except Exception:
            full_resp = self.tokenizer.decode(outputs[0], skip_special_tokens=True)

        # Normalize text for repetition detection
        norm = re.sub(r"\s+", " ", full_resp).strip().lower()

        repetition_threshold = 5
        skipped = False
        repeat_info = None

        if len(norm) >= 50:
            for length in range(8, min(64, len(norm) // 2 + 1)):
                tail_len = length * (repetition_threshold + 1)
                tail = norm[-tail_len:] if len(norm) >= tail_len else norm
                if len(tail) < length * 2:
                    continue

                chunks = [tail[i : i + length] for i in range(0, len(tail), length)]
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
                        "repeat_chunk": last_chunk[:100],
                        "repeat_length": length,
                        "repeat_count": cnt,
                    }
                    break

        if skipped:
            return {
                "full_response": "SKIPPED_DUE_TO_REPETITION",
                "thinking_process": "",
                "predicted_answer": None,
                "generation_time": generation_time,
                "has_thinking": False,
                "has_json": False,
                "skip_reason": f"detected_repetition: "
                f"chunk_len={repeat_info['repeat_length']}, "
                f"count={repeat_info['repeat_count']}",
            }

        return self._parse_response(full_resp, generation_time)

    def _parse_response(self, response: str, gen_time: float) -> Dict:
        """
        Parse model response.

        Priority:
        1) JSON format: {"answer": N}
        2) Explicit answer patterns.
        """
        result = {
            "full_response": response,
            "thinking_process": "",
            "predicted_answer": None,
            "generation_time": gen_time,
            "has_thinking": False,
            "has_json": False,
        }

        # Extract thinking trace if "thinking" model
        if self.is_thinking and "</think>" in response:
            result["has_thinking"] = True
            parts = response.split("</think>")
            if len(parts) >= 2:
                think_start = response.find("<think>")
                if think_start != -1:
                    result["thinking_process"] = response[
                        think_start : response.find("</think>") + 8
                    ]
                else:
                    result["thinking_process"] = parts[0]
                response = parts[-1].strip()

        # 1) JSON patterns
        json_patterns = [
            r'\{["\']?answer["\']?\s*:\s*(\d+)\s*\}',
            r'\{["\']?answer["\']?\s*:\s*["\'](\d+)["\']\s*\}',
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

        # 2) fallback: explicit textual statements
        answer_patterns = [
            r'(?:final answer|my answer|the answer|answer)["\']?\s*(?:is|:)\s*["\']?(\d+)["\']?',
            r'(?:option|choice)["\']?\s*["\']?(\d+)["\']?(?:\s+is correct|\s+is the answer)',
            r'(?:I choose|I select|select|choose)["\']?\s*(?:option|choice)?\s*["\']?(\d+)["\']?',
            r"correct answer.*?(?:is|:)\s*["\']?(\d+)["\']?",
        ]

        search_text = response[-100:] if len(response) > 100 else response

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

        return result

    def test_single_item(self, item: Dict, language: str = "english") -> Dict:
        """Run a single question through the model and compute correctness."""
        prompt = self.format_question(item, language)
        result = self.generate_answer(prompt)

        correct_answer = int(item["answer_index"])
        predicted_answer = result["predicted_answer"]
        is_correct = (
            predicted_answer == correct_answer if predicted_answer is not None else False
        )

        rag_meta = getattr(self, "_last_rag_meta", None) or {}
        rag_base_keys = rag_meta.get("retrieved_base_keys")
        rag_sims = rag_meta.get("retrieved_similarities")
        rag_idxs = rag_meta.get("retrieved_indices")

        return {
            "base_key": item["base_key"],
            "language": language,
            "context": item[language]["context"],
            "question": item[language]["question"],
            "correct_answer": correct_answer,
            "predicted_answer": predicted_answer,
            "is_correct": is_correct,
            "full_response": result["full_response"],
            "thinking_process": result["thinking_process"],
            "has_thinking": result["has_thinking"],
            "has_json": result["has_json"],
            "generation_time": result["generation_time"],
            "knowledge_points": ", ".join(item.get("knowledge_points", [])),
            "options": str(item[language].get("options", [])),
            "used_rag": self.use_rag,
            "rag_k": self.rag_k if self.use_rag else None,
            "rag_retrieved_base_keys": json.dumps(rag_base_keys)
            if rag_base_keys is not None
            else None,
            "rag_retrieved_similarities": json.dumps(rag_sims)
            if rag_sims is not None
            else None,
            "rag_retrieved_indices": json.dumps(rag_idxs)
            if rag_idxs is not None
            else None,
        }

    def test_dataset(
        self,
        data: List[Dict],
        languages: List[str] = ["english"],
        limit: int = None,
        checkpoint_dir: str = "./checkpoints",
        checkpoint_interval: int = 10,
        resume: bool = False,
        realtime_csv: bool = True,
        output_dir: str = "./results",
    ) -> pd.DataFrame:
        """
        Run evaluation on a dataset.

        Features:
        - Checkpointing (pickle).
        - Resume from last checkpoint.
        - Optional real-time CSV append logging.
        """
        if limit:
            data = data[:limit]

        os.makedirs(checkpoint_dir, exist_ok=True)
        os.makedirs(output_dir, exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_short_name = self.model_name.split("/")[-1]
        rag_suffix = f"_rag_k{self.rag_k}" if self.use_rag else "_baseline"

        realtime_csv_file = None
        if realtime_csv:
            realtime_csv_file = f"{output_dir}/{model_short_name}_realtime_{timestamp}.csv"
            print(f"[INFO] Realtime CSV file: {realtime_csv_file}")

        checkpoint_file = os.path.join(
            checkpoint_dir, f"{model_short_name}_checkpoint.pkl"
        )

        results = []
        completed_items = set()
        csv_exists = False

        # Resume from checkpoint
        if resume and os.path.exists(checkpoint_file):
            try:
                with open(checkpoint_file, "rb") as f:
                    checkpoint_data = pickle.load(f)
                    results = checkpoint_data["results"]
                    completed_items = checkpoint_data["completed_items"]
                    print(
                        f"[INFO] Resumed from checkpoint. Completed: {len(completed_items)}"
                    )

                    if realtime_csv and results:
                        pd.DataFrame(results).to_csv(
                            realtime_csv_file,
                            index=False,
                            encoding="utf-8-sig",
                        )
                        csv_exists = True
            except Exception as e:
                print(f"[WARN] Failed to load checkpoint: {e}")

        total_tests = len(data) * len(languages)
        remaining_tests = total_tests - len(completed_items)

        print(
            f"\n[INFO] Starting evaluation "
            f"(total: {total_tests}, remaining: {remaining_tests})"
        )

        try:
            with tqdm(total=remaining_tests, desc="Progress") as pbar:
                for item in data:
                    for lang in languages:
                        test_id = f"{item['base_key']}_{lang}"
                        if test_id in completed_items:
                            continue

                        try:
                            result = self.test_single_item(item, lang)
                            results.append(result)
                            completed_items.add(test_id)

                            if realtime_csv and realtime_csv_file:
                                result_df = pd.DataFrame([result])
                                result_df.to_csv(
                                    realtime_csv_file,
                                    mode="a",
                                    header=not csv_exists,
                                    index=False,
                                    encoding="utf-8-sig",
                                )
                                csv_exists = True

                            if len(results) % checkpoint_interval == 0:
                                self._save_checkpoint(
                                    checkpoint_file, results, completed_items
                                )
                                if realtime_csv:
                                    print(
                                        f"\n[INFO] Checkpoint saved + CSV updated "
                                        f"({len(results)}/{total_tests})"
                                    )
                                else:
                                    print(
                                        f"\n[INFO] Checkpoint saved "
                                        f"({len(results)}/{total_tests})"
                                    )

                        except KeyboardInterrupt:
                            print("\n[INFO] KeyboardInterrupt detected.")
                            self._save_checkpoint(
                                checkpoint_file, results, completed_items
                            )
                            print("[INFO] Progress saved to checkpoint.")
                            if realtime_csv:
                                print(f"[INFO] CSV file: {realtime_csv_file}")

                            df = pd.DataFrame(results)
                            return df

                        except Exception as e:
                            print(f"\n[ERROR] {test_id}: {str(e)}")
                            error_result = {
                                "base_key": item["base_key"],
                                "language": lang,
                                "is_correct": False,
                                "full_response": f"ERROR: {str(e)}",
                            }
                            results.append(error_result)

                            if realtime_csv and realtime_csv_file:
                                pd.DataFrame([error_result]).to_csv(
                                    realtime_csv_file,
                                    mode="a",
                                    header=not csv_exists,
                                    index=False,
                                    encoding="utf-8-sig",
                                )
                                csv_exists = True

                        pbar.update(1)

            print("\n[INFO] Evaluation finished.")
            if realtime_csv:
                print(f"[INFO] Realtime CSV stored at: {realtime_csv_file}")

            if len(completed_items) == total_tests and os.path.exists(checkpoint_file):
                os.remove(checkpoint_file)
                print("[INFO] Checkpoint removed (all tests completed).")

        except Exception as e:
            print(f"\n[ERROR] Evaluation error: {e}")
            self._save_checkpoint(checkpoint_file, results, completed_items)

        df = pd.DataFrame(results)
        return df

    def _save_checkpoint(
        self, checkpoint_file: str, results: List, completed_items: set
    ):
        """Save checkpoint to disk."""
        try:
            with open(checkpoint_file, "wb") as f:
                pickle.dump(
                    {
                        "results": results,
                        "completed_items": completed_items,
                        "timestamp": datetime.now().isoformat(),
                    },
                    f,
                )
        except Exception as e:
            print(f"[WARN] Failed to save checkpoint: {e}")

    def _print_statistics(self, df: pd.DataFrame):
        """Print summary statistics for a single model run."""
        print("\n" + "=" * 80)
        print("Evaluation summary")
        print("=" * 80)

        print(f"\nTotal tests: {len(df)}")
        print(f"Correct: {df['is_correct'].sum()}")
        print(f"Accuracy: {df['is_correct'].mean() * 100:.2f}%")
        print(f"Average time: {df['generation_time'].mean():.2f} seconds")
        total_time = df["generation_time"].sum()
        print(
            f"Total time: {total_time:.1f} seconds "
            f"({total_time / 60:.1f} minutes)"
        )

        if self.is_thinking:
            has_thinking = df["has_thinking"].sum()
            print(
                f"Has thinking trace: {has_thinking} "
                f"({has_thinking / len(df) * 100:.1f}%)"
            )

        has_json = df["has_json"].sum()
        print(
            f"JSON answer format detected: {has_json} "
            f"({has_json / len(df) * 100:.1f}%)"
        )

        predicted = df["predicted_answer"].notna().sum()
        print(
            f"Successfully extracted answers: {predicted} "
            f"({predicted / len(df) * 100:.1f}%)"
        )

        if len(df["language"].unique()) > 1:
            print("\nPer-language statistics:")
            lang_stats = df.groupby("language").agg(
                {
                    "is_correct": ["count", "sum", "mean"],
                    "generation_time": "mean",
                    "has_json": "sum",
                }
            ).round(3)
            lang_stats.columns = [
                "num_questions",
                "num_correct",
                "accuracy",
                "avg_time_sec",
                "num_json",
            ]
            lang_stats["accuracy"] = (lang_stats["accuracy"] * 100).round(2)
            print(lang_stats)

    def _analyze_errors(self, df: pd.DataFrame):
        """Analyze incorrect predictions."""
        errors = df[df["is_correct"] == False].copy()

        print("\n" + "=" * 80)
        print(
            f"Error analysis (total {len(errors)} errors, "
            f"error rate {len(errors) / len(df) * 100:.2f}%)"
        )
        print("=" * 80)

        no_answer = errors["predicted_answer"].isna().sum()
        wrong_answer = len(errors) - no_answer

        print("\nError types:")
        print(
            f"  - Answer could not be extracted: {no_answer} "
            f"({no_answer / len(errors) * 100:.1f}%)"
        )
        print(
            f"  - Answer extracted but incorrect: {wrong_answer} "
            f"({wrong_answer / len(errors) * 100:.1f}%)"
        )


def load_jsonl_data(file_path: str) -> List[Dict]:
    """Load JSONL data file into a list of dicts."""
    data = []
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                data.append(json.loads(line))

    print(f"[INFO] Loaded {len(data)} questions from {file_path}")
    return data


def compare_models(
    data: List[Dict],
    model_configs: List[Dict],
    languages: List[str] = ["english"],
    output_dir: str = "./results",
    limit: int = None,
    checkpoint_dir: str = "./checkpoints",
    resume: bool = False,
):
    """
    Compare multiple models on the same dataset.

    Args:
        data: Test data (list of question dicts).
        model_configs: List of model config dicts.
        languages: Languages to test.
        output_dir: Directory for detailed results.
        limit: Limit the number of test items.
        checkpoint_dir: Directory for checkpoints.
        resume: Whether to resume from checkpoints.
    """
    os.makedirs(output_dir, exist_ok=True)

    all_results = {}
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    for config in model_configs:
        model_name = config["name"]
        use_quant = config.get("use_quantization", False)
        quant_bits = config.get("quantization_bits", 4)
        local_path = config.get("local_path", None)
        temperature = config.get("temperature", 0.0)
        use_rag = config.get("use_rag", False)
        rag_system = config.get("rag_system", None)
        rag_k = config.get("rag_k", 3)

        print("\n" + "=" * 80)
        print(f"Evaluating model: {model_name}")
        if use_quant:
            print(f"Quantization: {quant_bits}-bit")
        if local_path:
            print(f"Local path: {local_path}")
        print(f"Temperature: {temperature}")
        print("=" * 80 + "\n")

        try:
            tester = ProbabilityTestFramework(
                model_name,
                use_quantization=use_quant,
                quantization_bits=quant_bits,
                local_model_path=local_path,
                temperature=temperature,
                use_rag=use_rag,
                rag_system=rag_system,
                rag_k=rag_k,
            )

            results_df = tester.test_dataset(
                data,
                languages,
                limit,
                checkpoint_dir=checkpoint_dir,
                checkpoint_interval=10,
                resume=resume,
            )

            model_short_name = model_name.split("/")[-1]
            quant_suffix = f"_q{quant_bits}" if use_quant else ""
            rag_suffix = f"_rag_k{rag_k}" if use_rag else "_baseline"
            temp_suffix = f"_t{temperature}".replace(".", "")
            output_file = (
                f"{output_dir}/{model_short_name}"
                f"{quant_suffix}{temp_suffix}_{timestamp}.csv"
            )
            results_df.to_csv(output_file, index=False, encoding="utf-8-sig")
            print(f"\n[INFO] Detailed results saved to: {output_file}")

            all_results[f"{model_name}{quant_suffix}"] = results_df

            del tester
            torch.cuda.empty_cache()

        except Exception as e:
            print(f"\n[ERROR] Model {model_name} evaluation failed: {str(e)}")
            import traceback

            traceback.print_exc()

    if len(all_results) > 1:
        generate_comparison_report(all_results, output_dir, timestamp)

    return all_results


def generate_comparison_report(
    all_results: Dict[str, pd.DataFrame],
    output_dir: str,
    timestamp: str,
):
    """Generate a comparison report across multiple models."""
    print("\n" + "=" * 80)
    print("Model comparison report")
    print("=" * 80)

    comparison_data = []

    for model_name, df in all_results.items():
        comparison_data.append(
            {
                "model": model_name.split("/")[-1],
                "num_questions": len(df),
                "accuracy(%)": f"{df['is_correct'].mean()*100:.2f}",
                "avg_time(sec)": f"{df['generation_time'].mean():.2f}",
                "total_time(min)": f"{df['generation_time'].sum()/60:.1f}",
                "num_successful_predictions": df["predicted_answer"]
                .notna()
                .sum(),
                "num_json_answers": df["has_json"].sum(),
            }
        )

    comparison_df = pd.DataFrame(comparison_data)
    print("\n", comparison_df.to_string(index=False))

    report_file = f"{output_dir}/comparison_report_{timestamp}.csv"
    comparison_df.to_csv(report_file, index=False, encoding="utf-8-sig")
    print(f"\n[INFO] Comparison report saved to: {report_file}")


# ============================================================================
# Main entry
# ============================================================================

if __name__ == "__main__":
    print("=" * 80)
    print("Probability MCQ evaluation framework v3.0")
    print(" - Supports checkpointing and resume")
    print(" - Supports local model loading")
    print(" - Supports temperature = 0 (deterministic output)")
    print("=" * 80)

    # 1. Load test data
    data = load_jsonl_data("probability_test_set.jsonl")

    # 2. Basic config
    QUICK_TEST = False
    test_limit = 10 if QUICK_TEST else None

    test_languages = ["english", "danish"]

    model_configs = [
        {
            "name": "Qwen/Qwen3-14B",
            "local_path": "/root/models/qwen3-14b-q4",
            "use_quantization": True,
            "quantization_bits": 4,
            "timeout": 120,
            "temperature": 0.0,
        },
        # Add more models here if needed
    ]

    print("\nConfiguration:")
    print(
        f"  Test mode: "
        f"{'Quick test (first ' + str(test_limit) + ' items)' if QUICK_TEST else 'Full test'}"
    )
    print(f"  Languages: {', '.join(test_languages)}")
    print(f"  Number of models: {len(model_configs)}")
    print("  Checkpoint: enabled (every 10 questions)")
    print("  Interrupt: Ctrl+C to safely interrupt and save progress")

    results = compare_models(
        data=data,
        model_configs=model_configs,
        languages=test_languages,
        output_dir="./probability_test_results",
        checkpoint_dir="./checkpoints",
        limit=test_limit,
        resume=False,
    )

    print("\n" + "=" * 80)
    print("All evaluations finished.")
    print("=" * 80)
