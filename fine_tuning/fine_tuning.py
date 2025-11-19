#!/usr/bin/env python3
import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
    Trainer,
    DataCollatorWithPadding
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from datasets import load_dataset
import argparse
from functools import partial
from pathlib import Path

# ============================================================================
# Model loading
# ============================================================================

def load_model_and_tokenizer(model_name="Qwen/Qwen3-14B"):
    print(f"Loading base model: {model_name}")
    
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16
    )
    
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
        torch_dtype=torch.bfloat16
    )
    
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        trust_remote_code=True,
        padding_side="right"
    )
    
    # Ensure pad_token is set
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        model.config.pad_token_id = tokenizer.pad_token_id
    
    print("Base model loaded.")
    return model, tokenizer


# ============================================================================
# LoRA configuration
# ============================================================================

def setup_lora(model, lora_r=64, lora_alpha=64):
    print(f"Configuring LoRA (rank={lora_r}, alpha={lora_alpha})")
    
    model = prepare_model_for_kbit_training(model)
    
    lora_config = LoraConfig(
        r=lora_r,
        lora_alpha=lora_alpha,
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj"
        ],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM"
    )
    
    model = get_peft_model(model, lora_config)
    
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"Trainable parameters: {trainable:,} ({100*trainable/total:.2f}%)")
    
    return model


# ============================================================================
# Data processing: fixed 8192, loss only on non-pad assistant tokens
# ============================================================================

def format_conversation(example, tokenizer, max_length=8192):
    """
    - Pad + truncate to max_length
    - labels:
        * pad positions = -100 (ignored in loss)
        * user/system positions = -100
        * assistant tokens = token id (contribute to loss)
    """
    messages = example["messages"]

    # 1) Render full chat using the model's chat template
    formatted_text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=False
    )

    # 2) Tokenize → fixed max_length
    tokenized = tokenizer(
        formatted_text,
        truncation=True,
        max_length=max_length,
        padding="max_length",
        return_tensors=None
    )

    input_ids = tokenized["input_ids"]
    attention_mask = tokenized["attention_mask"]

    # 3) Initialize labels: copy ids where attention_mask == 1, else -100
    labels = [
        (tok if mask == 1 else -100)
        for tok, mask in zip(input_ids, attention_mask)
    ]

    # Selective supervision: only compute loss on assistant tokens.
    # Build a "non-assistant" template and use its length as a cutoff.
    user_only = [m for m in messages if m["role"] != "assistant"]

    if user_only:
        user_text = tokenizer.apply_chat_template(
            user_only,
            tokenize=False,
            add_generation_prompt=True  # prepare model to answer
        )

        user_tokens = tokenizer(
            user_text,
            truncation=True,
            max_length=max_length,
            padding="max_length",
            return_tensors=None
        )["input_ids"]

        # Compute the actual prefix length (up to first pad)
        user_len = 0
        for t in user_tokens:
            if t == tokenizer.pad_token_id:
                break
            user_len += 1

        # Treat the first user_len tokens as non-assistant → label = -100
        cutoff = min(user_len, len(labels))
        for i in range(cutoff):
            labels[i] = -100

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels
    }


def prepare_dataset(data_path, tokenizer, max_length=8192):
    print(f"Loading dataset from: {data_path}")
    dataset = load_dataset('json', data_files=str(data_path), split='train')

    dataset = dataset.map(
        partial(format_conversation, tokenizer=tokenizer, max_length=max_length),
        remove_columns=dataset.column_names,
        desc="Formatting dataset"
    )

    # Filter out samples with no training signal (all labels == -100)
    dataset = dataset.filter(lambda x: any(l != -100 for l in x["labels"]))
    print(f"Dataset size after filtering: {len(dataset)}")

    return dataset


# ============================================================================
# Training configuration (loss / eval_loss logged to TensorBoard)
# ============================================================================

def get_training_args(
    output_dir="./qwen3-qlora-output",
    learning_rate=1e-5,
    num_epochs=5,
    batch_size=16,
    gradient_accumulation_steps=2,
    eval_steps=25,
    save_steps=50,
    logging_steps=5
):
    return TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=num_epochs,
        learning_rate=learning_rate,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        gradient_checkpointing=True,
        optim="paged_adamw_8bit",
        bf16=True,
        eval_strategy="steps",
        eval_steps=eval_steps,
        save_strategy="steps",
        save_steps=save_steps,
        logging_steps=logging_steps,
        max_grad_norm=1.0,
        save_total_limit=3,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        report_to="tensorboard",
        dataloader_num_workers=4,
        remove_unused_columns=False
    )


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="Qwen3-14B QLoRA training (assistant-only loss, 8192 context)")
    
    parser.add_argument("--model_name", default="Qwen/Qwen3-14B")
    parser.add_argument("--lora_r", type=int, default=64)
    parser.add_argument("--lora_alpha", type=int, default=64)
    parser.add_argument("--train_file", required=True)
    parser.add_argument("--val_file", required=True)
    parser.add_argument("--output_dir", default="./qwen3-qlora-output")
    parser.add_argument("--learning_rate", type=float, default=1e-5)
    parser.add_argument("--num_epochs", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=2)
    parser.add_argument("--eval_steps", type=int, default=25)
    parser.add_argument("--save_steps", type=int, default=50)
    
    args = parser.parse_args()
    
    print("=" * 80)
    print("Qwen3-14B QLoRA training - 8192 context, assistant-only loss (no MCQ formatting)")
    print("=" * 80)
    
    # 1. Load base model
    model, tokenizer = load_model_and_tokenizer(args.model_name)

    # 2. Fixed context length
    ctx_len = 8192
    print(f"Using context length: {ctx_len}")
    
    # 3. Set up LoRA
    model = setup_lora(model, args.lora_r, args.lora_alpha)
    
    # 4. Build datasets
    train_dataset = prepare_dataset(args.train_file, tokenizer, max_length=ctx_len)
    eval_dataset = prepare_dataset(args.val_file, tokenizer, max_length=ctx_len)
    
    # 5. Training arguments
    training_args = get_training_args(
        output_dir=args.output_dir,
        learning_rate=args.learning_rate,
        num_epochs=args.num_epochs,
        batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        eval_steps=args.eval_steps,
        save_steps=args.save_steps
    )
    
    # 6. Data collator
    data_collator = DataCollatorWithPadding(tokenizer)
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
    )
    
    # 7. Train
    trainer.train()
    
    final_path = f"{args.output_dir}/final_model"
    trainer.save_model(final_path)
    tokenizer.save_pretrained(final_path)
    
    print("Training finished.")
    print(f"Final model saved to: {final_path}")


if __name__ == "__main__":
    main()
