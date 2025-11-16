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
import json
import re
import argparse
from functools import partial
from pathlib import Path

# ============================================================================
# 模型加载
# ============================================================================

def load_model_and_tokenizer(model_name="Qwen/Qwen3-14B"):
    print(f"🚀 加载模型: {model_name}")
    
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
    
    # 强制设置 pad_token
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        model.config.pad_token_id = tokenizer.pad_token_id
    
    print(f"✓ 模型已加载")
    return model, tokenizer


# ============================================================================
# LoRA配置
# ============================================================================

def setup_lora(model, lora_r=64, lora_alpha=64):
    print(f"⚙️ 配置LoRA (rank={lora_r}, alpha={lora_alpha})")
    
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
    print(f"📊 可训练参数: {trainable:,} ({100*trainable/total:.2f}%)")
    
    return model


# ============================================================================
# 数据处理：固定 8192，上下文内只对「非 pad 且 assistant 部分」算 loss
# ============================================================================

def format_conversation(example, tokenizer, max_length=8192):
    """
    - 统一 padding + truncation 到 max_length
    - labels:
        * pad 位置 = -100（不算 loss）
        * user/system 位置 = -100（不算 loss）
        * 认为是 assistant 的位置 = token id（算 loss）
    """
    messages = example["messages"]

    # 1) 拼成聊天模板文本（完整对话）
    formatted_text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=False
    )

    # 2) Tokenize → 固定 max_length
    tokenized = tokenizer(
        formatted_text,
        truncation=True,
        max_length=max_length,
        padding="max_length",
        return_tensors=None
    )

    input_ids = tokenized["input_ids"]
    attention_mask = tokenized["attention_mask"]

    # 3) 先基于 attention_mask，把 pad 位置全部设为 -100
    labels = [
        (tok if mask == 1 else -100)
        for tok, mask in zip(input_ids, attention_mask)
    ]

    # ========= selective supervision：只给 assistant 部分算 loss =========
    # 构造不含 assistant 的模板，用它的长度来估计「非 assistant 的前缀长度」
    user_only = [m for m in messages if m["role"] != "assistant"]

    if user_only:
        user_text = tokenizer.apply_chat_template(
            user_only,
            tokenize=False,
            add_generation_prompt=True  # 让模型准备回答的位置
        )

        user_tokens = tokenizer(
            user_text,
            truncation=True,
            max_length=max_length,
            padding="max_length",
            return_tensors=None
        )["input_ids"]

        # 找 user_tokens 中真正内容的长度（第一个 pad 出现的位置）
        user_len = 0
        for t in user_tokens:
            if t == tokenizer.pad_token_id:
                break
            user_len += 1

        # 把完整序列中“前 user_len 个 token”都视作非 assistant 部分 → label = -100
        cutoff = min(user_len, len(labels))
        for i in range(cutoff):
            labels[i] = -100

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels
    }


def prepare_dataset(data_path, tokenizer, max_length=8192):
    print(f"📂 加载数据: {data_path}")
    dataset = load_dataset('json', data_files=str(data_path), split='train')

    dataset = dataset.map(
        partial(format_conversation, tokenizer=tokenizer, max_length=max_length),
        remove_columns=dataset.column_names,
        desc="格式化数据"
    )

    # 过滤掉没有训练信号的样本（labels 全是 -100 的样本）
    dataset = dataset.filter(lambda x: any(l != -100 for l in x["labels"]))
    print(f"✓ 数据集大小: {len(dataset)}")

    return dataset


# ============================================================================
# 训练配置（自动记录 loss / eval_loss 到日志和 tensorboard）
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
        eval_strategy="steps",   # 按步数做 eval，记录 eval_loss
        eval_steps=eval_steps,
        save_strategy="steps",
        save_steps=save_steps,
        logging_steps=logging_steps,   # 训练 loss 记录间隔
        max_grad_norm=1.0,
        save_total_limit=3,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        report_to="tensorboard",       # 会写 TensorBoard 日志
        dataloader_num_workers=4,
        remove_unused_columns=False
    )


# ============================================================================
# 主程序
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="Qwen3-14B QLoRA训练")
    
    parser.add_argument("--model_name", default="Qwen/Qwen3-14B")
    parser.add_argument("--lora_r", type=int, default=256)
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
    print("🚀 Qwen3-14B QLoRA训练 - 8192上下文 + assistant-only loss 版（无MCQ）")
    print("=" * 80)
    
    # 1. 加载模型
    model, tokenizer = load_model_and_tokenizer(args.model_name)

    # 2. 固定使用 8192 上下文长度
    ctx_len = 8192
    print(f"🧠 使用上下文长度: {ctx_len}")
    
    # 3. LoRA
    model = setup_lora(model, args.lora_r, args.lora_alpha)
    
    # 4. 数据集（使用 8192 max_length）
    train_dataset = prepare_dataset(args.train_file, tokenizer, max_length=ctx_len)
    eval_dataset = prepare_dataset(args.val_file, tokenizer, max_length=ctx_len)
    
    # 5. 训练参数
    training_args = get_training_args(
        output_dir=args.output_dir,
        learning_rate=args.learning_rate,
        num_epochs=args.num_epochs,
        batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        eval_steps=args.eval_steps,
        save_steps=args.save_steps
    )
    
    # 6. DataCollator
    data_collator = DataCollatorWithPadding(tokenizer)
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
    )
    
    # 7. 开始训练
    trainer.train()
    
    final_path = f"{args.output_dir}/final_model"
    trainer.save_model(final_path)
    tokenizer.save_pretrained(final_path)
    
    print("训练完成！")


if __name__ == "__main__":
    main()
