import random
from torch.utils.data import Dataset, Subset
from transformers import Trainer, TrainingArguments, logging
import torch
from datasets import load_dataset, DatasetDict
import wandb
from safetensors.torch import load_file

# Ccubedモデルとトークナイザーのインポート
from transformers import AutoTokenizer, AutoModelForCausalLM
from c_cubed_src.modeling_c_cubed import CcubedForConditionalGeneration
from c_cubed_src.tokenization_c_cubed import CcubedDualTokenizer

from accelerate import infer_auto_device_map, dispatch_model
import os
import psutil
import subprocess
import re
import threading
import time
import psutil
from datetime import datetime

phase = 1

try:
    tokenizer = CcubedDualTokenizer.from_pretrained("./tokenizer_production", trust_remote_code=True, use_fast=False)
except:
    print("[INFO] Failed to load tokenizer with use_fast=False. Retrying with use_fast=True.")
    tokenizer = CcubedDualTokenizer.from_pretrained("./tokenizer_production", trust_remote_code=True)

tokenizer.text_tokenizer.pad_token = tokenizer.text_tokenizer.eos_token
tokenizer.context_tokenizer.pad_token = tokenizer.context_tokenizer.eos_token
tokenizer.text_tokenizer.pad_token_id = tokenizer.text_tokenizer.eos_token_id
tokenizer.context_tokenizer.pad_token_id = tokenizer.context_tokenizer.eos_token_id


if phase == 1:
    # データセットの読み込み
    dataset = load_dataset("sudy-super/c_cubed_restoration")

    # データセットの取得
    train_data = dataset["train"]
    val_data = dataset["validation"]
elif phase == 2:
    dataset_long = load_dataset("sudy-super/c_cubed_finetune")
    dataset_ja = load_dataset("sudy-super/c_cubed_oasst2_ja")
    dataset_en = load_dataset("sudy-super/c_cubed_oasst2_en")

    # 各データセットのスプリットを取得
    train_data_long = dataset_long["train"]
    val_data_long = dataset_long["validation"]

    train_data_ja = dataset_ja["train"]
    val_data_ja = dataset_ja["validation"]

    train_data_en = dataset_en["train"]
    val_data_en = dataset_en["validation"]

    print("[INFO] Datasets loaded successfully.")


# `generate_inputs`関数をバッチ処理に対応
def generate_inputs(batch):
    conversations_list = batch["conversations"]
    contexts_list = batch.get("context", [""] * len(conversations_list))

    contexts = []
    texts = []
    for context, conversations in zip(contexts_list, conversations_list): # for context, conversations in zip(batch.get("context", [""]), batch["conversations"]):
        if not context:
            context = tokenizer.context_tokenizer.pad_token # ""  # contextがNoneまたは空の場合、空文字列に設定

        text = "<|im_start|>system\nYou are Qwen, created by Alibaba Cloud. You are a helpful assistant.<|im_end|>"
        """
        for c in conversations:
            if c["from"] == "user":
                text += f"<|user|>{c['value']}</s>\n"
            elif c["from"] == "assistant":
                text += f"<|assistant|>{c['value']}</s>"
        """
        for c in conversations:
            if c["from"] == "user":
                text += f"\n<|im_start|>user\n{c['value']}<|im_end|>"
            elif c["from"] == "assistant":
                text += f"\n<|im_start|>assistant\n{c['value']}<|im_end|>"
        contexts.append(context)
        texts.append(text)
    return {'context': contexts, 'text': texts}

def generate_inputs_for_restoration(batch):
    conversations_list = batch["text"]
    contexts_list = batch["text"]

    contexts = []
    texts = []
    for context, conversations in zip(contexts_list, conversations_list): # for context, conversations in zip(batch.get("context", [""]), batch["conversations"]):
        if not context:
            context = tokenizer.context_tokenizer.pad_token # ""  # contextがNoneまたは空の場合、空文字列に設定

        text = "<|im_start|>system\nYou are Qwen, created by Alibaba Cloud. You are a helpful assistant.<|im_end|>"
        
        text += f"\n<|im_start|>user\nPlease repeat the input context.<|im_end|>"
        text += f"\n<|im_start|>assistant\n{context}<|im_end|>"
        contexts.append(context)
        texts.append(text)
    return {'context': contexts, 'text': texts}

# `tokenize`関数をバッチ処理に対応
def tokenize(batch):
    # 最大トークン数の設定
    max_context_tokens = 131072

    # 各サンプルに対してcontextのトークン数を確認し、必要に応じてカット
    truncated_contexts = []
    for context in batch['context']:
        # contextを単独でトークン化してトークン数を確認
        context_tokens = tokenizer.context_tokenizer.tokenize(context)
        if len(context_tokens) > max_context_tokens:
            # トークン数が65536を超える場合、カット
            context = tokenizer.context_tokenizer.convert_tokens_to_string(context_tokens[:max_context_tokens])
        truncated_contexts.append(context)
    
    text_tokenized = tokenizer.text_tokenizer(batch['text'], add_special_tokens=False)
    text_lengths = [len(ids) for ids in text_tokenized['input_ids']]

    # contextをカットしたリストを用いて最終的にトークン化
    tokenized_outputs = tokenizer(
        context=truncated_contexts,
        text=batch['text'],
        truncation=True,
        max_length=max_context_tokens,
        padding=False,
        add_special_tokens=False,
    )

    tokenized_outputs['length'] = [len(ids) for ids in tokenized_outputs['input_ids']]
    tokenized_outputs['text_length'] = text_lengths

    return tokenized_outputs


def preprocess_and_tokenize_with_context(dataset, desc_prefix):
    dataset = dataset.map(
        generate_inputs_for_restoration,
        batched=True,
        num_proc=8,
        desc=f"Generating inputs for {desc_prefix}",
        load_from_cache_file=True
    ).filter(lambda x: x['text'] != '', num_proc=8).filter(lambda x: x['context'] != '', num_proc=8).filter(lambda x: x['context'] != tokenizer.context_tokenizer.pad_token, num_proc=8)

    dataset = dataset.map(
        tokenize,
        batched=True,
        num_proc=8,
        remove_columns=dataset.column_names,
        desc=f"Tokenizing {desc_prefix}",
        load_from_cache_file=True
    )

    dataset = dataset.filter(lambda x: x['text_length'] <= max_text_length, num_proc=8)
    return dataset

# 前処理とトークン化を新しいデータセットにも適用
def preprocess_and_tokenize(dataset, desc_prefix):
    dataset = dataset.map(
        generate_inputs,
        batched=True,
        num_proc=8,
        desc=f"Generating inputs for {desc_prefix}",
        load_from_cache_file=True
    ).filter(lambda x: x['text'] != '', num_proc=8)

    dataset = dataset.map(
        tokenize,
        batched=True,
        num_proc=8,
        remove_columns=dataset.column_names,
        desc=f"Tokenizing {desc_prefix}",
        load_from_cache_file=True
    )

    dataset = dataset.filter(lambda x: x['text_length'] <= max_text_length, num_proc=8)
    return dataset

from datasets import concatenate_datasets

if phase == 1:
    max_text_length = 131072
    # データのシャッフルとフィルタリング、バッチ処理対応
    train_data = train_data.shuffle(seed=42)
    val_data = val_data.shuffle(seed=42)

    train_data_phase1 = preprocess_and_tokenize_with_context(train_data, "train_data")
    val_data_phase1 = preprocess_and_tokenize_with_context(val_data, "val_data")

    print("[INFO] Data preprocessing and tokenization completed.")

    # データセットの件数をカウントして表示
    print(f"Number of train samples (phase1): {len(train_data_phase1)}")
    print(f"Number of validation samples (phase1): {len(val_data_phase1)}")
elif phase == 2:
    max_text_length = 1024
    train_data_long = train_data_long.shuffle(seed=42)
    val_data_long = val_data_long.shuffle(seed=42)

    train_data_ja = train_data_ja.shuffle(seed=42)
    val_data_ja = val_data_ja.shuffle(seed=42)

    train_data_en = train_data_en.shuffle(seed=42)
    val_data_en = val_data_en.shuffle(seed=42)

    train_data_long = preprocess_and_tokenize(train_data_long, "train_data_long")
    val_data_long = preprocess_and_tokenize(val_data_long, "val_data_long")

    train_data_ja = preprocess_and_tokenize(train_data_ja, "train_data_ja")
    val_data_ja = preprocess_and_tokenize(val_data_ja, "val_data_ja")

    train_data_en = preprocess_and_tokenize(train_data_en, "train_data_en")
    val_data_en = preprocess_and_tokenize(val_data_en, "val_data_en")

    print("[INFO] Phase2 data preprocessing and tokenization completed.")

    # データセットの前処理と結合
    train_data_phase2 = concatenate_datasets([
        train_data_long,
        train_data_ja,
        train_data_en
    ])

    val_data_phase2 = concatenate_datasets([
        val_data_long,
        val_data_ja, 
        val_data_en
    ])

    train_data_phase2 = train_data_phase2.shuffle(seed=42)
    val_data_phase2 = val_data_phase2.shuffle(seed=42)

    print(f"Number of train samples(long): {len(train_data_long)}")
    print(f"Number of validation samples(long): {len(val_data_long)}")
    print(f"Number of train samples(ja): {len(train_data_ja)}")
    print(f"Number of validation samples(ja): {len(val_data_ja)}")
    print(f"Number of train samples(en): {len(train_data_en)}")
    print(f"Number of validation samples(en): {len(val_data_en)}")

    print(f"Number of train samples (phase2): {len(train_data_phase2)}")
    print(f"Number of validation samples (phase2): {len(val_data_phase2)}")

if phase == 1:
    # データセットを作成
    dataset_dict = DatasetDict({
        "train": Dataset.from_list(train_data_phase1),
        "validation": Dataset.from_list(val_data_phase1)
    })

    # データセットをHugging Face Hubにアップロード
    dataset_name = "sudy-super/c_cubed_restoration_tokenized"
    dataset_dict.push_to_hub(dataset_name)
elif phase == 2:
    dataset_dict = DatasetDict({
        "train": Dataset.from_list(train_data_phase2),
        "validation": Dataset.from_list(val_data_phase2)
    })

    dataset_name = "sudy-super/c_cubed_finetune_tokenized"
    dataset_dict.push_to_hub(dataset_name)