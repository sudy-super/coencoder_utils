import random
from torch.utils.data import Dataset, Subset
from transformers import Trainer, TrainingArguments, logging
import torch
from datasets import load_dataset
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

#import torch.distributed as dist

phase = 1

# DeepSpeedがtorch.distributedの初期化を行うため、その後でランクを取得します
#dist.init_process_group(backend='nccl')  # 必要に応じてバックエンドを指定

# グローバルランク0のプロセスのみでWandBを初期化
"""
if dist.get_rank() == 0:
    if phase == 1:
        wandb.init(project="c_cubed_phase1", name="1e-3_c_cubed_connector", entity="sudy_super")
    elif phase == 2:
        wandb.init(project="c_cubed_phase2", name="2e-5_c_cubed_lm", entity="sudy_super")
    else:
        raise ValueError("Invalid phase value. Must be 1 or 2.")
"""

torch.manual_seed(42)
torch.cuda.manual_seed(42)

if phase == 1:
    model_name = "sudy-super/coencoder_test2"
elif phase == 2:
    model_name = "sudy-super/coencoder_test2_phase1_2"
else:
    raise ValueError("Invalid phase value. Must be 1 or 2.")

# CoEncoderトークナイザーとモデルの読み込み
try:
    tokenizer = CcubedDualTokenizer.from_pretrained("./co_model_production", trust_remote_code=True, use_fast=False)
except:
    print("[INFO] Failed to load tokenizer with use_fast=False. Retrying with use_fast=True.")
    tokenizer = CcubedDualTokenizer.from_pretrained("./co_model_production", trust_remote_code=True)

model = CcubedForConditionalGeneration.from_pretrained(
    model_name,
    torch_dtype=torch.bfloat16,
    trust_remote_code=True,
    attn_implementation="flash_attention_2"
)


tokenizer.text_tokenizer.pad_token = tokenizer.text_tokenizer.eos_token
tokenizer.context_tokenizer.pad_token = tokenizer.context_tokenizer.eos_token
tokenizer.text_tokenizer.pad_token_id = tokenizer.text_tokenizer.eos_token_id
tokenizer.context_tokenizer.pad_token_id = tokenizer.context_tokenizer.eos_token_id

model.model_parallel = True
model.gradient_checkpointing_enable()
torch.autograd.set_detect_anomaly(True)

# context_towerとlanguage_modelの重みを凍結
for param in model.context_tower.parameters():
    param.requires_grad = False

for param in model.connector.parameters():
    param.requires_grad = True

for param in model.language_model.parameters():
    if phase == 1:
        param.requires_grad = False
    elif phase == 2:
        param.requires_grad = True
    else:
        raise ValueError("Invalid phase value. Must be 1 or 2.")

for name, param in model.named_parameters():
    if param.requires_grad:
        print(f"training param - {name}: {param.shape}")


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


# train_data_sorted = train_data_used.sort('length')

class DataCollatorAssistantWithContext:
    """
    1) context_input_ids, context_attention_mask を含む入力をパディング・整形し、
       コンテキストがすべて pad のときはそれを削除、部分的に pad のときは attention_mask を 0 に。
    2) メインテキスト (input_ids, attention_mask) もパディングし、labels もパディング。
    3) 「<|im_start|>assistant ~ <|im_end|>」以外のトークンを labels = -100 に置き換える。
    """

    def __init__(self, tokenizer):
        """
        tokenizer には以下のような想定:
            tokenizer.context_tokenizer
            tokenizer.text_tokenizer
            または context / text 共通で tokenizer を流用するなら
            tokenizer.pad(...) で分けて使ってもOK。

        ここでは例として tokenizer.context_tokenizer, tokenizer.text_tokenizer を使う想定。
        """
        self.tokenizer = tokenizer

        # 以下は「アシスタント部分のみを学習対象とする」ための特殊トークン設定
        self.start_token_id = tokenizer.convert_tokens_to_ids("<|im_start|>")
        self.end_token_id   = tokenizer.convert_tokens_to_ids("<|im_end|>")

        # "assistant" のトークン ID (必要に応じて変更)
        self.assistant_token_id = 77091

        # 改行のトークンID (必要に応じて変更)
        self.newline_token_id = 198

    def __call__(self, features):
        """
        features は下記形式のリストを想定:
        [
            {
                "context_input_ids": [...],
                "context_attention_mask": [...],  # 無い場合もあるので .get() などで扱う
                "input_ids": [...],
                "attention_mask": [...],
            },
            ...
        ]
        """

        # --------
        # 1) コンテキスト部分のパディングと削除判定
        # --------
        # context_input_ids, attention_mask が与えられていない場合も考慮
        context_features = []
        for f in features:
            if "context_input_ids" in f:
                context_features.append({
                    'input_ids': f["context_input_ids"],
                    'attention_mask': f.get("context_attention_mask", [1]*len(f["context_input_ids"]))
                })
            else:
                # コンテキストが無い場合は空配列（または適当に処理）を入れておく
                context_features.append({
                    'input_ids': [],
                    'attention_mask': []
                })

        # パディング (context用)
        context_batch = self.tokenizer.context_tokenizer.pad(
            context_features,
            padding=True,
            return_tensors="pt"
        )

        pad_token_id = self.tokenizer.context_tokenizer.pad_token_id

        # バッチ全体が pad かの判定
        # [batch_size, seq_len] => 行方向に all pad => さらに全サンプル True => batch 全体 True
        all_context_is_pad = (context_batch["input_ids"] == pad_token_id).all(dim=1).all()

        # --------
        # 2) テキスト部分のパディングと labels 作成 (まだ -100 処理はしない)
        # --------
        text_features = [{
            'input_ids': f['input_ids'],
            'attention_mask': f['attention_mask']
        } for f in features]

        text_batch = self.tokenizer.text_tokenizer.pad(
            text_features,
            padding=True,
            return_tensors="pt"
        )

        # ラベルも同様にパディング (初期値は input_ids と同じ)
        label_features = [{'input_ids': f['input_ids']} for f in features]
        labels_batch = self.tokenizer.text_tokenizer.pad(
            label_features,
            padding=True,
            return_tensors="pt"
        )

        # --------
        # 3) コンテキスト削除 or attention_mask 調整
        # --------
        if all_context_is_pad:
            # バッチ内のコンテキストがすべて pad トークンなら context を取り除く
            batch = {
                'input_ids': text_batch['input_ids'],
                'attention_mask': text_batch['attention_mask'],
                'labels': labels_batch['input_ids'],
            }
        else:
            # バッチ内の一部が pad のサンプルだけ attention_mask = 0 に
            is_all_padding_sample = (context_batch['input_ids'] == pad_token_id).all(dim=1)
            context_batch['attention_mask'][is_all_padding_sample] = 0

            batch = {
                'context_input_ids': context_batch['input_ids'],
                'context_attention_mask': context_batch['attention_mask'],
                'input_ids': text_batch['input_ids'],
                'attention_mask': text_batch['attention_mask'],
                'labels': labels_batch['input_ids'],
            }

        # --------
        # 4) 「<|im_start|>assistant ~ <|im_end|>」以外を -100 に置き換える
        #    （メインテキスト側: batch["labels"] に対して実行）
        # --------
        # labels は [batch_size, seq_len] でパディング後のトークン ID が入っている
        # 上で作った labels_batch['input_ids'] を batch["labels"] として登録済み
        labels = batch["labels"]
        input_ids = batch["input_ids"]  # テキスト用 input_ids

        batch_size = labels.size(0)
        seq_len = labels.size(1)

        for idx in range(batch_size):
            token_ids = input_ids[idx]
            in_assistant = False

            i = 0
            while i < seq_len:
                tid = token_ids[i]

                # <|im_start|> に遭遇し、その次が assistant トークンなら開始
                if tid == self.start_token_id:
                    if (i + 1 < seq_len) and (token_ids[i+1] == self.assistant_token_id):
                        in_assistant = True
                        # <|im_start|>, assistant は学習対象外
                        labels[idx, i]   = -100
                        labels[idx, i+1] = -100

                        # もし直後の改行も無視するなら
                        if (i + 2 < seq_len) and (token_ids[i+2] == self.newline_token_id):
                            labels[idx, i+2] = -100
                        i += 2  # 2トークン分飛ばす
                        continue
                    else:
                        # assistant トークンでなければ学習対象外
                        labels[idx, i] = -100
                        in_assistant = False

                elif tid == self.end_token_id:
                    # アシスタント区間終了
                    in_assistant = False
                    # 終了トークン自体も学習対象外
                    # labels[idx, i] = -100
                    # 終了トークンは学習対象に

                else:
                    # アシスタント区間外は学習対象外に
                    if not in_assistant:
                        labels[idx, i] = -100

                i += 1

        batch["labels"] = labels

        return batch


"""
# 最初のバッチのトークン数を出力
first_batch = train_data[:1]
for i in range(len(first_batch)):
    context_tokens_count = len(first_batch['context_input_ids'][i])
    text_tokens_count = len(first_batch['input_ids'][i])
    print(f"Context tokens count: {context_tokens_count}")
    print(f"Text tokens count: {text_tokens_count}")
"""


# Hugging Faceの進捗バーを強制的に有効化
logging.set_verbosity_info()
logging.enable_progress_bar()

# トレーニング引数の設定
args = TrainingArguments(
    num_train_epochs=1,
    per_device_train_batch_size=1,
    per_device_eval_batch_size=1,
    gradient_accumulation_steps=2 if phase==2 else 4, # Phase1: 2, Phase2: 1
    learning_rate=2e-5 if phase==2 else 1e-3, # Phase1: 1e-3, Phase2: 2e-5
    # label_smoothing_factor=0.1 if phase==2 else 0.0,
    adam_beta2=0.95,
    weight_decay=0.1,
    lr_scheduler_type="cosine",
    warmup_ratio=0.03,
    disable_tqdm=False,  # tqdmの進捗バーを有効化
    logging_steps=1,  # ロギング頻度を設定
    log_level="info",
    logging_strategy="steps",
    eval_strategy="steps",
    save_strategy="steps",
    eval_steps=50, # Phase1: 73, Phase2: 73
    save_steps=2000, # Phase1: 949, Phase2: 2506
    output_dir="output",
    report_to="wandb",
    save_total_limit=3,
    push_to_hub=False,
    seed=42,
    bf16=True,  # bf16を有効化
    bf16_full_eval=True,
    #deepspeed="ds_config_mn.json",  # DeepSpeed設定ファイルの指定
    gradient_checkpointing=True,
    optim="adamw_torch_fused",
    dataloader_pin_memory=False,
    dataloader_num_workers=2,
    #local_rank=int(os.environ.get("LOCAL_RANK", -1)),
    #ddp_timeout=7200,
    # group_by_length=True,
)

# Trainerの設定
trainer = Trainer(
    model=model,
    args=args,
    train_dataset=train_data_phase2 if phase == 2 else train_data_phase1,
    eval_dataset=val_data_phase2 if phase == 2 else val_data_phase1,
    data_collator=DataCollatorAssistantWithContext(tokenizer), # data_collator,
)

print("[INFO] Trainer initialized successfully.")
# トレーニング開始
trainer.train()

for name, param in model.connector.named_parameters():
    if param.requires_grad:
        print(f"trained param - {name}: {param.shape}")

# 学習済みモデルの保存
model.save_pretrained("c3_output_model", safe_serialization=True)