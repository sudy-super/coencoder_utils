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

import torch.distributed as dist

phase = 1
"""
# DeepSpeedがtorch.distributedの初期化を行うため、その後でランクを取得します
dist.init_process_group(backend='nccl')  # 必要に応じてバックエンドを指定

# グローバルランク0のプロセスのみでWandBを初期化

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
    model_name = "sudy-super/c_cubed"
elif phase == 2:
    model_name = "sudy-super/c_cubed_phase1"
else:
    raise ValueError("Invalid phase value. Must be 1 or 2.")

# CoEncoderトークナイザーとモデルの読み込み
try:
    tokenizer = CcubedDualTokenizer.from_pretrained("./tokenizer_production", trust_remote_code=True, use_fast=False)
except:
    print("[INFO] Failed to load tokenizer with use_fast=False. Retrying with use_fast=True.")
    tokenizer = CcubedDualTokenizer.from_pretrained("./tokenizer_production", trust_remote_code=True)

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
    dataset = load_dataset("sudy-super/c_cubed_restoration_tokenized_98304")

    # データセットの取得
    train_data_phase1 = dataset["train"]
    val_data_phase1 = dataset["validation"]

    train_data_phase1 = train_data_phase1.shuffle(seed=42)
    val_data_phase1 = val_data_phase1.shuffle(seed=42)

    # データセットの件数をカウントして表示
    print(f"Number of train samples (phase1): {len(train_data_phase1)}")
    print(f"Number of validation samples (phase1): {len(val_data_phase1)}")
elif phase == 2:
    dataset = load_dataset("sudy-super/c_cubed_finetune_tokenized")

    # データセットの取得
    train_data_phase2 = dataset["train"]
    val_data_phase2 = dataset["validation"]

    train_data_phase2 = train_data_phase2.shuffle(seed=42)
    val_data_phase2 = val_data_phase2.shuffle(seed=42)

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
        self.start_token_id = tokenizer.text_tokenizer.convert_tokens_to_ids("<|im_start|>")
        self.end_token_id   = tokenizer.text_tokenizer.convert_tokens_to_ids("<|im_end|>")

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

class CustomTrainer(Trainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
    def training_step(self, model, inputs, optimizer=None):
        try:
            # 通常のトレーニングステップを実行
            loss = super().training_step(model, inputs, optimizer)
        
            input_ids = inputs.get('input_ids', None)
            context_input_ids = inputs.get('context_input_ids', None)
            if input_ids is not None:
                if isinstance(input_ids, torch.Tensor):
                    text_lengths = [input_ids.size(1)]
                else:
                    text_lengths = [len(ids) for ids in input_ids]
                print(f"Text lengths: {text_lengths}")
            if context_input_ids is not None:
                if isinstance(context_input_ids, torch.Tensor):
                    context_lengths = [context_input_ids.size(1)]
                else:
                    context_lengths = [len(ids) for ids in context_input_ids]
                print(f"Context lengths: {context_lengths}")
            else:
                print("Error occurred during training but could not retrieve input_ids or context_input_ids")
            
            return loss
        except Exception as e:
            input_ids = inputs.get('input_ids', None)
            context_input_ids = inputs.get('context_input_ids', None)
            if input_ids is not None:
                if isinstance(input_ids, torch.Tensor):
                    text_lengths = [input_ids.size(1)]
                else:
                    text_lengths = [len(ids) for ids in input_ids]
                print(f"Error occurred during training on batch with text lengths: {text_lengths}")
            if context_input_ids is not None:
                if isinstance(context_input_ids, torch.Tensor):
                    context_lengths = [context_input_ids.size(1)]
                else:
                    context_lengths = [len(ids) for ids in context_input_ids]
                print(f"Error occurred during training on batch with context lengths: {context_lengths}")
            else:
                print("Error occurred during training but could not retrieve input_ids or context_input_ids")
            # エラーが発生した場合、データの長さを出力    
            # 例外を再度発生させる
            raise e


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
    deepspeed="ds_config_mn.json",  # DeepSpeed設定ファイルの指定
    gradient_checkpointing=True,
    optim="adamw_torch_fused",
    dataloader_pin_memory=False,
    dataloader_num_workers=2,
    local_rank=int(os.environ.get("LOCAL_RANK", -1)),
    #ddp_timeout=7200,
    # group_by_length=True,
)

# Trainerの設定
trainer = CustomTrainer(
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