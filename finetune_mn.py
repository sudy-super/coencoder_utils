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

# import torch.distributed as dist

phase = 2

# DeepSpeedがtorch.distributedの初期化を行うため、その後でランクを取得します
# dist.init_process_group(backend='nccl')  # 必要に応じてバックエンドを指定

# グローバルランク0のプロセスのみでWandBを初期化

if int(os.environ.get("LOCAL_RANK", -1)) == 0: # dist.get_rank() == 0:
    if phase == 1:
        wandb.init(project="c_cubed_phase1", name="1e-3_c_cubed_connector", entity="sudy_super")
    elif phase == 2:
        wandb.init(project="c_cubed_phase2", name="2e-5_c_cubed_lm", entity="sudy_super")
    else:
        raise ValueError("Invalid phase value. Must be 1 or 2.")


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
    dataset = load_dataset("sudy-super/c_cubed_restoration_tokenized_40k_96256")

    # データセットの取得
    train_data_phase1 = dataset["train"]
    val_data_phase1 = dataset["validation"]

    train_data_phase1 = train_data_phase1.shuffle(seed=42)
    val_data_phase1 = val_data_phase1.shuffle(seed=42)
    val_data_phase1 = val_data_phase1.select(range(len(val_data_phase1) // 2))

    # データセットの件数をカウントして表示
    print(f"Number of train samples (phase1): {len(train_data_phase1)}")
    print(f"Number of validation samples (phase1): {len(val_data_phase1)}")
elif phase == 2:
    dataset = load_dataset("sudy-super/c_cubed_finetune_tokenized_4096")

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
    1) 全サンプルで必ず `context_input_ids`, `context_attention_mask` を持つようにする
       （無いサンプルには空リストをセット）
    2) コンテキスト・メインテキストをパディング
    3) 「<|im_start|>assistant ~ <|im_end|>」以外のトークンはラベルを -100 に
    """

    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

        # 特殊トークン
        self.start_token_id = tokenizer.text_tokenizer.convert_tokens_to_ids("<|im_start|>")
        self.end_token_id   = tokenizer.text_tokenizer.convert_tokens_to_ids("<|im_end|>")
        self.assistant_token_id = 77091
        self.newline_token_id = 198

    def __call__(self, features):
        # --------
        # 0) 全サンプルに "context_input_ids" と "context_attention_mask" を強制的に用意
        # --------
        for f in features:
            if "context_input_ids" not in f:
                f["context_input_ids"] = []
            if "context_attention_mask" not in f:
                f["context_attention_mask"] = [1] * len(f["context_input_ids"])

        # --------
        # 1) コンテキスト部分をパディング (常に残す)
        # --------
        context_features = [
            {
                'input_ids': f["context_input_ids"],
                'attention_mask': f["context_attention_mask"]
            }
            for f in features
        ]
        context_batch = self.tokenizer.context_tokenizer.pad(
            context_features,
            padding=True,
            return_tensors="pt"
        )
        # こちらで常に同じ次元の `context_input_ids`, `context_attention_mask` を得る

        # --------
        # 2) テキスト部分のパディングと labels 作成 (まだ -100 処理はしない)
        # --------
        text_features = [
            {
                'input_ids': f['input_ids'],
                'attention_mask': f['attention_mask']
            }
            for f in features
        ]
        text_batch = self.tokenizer.text_tokenizer.pad(
            text_features,
            padding=True,
            return_tensors="pt"
        )
        label_features = [{'input_ids': f['input_ids']} for f in features]
        labels_batch = self.tokenizer.text_tokenizer.pad(
            label_features,
            padding=True,
            return_tensors="pt"
        )

        # --------
        # 3) バッチをまとめる（コンテキストは常に残す）
        # --------
        batch = {
            'context_input_ids': context_batch['input_ids'],
            'context_attention_mask': context_batch['attention_mask'],
            'input_ids': text_batch['input_ids'],
            'attention_mask': text_batch['attention_mask'],
            'labels': labels_batch['input_ids'],
        }

        # --------
        # 4) 「<|im_start|>assistant ~ <|im_end|>」以外を -100 に置き換える
        # --------
        labels = batch["labels"]
        input_ids = batch["input_ids"]

        batch_size = labels.size(0)
        seq_len = labels.size(1)

        for idx in range(batch_size):
            token_ids = input_ids[idx]
            in_assistant = False
            i = 0
            while i < seq_len:
                tid = token_ids[i]
                # <|im_start|>assistant ~ <|im_end|> の区間のみ学習対象
                if tid == self.start_token_id:
                    # 次のトークンが assistant ならアシスタントパート開始
                    if (i + 1 < seq_len) and (token_ids[i+1] == self.assistant_token_id):
                        in_assistant = True
                        # <|im_start|>, assistant 自体は予測不要
                        labels[idx, i]   = -100
                        labels[idx, i+1] = -100
                        i += 2
                        # 必要なら続く改行なども -100 に
                        continue
                    else:
                        # assistant でないなら学習対象外として -100
                        labels[idx, i] = -100
                        in_assistant = False
                elif tid == self.end_token_id:
                    # アシスタント区間終了
                    in_assistant = False
                    # 終了トークンも学習対象外にするなら -100
                    # labels[idx, i] = -100
                    # ※学習させたいならコメントアウト
                else:
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
        self.step_context_lengths = []
        self.step_compressed_lengths = []
        self.pad_token_id = tokenizer.context_tokenizer.pad_token_id

    def _maybe_log_save_evaluate(self, tr_loss, model, trial, epoch, ignore_keys_for_eval, *args, **kwargs):
        """
        Log context length metrics to wandb before calling parent method.
        Only logs if we have any stored context/compressed lengths.
        """
        if len(self.step_context_lengths) > 0 and len(self.step_compressed_lengths) > 0:
            # Calculate average lengths for this logging step
            avg_context_length = sum(self.step_context_lengths) / len(self.step_context_lengths)
            avg_compressed_length = sum(self.step_compressed_lengths) / len(self.step_compressed_lengths)
            
            # Create length distribution data
            length_pairs = list(zip(self.step_context_lengths, self.step_compressed_lengths))
            compression_ratios = [
                comp / orig if orig > 0 else 0
                for orig, comp in length_pairs
            ]
            
            # Log metrics directly to wandb
            if self.args.report_to == ["wandb"] and wandb.run is not None:
                wandb.log({
                    "context_length/average": avg_context_length,
                    "context_length/compressed_average": avg_compressed_length,
                    "context_length/compression_ratio": sum(compression_ratios) / len(compression_ratios),
                    "context_length/max_original": max(self.step_context_lengths),
                    "context_length/max_compressed": max(self.step_compressed_lengths),
                    "context_length/min_original": min(self.step_context_lengths),
                    "context_length/min_compressed": min(self.step_compressed_lengths),
                    "context_length/original_dist": wandb.Histogram(self.step_context_lengths),
                    "context_length/compressed_dist": wandb.Histogram(self.step_compressed_lengths),
                    "context_length/compression_ratio_dist": wandb.Histogram(compression_ratios)
                })
            
            # Reset accumulated lengths
            self.step_context_lengths = []
            self.step_compressed_lengths = []
            
        return super()._maybe_log_save_evaluate(tr_loss, model, trial, epoch, ignore_keys_for_eval, *args, **kwargs)

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        """
        Override compute_loss to capture the context lengths while maintaining original functionality.
        If `context_input_ids` is not present, simply skip that logging.
        """
        # Handle label smoothing
        if (self.label_smoother is not None or self.compute_loss_func is not None) and "labels" in inputs:
            labels = inputs.pop("labels")
        else:
            labels = None

        # Handle loss kwargs
        if self.model_accepts_loss_kwargs:
            loss_kwargs = {}
            if num_items_in_batch is not None:
                loss_kwargs["num_items_in_batch"] = num_items_in_batch
            inputs = {**inputs, **loss_kwargs}

        # Forward pass
        """
        i_s = inputs["input_ids"].shape
        print(f"input_ids shape: {i_s}")
        try:
            c_s = inputs["context_input_ids"].shape
            print(f"context_input_ids shape: {c_s}")
        except KeyError:
            print("context_input_ids shape: N/A")
        """
        outputs = model(**inputs)

        # Log context lengths if context_input_ids exist and are not all pads
        context_input_ids = inputs.get('context_input_ids', None)
        if context_input_ids is not None and isinstance(context_input_ids, torch.Tensor):
            # ここで「すべてpadトークンのみかどうか」をチェック
            # self.pad_token_id は __init__ で設定済み
            if not torch.all(context_input_ids.eq(self.pad_token_id)):
                # 全部がpadトークンではなかった場合のみログ用リストを更新
                context_length = context_input_ids.size(1)
                
                # Check if model outputs compressed context_hidden_states
                if hasattr(outputs, 'context_hidden_states') and outputs.context_hidden_states is not None:
                    compressed_length = outputs.context_hidden_states.size(1)
                    self.step_context_lengths.append(context_length)
                    self.step_compressed_lengths.append(compressed_length)

        # Save past state if it exists
        if self.args.past_index >= 0:
            self._past = outputs[self.args.past_index]

        # Compute loss
        if labels is not None:
            unwrapped_model = self.accelerator.unwrap_model(model)
            if is_peft_model(unwrapped_model):
                model_name = unwrapped_model.base_model.model._get_name()
            else:
                model_name = unwrapped_model._get_name()
            
            if self.compute_loss_func is not None:
                loss = self.compute_loss_func(outputs, labels, num_items_in_batch=num_items_in_batch)
            elif model_name in MODEL_FOR_CAUSAL_LM_MAPPING_NAMES.values():
                loss = self.label_smoother(outputs, labels, shift_labels=True)
            else:
                loss = self.label_smoother(outputs, labels)
        else:
            if isinstance(outputs, dict) and "loss" not in outputs:
                raise ValueError(
                    "The model did not return a loss from the inputs, only the following keys: "
                    f"{','.join(outputs.keys())}. For reference, the inputs it received are {','.join(inputs.keys())}."
                )
            loss = outputs["loss"] if isinstance(outputs, dict) else outputs[0]

        # Handle multi-device token averaging
        if self.args.average_tokens_across_devices and self.model_accepts_loss_kwargs:
            loss *= self.accelerator.num_processes

        return (loss, outputs) if return_outputs else loss

    def training_step(self, model, inputs, optimizer=None):
        """
        Perform a training step, with added error handling and logging.
        """
        try:
            loss = super().training_step(model, inputs, optimizer)
            return loss
            
        except Exception as e:
            input_ids = inputs.get('input_ids', None)
            context_input_ids = inputs.get('context_input_ids', None)

            # Log info about input_ids (if present)
            if input_ids is not None:
                if isinstance(input_ids, torch.Tensor):
                    text_lengths = [input_ids.size(1)]
                else:
                    text_lengths = [len(ids) for ids in input_ids]
                print(f"Error occurred during training on batch with text lengths: {text_lengths}")

            # Log info about context_input_ids (if present)
            if context_input_ids is not None:
                if isinstance(context_input_ids, torch.Tensor):
                    context_lengths = [context_input_ids.size(1)]
                else:
                    context_lengths = [len(ids) for ids in context_input_ids]
                print(f"Error occurred during training on batch with context lengths: {context_lengths}")
            else:
                print("Error occurred during training; no context_input_ids found in inputs.")

            raise e


# Hugging Faceの進捗バーを強制的に有効化
logging.set_verbosity_info()
logging.enable_progress_bar()

# トレーニング引数の設定
args = TrainingArguments(
    num_train_epochs=1,
    per_device_train_batch_size=1,
    per_device_eval_batch_size=1,
    gradient_accumulation_steps=8 if phase==2 else 8, # Phase1: 2, Phase2: 1
    learning_rate=2e-5 if phase==2 else 1e-3, # Phase1: 1e-3, Phase2: 2e-5
    # label_smoothing_factor=0.1 if phase==2 else 0.0,
    adam_beta2=0.95,
    adam_epsilon=1e-8,
    weight_decay=0.1,
    lr_scheduler_type="cosine",
    warmup_ratio=0.03,
    disable_tqdm=False,  # tqdmの進捗バーを有効化
    logging_steps=1,  # ロギング頻度を設定
    log_level="info",
    logging_strategy="steps",
    eval_strategy="steps",
    save_strategy="steps",
    eval_steps=100, # Phase1: 73, Phase2: 73
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
    dataloader_num_workers=8,
    local_rank=int(os.environ.get("LOCAL_RANK", -1)),
    ddp_timeout=7200,
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

import time
time.sleep(10)