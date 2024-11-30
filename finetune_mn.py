import random
from torch.utils.data import Dataset, Subset
from transformers import Trainer, TrainingArguments, logging
import torch
from datasets import load_dataset
import wandb

# CoEncoderモデルとトークナイザーのインポート
from transformers import AutoTokenizer, AutoModelForCausalLM
from coencoder_src.modeling_co_encoder import CoEncoderForConditionalGeneration
from coencoder_src.tokenization_co_encoder import CoEncoderDualTokenizer

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

# DeepSpeedがtorch.distributedの初期化を行うため、その後でランクを取得します
dist.init_process_group(backend='nccl')  # 必要に応じてバックエンドを指定

# グローバルランク0のプロセスのみでWandBを初期化
if dist.get_rank() == 0:
    wandb.init(project="coencoder_finetune_mn", name="1e-3_coencoder_connector", entity="sudy_super")

torch.manual_seed(42)

model_name = "sudy-super/coencoder_test2"

# CoEncoderトークナイザーとモデルの読み込み
tokenizer = CoEncoderDualTokenizer.from_pretrained("co_model", trust_remote_code=True)
model = CoEncoderForConditionalGeneration.from_pretrained(
    model_name,
    torch_dtype=torch.bfloat16,
    trust_remote_code=True,
    attn_implementation="flash_attention_2"
)

model.model_parallel = True

tokenizer.text_tokenizer.pad_token = tokenizer.text_tokenizer.eos_token

model.gradient_checkpointing_enable()


# context_towerとlanguage_modelの重みを凍結
for param in model.context_tower.parameters():
    param.requires_grad = False

for param in model.language_model.parameters():
    param.requires_grad = False


# データセットの読み込み
dataset = load_dataset("sudy-super/coencoder_data")

# データセットの取得
train_data = dataset["train"]
val_data = dataset["validation"]
test_data = dataset["test"]

# データセットの件数をカウントして表示
print(f"Number of train samples: {len(train_data)}")
print(f"Number of validation samples: {len(val_data)}")
print(f"Number of test samples: {len(test_data)}")


# `generate_inputs`関数をバッチ処理に対応
def generate_inputs(batch):
    contexts = []
    texts = []
    for context, conversations in zip(batch.get("context", []), batch["conversations"]):
        if not context:
            context = ""  # contextがNoneまたは空の場合、空文字列に設定
        text = """<|begin_of_text|><|start_header_id|>system<|end_header_id|>

Cutting Knowledge Date: December 2023
Today Date: 9 Nov 2024

<|eot_id|>"""
        for c in conversations:
            if c["from"] == "user":
                text += f"""<|start_header_id|>user<|end_header_id|>

{c['value']}<|eot_id|>"""
            elif c["from"] == "assistant":
                text += f"""<|start_header_id|>assistant<|end_header_id|>

{c['value']}<|eot_id|>"""
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

    # contextをカットしたリストを用いて最終的にトークン化
    tokenized_outputs = tokenizer(
        context=truncated_contexts,
        text=batch['text'],
        truncation=True,
        max_length=131072,
        padding=False,
    )

    tokenized_outputs['length'] = [len(ids) for ids in tokenized_outputs['input_ids']]

    return tokenized_outputs

def data_collator(features):
    # context部分のトークンをパディング
    context_features = [{
        'input_ids': f['context_input_ids'],
        'attention_mask': f.get('context_attention_mask', [1] * len(f['context_input_ids']))
    } for f in features]
    context_batch = tokenizer.context_tokenizer.pad(
        context_features,
        padding=True,
        max_length=None,
        return_tensors="pt"
    )
    # text部分のトークンをパディング
    text_features = [{
        'input_ids': f['input_ids'],
        'attention_mask': f['attention_mask']
    } for f in features]
    text_batch = tokenizer.text_tokenizer.pad(
        text_features,
        padding=True,
        max_length=None,
        return_tensors="pt"
    )
    # ラベルのパディング（input_idsと同じ）
    label_features = [{'input_ids': f['input_ids']} for f in features]
    labels_batch = tokenizer.text_tokenizer.pad(
        label_features,
        padding=True,
        max_length=None,
        return_tensors="pt"
    )
    # パディングされたバッチを統合
    batch = {
        'context_input_ids': context_batch['input_ids'],
        'context_attention_mask': context_batch['attention_mask'],
        'input_ids': text_batch['input_ids'],
        'attention_mask': text_batch['attention_mask'],
        'labels': labels_batch['input_ids']
    }
    return batch

# データのシャッフルとフィルタリング、バッチ処理対応
train_data = train_data.shuffle(seed=42)
val_data = val_data.shuffle(seed=42)
test_data = test_data.shuffle(seed=42)

# データの前処理（キャッシュファイル名を削除）
train_data = train_data.map(
    generate_inputs,
    batched=True,
    num_proc=8,
    desc="Generating inputs for train data",
    load_from_cache_file=True
).filter(lambda x: x['text'] != '', num_proc=4).filter(lambda x: x['context'] != '', num_proc=4)

val_data = val_data.map(
    generate_inputs,
    batched=True,
    num_proc=8,
    desc="Generating inputs for validation data",
    load_from_cache_file=True
).filter(lambda x: x['text'] != '', num_proc=4).filter(lambda x: x['context'] != '', num_proc=4)

test_data = test_data.map(
    generate_inputs,
    batched=True,
    num_proc=8,
    desc="Generating inputs for test data",
    load_from_cache_file=True
).filter(lambda x: x['text'] != '', num_proc=4).filter(lambda x: x['context'] != '', num_proc=4)

# データのトークン化（キャッシュファイル名を削除）
train_data = train_data.map(
    tokenize,
    batched=True,
    num_proc=8,
    remove_columns=train_data.column_names,
    desc="Tokenizing train data",
    load_from_cache_file=True
)
val_data = val_data.map(
    tokenize,
    batched=True,
    num_proc=8,
    remove_columns=val_data.column_names,
    desc="Tokenizing validation data",
    load_from_cache_file=True
)
test_data = test_data.map(
    tokenize,
    batched=True,
    num_proc=8,
    remove_columns=test_data.column_names,
    desc="Tokenizing test data",
    load_from_cache_file=True
)

from datasets import concatenate_datasets, Dataset

def move_random_samples(eval_dataset, train_dataset, num_samples=4500):
    # 評価データセットのインデックスを取得
    eval_indices = list(range(len(eval_dataset)))
    # ランダムにインデックスをサンプリング
    random.seed(42)
    selected_indices = random.sample(eval_indices, num_samples)

    # サブセットを作成
    selected_subset = eval_dataset.select(selected_indices)
    remaining_eval_subset = eval_dataset.select([i for i in eval_indices if i not in selected_indices])

    # SubsetをDatasetオブジェクトに変換
    selected_subset = Dataset.from_dict(selected_subset.to_dict())

    # concatenate_datasetsでサブセットを結合
    train_dataset = concatenate_datasets([train_dataset, selected_subset])

    return train_dataset, remaining_eval_subset

# 評価データセットから4000件をトレーニングデータセットに移す
train_data, eval_data = move_random_samples(val_data, train_data, num_samples=4000)

num_train_samples = int(0.6 * len(train_data))
train_data_used = train_data.select(range(num_train_samples))
train_data_unused = train_data.select(range(num_train_samples, len(train_data)))

num_eval_samples = int(0.6 * len(eval_data))
eval_data_used = eval_data.select(range(num_eval_samples))
eval_data_unused = eval_data.select(range(num_eval_samples, len(eval_data)))


# サンプルの長さに基づいてデータをソートし、バッチを形成するためのカスタムサンプラー
from torch.utils.data import Sampler
import numpy as np

class GroupedLengthSampler(Sampler):
    def __init__(self, lengths, batch_size, shuffle=True):
        self.lengths = lengths
        self.batch_size = batch_size
        self.shuffle = shuffle

        # インデックスと長さを取得
        self.indices_lengths = list(enumerate(self.lengths))

        # 長さに基づいてソート
        self.indices_lengths.sort(key=lambda x: x[1])

        # バッチを形成
        self.batches = [self.indices_lengths[i:i + self.batch_size] for i in range(0, len(self.indices_lengths), self.batch_size)]

        if self.shuffle:
            random.seed(42)
            random.shuffle(self.batches)

        # フラットなインデックスリストを作成
        self.indices = [idx for batch in self.batches for idx, _ in batch]

    def __iter__(self):
        return iter(self.indices)

    def __len__(self):
        return len(self.indices)



"""
# 最初のバッチのトークン数を出力
first_batch = train_data[:1]
for i in range(len(first_batch)):
    context_tokens_count = len(first_batch['context_input_ids'][i])
    text_tokens_count = len(first_batch['input_ids'][i])
    print(f"Context tokens count: {context_tokens_count}")
    print(f"Text tokens count: {text_tokens_count}")
"""

from torch.utils.data import DataLoader
from queue import Queue

class NetworkMonitor:
    def __init__(self, rank, world_size):
        self.rank = rank
        self.world_size = world_size
        self.running = True
        self.previous_bytes = self._get_network_stats()
        self.metrics_queue = Queue()
        
    def _get_network_stats(self):
        net_io = psutil.net_io_counters()
        return {
            'bytes_sent': net_io.bytes_sent,
            'bytes_recv': net_io.bytes_recv,
            'timestamp': time.time()
        }

    def calculate_bandwidth(self, current_bytes, previous_bytes):
        time_diff = current_bytes['timestamp'] - previous_bytes['timestamp']
        if time_diff == 0:
            return 0, 0
            
        sent_bandwidth = (current_bytes['bytes_sent'] - previous_bytes['bytes_sent']) / time_diff
        recv_bandwidth = (current_bytes['bytes_recv'] - previous_bytes['bytes_recv']) / time_diff
        return sent_bandwidth, recv_bandwidth

    def monitor(self):
        while self.running:
            current_bytes = self._get_network_stats()
            sent_bandwidth, recv_bandwidth = self.calculate_bandwidth(
                current_bytes, self.previous_bytes
            )
            
            metrics = {
                'rank': self.rank,
                'send_bandwidth_mbps': sent_bandwidth / (1024 * 1024),
                'recv_bandwidth_mbps': recv_bandwidth / (1024 * 1024),
                'total_sent_gb': current_bytes['bytes_sent'] / (1024 * 1024 * 1024),
                'total_recv_gb': current_bytes['bytes_recv'] / (1024 * 1024 * 1024),
                'timestamp': current_bytes['timestamp']
            }
            
            self.metrics_queue.put(metrics)
            self.previous_bytes = current_bytes
            time.sleep(1.0)  # 1秒間隔での測定

    def stop(self):
        self.running = False

class CustomTrainer(Trainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.network_monitor = NetworkMonitor(dist.get_rank(), dist.get_world_size())
        self.monitor_thread = None
        self.last_log_time = time.time()

    def log_network_metrics(self):
        if dist.get_rank() == 0:
            current_time = time.time()
            if current_time - self.last_log_time >= 1.0:  # 1秒ごとにログ
                try:
                    metrics = self.network_monitor.metrics_queue.get_nowait()
                    wandb.log({
                        f'network/node_{metrics["rank"]}/send_bandwidth_mbps': metrics['send_bandwidth_mbps'],
                        f'network/node_{metrics["rank"]}/recv_bandwidth_mbps': metrics['recv_bandwidth_mbps'],
                        f'network/node_{metrics["rank"]}/total_sent_gb': metrics['total_sent_gb'],
                        f'network/node_{metrics["rank"]}/total_recv_gb': metrics['total_recv_gb']
                    })
                    self.last_log_time = current_time
                except Queue.Empty:
                    pass

    def training_step(self, model, inputs, optimizer=None):
        try:
            # 通常のトレーニングステップを実行
            loss = super().training_step(model, inputs, optimizer)
            # メトリクスのログ記録
            self.log_network_metrics()
            return loss
        except Exception as e:
            # エラーが発生した場合、データの長さを出力
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
            # 例外を再度発生させる
            raise e

    def train(self, *args, **kwargs):
        # モニタリングスレッドの開始
        self.monitor_thread = threading.Thread(target=self.network_monitor.monitor)
        self.monitor_thread.daemon = True
        self.monitor_thread.start()
        
        try:
            result = super().train(*args, **kwargs)
        finally:
            # モニタリングの停止
            self.network_monitor.stop()
            if self.monitor_thread:
                self.monitor_thread.join(timeout=5)
        
        return result
    
    def get_train_dataloader(self):
        if self.train_dataset is None:
            raise ValueError("Trainer: training requires a train_dataset.")

        lengths = self.train_dataset["length"]
        sampler = GroupedLengthSampler(
            lengths=lengths,
            batch_size=self.args.per_device_train_batch_size,
            shuffle=True
        )

        return DataLoader(
            self.train_dataset,
            batch_size=self.args.per_device_train_batch_size,
            sampler=sampler,
            collate_fn=self.data_collator,
            num_workers=self.args.dataloader_num_workers,
            pin_memory=self.args.dataloader_pin_memory,
        )

# Hugging Faceの進捗バーを強制的に有効化
logging.set_verbosity_info()
logging.enable_progress_bar()

# トレーニング引数の設定
args = TrainingArguments(
    num_train_epochs=1,
    per_device_train_batch_size=1,
    per_device_eval_batch_size=1,
    gradient_accumulation_steps=64,
    learning_rate=1e-3,
    adam_beta2=0.95,
    weight_decay=0.0,
    lr_scheduler_type="cosine",
    warmup_ratio=0.03,
    disable_tqdm=False,  # tqdmの進捗バーを有効化
    logging_steps=1,  # ロギング頻度を設定
    log_level="info",
    logging_strategy="steps",
    eval_strategy="steps",
    save_strategy="steps",
    eval_steps=139,
    save_steps=324,
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
    dataloader_pin_memory=True,
    dataloader_num_workers=4,
    local_rank=int(os.environ.get("LOCAL_RANK", -1)),
)

# Trainerの設定
trainer = CustomTrainer(
    model=model,
    args=args,
    train_dataset=train_data_used,
    eval_dataset=eval_data_used,
    data_collator=data_collator,
)

# トレーニング開始
trainer.train()

# 学習済みモデルの保存
model.save_pretrained("co_output_model")

# テストデータでの評価または予測
test_results = trainer.predict(test_dataset=test_data)
predictions = test_results.predictions

# 必要に応じて予測結果を保存
decoded_preds = tokenizer.batch_decode(predictions, skip_special_tokens=True)
with open("test_predictions.txt", "w", encoding="utf-8") as f:
    for pred in decoded_preds:
        f.write(pred + "\n")
