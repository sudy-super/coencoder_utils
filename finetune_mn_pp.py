import random
from torch.utils.data import Dataset, Subset, DataLoader, Sampler
from transformers import Trainer, TrainingArguments, logging
import torch
from datasets import load_dataset, concatenate_datasets, Dataset
import wandb
from transformers import AutoTokenizer, AutoModelForCausalLM
from coencoder_src.modeling_co_encoder import CoEncoderForConditionalGeneration
from coencoder_src.tokenization_co_encoder import CoEncoderDualTokenizer
import os
import psutil
import subprocess
import re
import threading
import time
from datetime import datetime
from queue import Queue
import deepspeed
from deepspeed.pipe import PipelineModule
from deepspeed.runtime.pipe import schedule
import torch.distributed as dist

# DeepSpeedの初期化
dist.init_process_group(backend='nccl')
deepspeed.init_distributed()

local_rank = int(os.environ.get("LOCAL_RANK", -1))
world_rank = dist.get_rank()

# グローバルランク0のプロセスのみでWandBを初期化
if world_rank == 0:
    wandb.init(project="coencoder_finetune_mn", name="1e-3_coencoder_connector", entity="sudy_super")

torch.manual_seed(42)

# データセットの読み込みと前処理用の関数
def generate_inputs(batch):
    contexts = []
    texts = []
    for context, conversations in zip(batch.get("context", []), batch["conversations"]):
        if not context:
            context = ""
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

def tokenize(batch):
    max_context_tokens = 131072
    truncated_contexts = []
    for context in batch['context']:
        context_tokens = tokenizer.context_tokenizer.tokenize(context)
        if len(context_tokens) > max_context_tokens:
            context = tokenizer.context_tokenizer.convert_tokens_to_string(context_tokens[:max_context_tokens])
        truncated_contexts.append(context)
    
    text_tokenized = tokenizer.text_tokenizer(batch['text'], add_special_tokens=False)
    text_lengths = [len(ids) for ids in text_tokenized['input_ids']]

    tokenized_outputs = tokenizer(
        context=truncated_contexts,
        text=batch['text'],
        truncation=True,
        max_length=max_context_tokens,
        padding=False,
    )

    tokenized_outputs['length'] = [len(ids) for ids in tokenized_outputs['input_ids']]
    tokenized_outputs['text_length'] = text_lengths
    return tokenized_outputs

def data_collator(features):
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

    label_features = [{'input_ids': f['input_ids']} for f in features]
    labels_batch = tokenizer.text_tokenizer.pad(
        label_features,
        padding=True,
        max_length=None,
        return_tensors="pt"
    )

    batch = {
        'context_input_ids': context_batch['input_ids'],
        'context_attention_mask': context_batch['attention_mask'],
        'input_ids': text_batch['input_ids'],
        'attention_mask': text_batch['attention_mask'],
        'labels': labels_batch['input_ids']
    }
    return batch

class GroupedLengthSampler(Sampler):
    def __init__(self, lengths, batch_size, shuffle=True):
        self.lengths = lengths
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.indices_lengths = list(enumerate(self.lengths))
        self.indices_lengths.sort(key=lambda x: x[1], reverse=True)
        self.batches = [self.indices_lengths[i:i + self.batch_size] 
                       for i in range(0, len(self.indices_lengths), self.batch_size)]
        
        if self.shuffle:
            random.seed(42)
            random.shuffle(self.batches)
        
        self.indices = [idx for batch in self.batches for idx, _ in batch]

    def __iter__(self):
        return iter(self.indices)

    def __len__(self):
        return len(self.indices)

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
            time.sleep(1.0)

    def stop(self):
        self.running = False

class CoEncoderPipeline(PipelineModule):
    def __init__(self, base_model, num_stages=32):
        stages = []
        
        # Stage 0-11: Context Tower
        for i in range(12):
            start_idx = i * 2
            stages.append([
                base_model.context_tower.tower.model.layers[start_idx],
                base_model.context_tower.tower.model.layers[start_idx + 1]
            ])
        
        # Stage 12-14: Connector
        stages.append([base_model.connector.dynamic_pooling])
        stages.append([base_model.connector.linear_1])
        stages.append([
            base_model.connector.act,
            base_model.connector.linear_2,
            base_model.language_model.model.embed_tokens
        ])
        
        # Stage 15-30: Language Model Layers
        for i in range(16):
            start_idx = i * 2
            stages.append([
                base_model.language_model.model.layers[start_idx],
                base_model.language_model.model.layers[start_idx + 1]
            ])
        
        # Stage 31: Final Layer Norm and LM Head
        stages.append([
            base_model.language_model.model.norm,
            base_model.language_model.lm_head
        ])
        
        # 親クラスの初期化
        super().__init__(
            layers=stages,
            loss_fn=base_model.forward,
            num_stages=num_stages,
            partition_method='uniform',
            activation_checkpoint_interval=0
        )

        self.base_model = base_model

        def gradient_checkpointing_enable(self):
            # モジュール内の全てのレイヤーで勾配チェックポイントを有効にする
            for layer in self.modules():
                if hasattr(layer, 'gradient_checkpointing_enable'):
                    layer.gradient_checkpointing_enable()

class DeepSpeedPipelineTrainer(Trainer):
    def __init__(self, model_engine, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.model_engine = model_engine
        self.network_monitor = NetworkMonitor(dist.get_rank(), dist.get_world_size())
        self.monitor_thread = None
        self.last_log_time = time.time()

    def log_network_metrics(self):
        if dist.get_rank() == 0:
            current_time = time.time()
            if current_time - self.last_log_time >= 1.0:
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

    def training_step(self, model, inputs):
        try:
            outputs = self.model_engine(inputs)
            loss = outputs[0]
            
            self.model_engine.backward(loss)
            self.model_engine.step()
            self.log_network_metrics()
            
            return loss.detach()
            
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
            raise e

    def train(self, *args, **kwargs):
        self.monitor_thread = threading.Thread(target=self.network_monitor.monitor)
        self.monitor_thread.daemon = True
        self.monitor_thread.start()
        
        try:
            result = super().train(*args, **kwargs)
        finally:
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
            shuffle=False
        )

        return DataLoader(
            self.train_dataset,
            batch_size=self.args.per_device_train_batch_size,
            sampler=sampler,
            collate_fn=self.data_collator,
            num_workers=self.args.dataloader_num_workers,
            pin_memory=self.args.dataloader_pin_memory,
        )

# モデル名と設定
model_name = "sudy-super/coencoder_test2"

# トークナイザーの読み込み
tokenizer = CoEncoderDualTokenizer.from_pretrained("co_model", trust_remote_code=True)
tokenizer.text_tokenizer.pad_token = tokenizer.text_tokenizer.eos_token

# ベースモデルの読み込み
base_model = CoEncoderForConditionalGeneration.from_pretrained(
    model_name,
    torch_dtype=torch.bfloat16,
    trust_remote_code=True,
    attn_implementation="flash_attention_2"
)

# データセットの読み込み
dataset = load_dataset("sudy-super/coencoder_data")

# データセットの取得と前処理
train_data = dataset["train"]
val_data = dataset["validation"]
test_data = dataset["test"]

# データの前処理
def process_dataset(dataset, is_train=True):
    # 入力生成
    dataset = dataset.map(
        generate_inputs,
        batched=True,
        num_proc=8,
        desc="Generating inputs",
        load_from_cache_file=True
    ).filter(
        lambda x: x['text'] != '' and x['context'] != '', 
        num_proc=8
    )
    
    # トークン化
    return dataset.map(
        tokenize,
        batched=True,
        num_proc=8,
        remove_columns=dataset.column_names,
        desc="Tokenizing data",
        load_from_cache_file=True
    )

train_data = process_dataset(train_data, is_train=True)
val_data = process_dataset(val_data, is_train=False)
test_data = process_dataset(test_data, is_train=False)

train_data = train_data.filter(lambda x: x['text_length'] <= 4096, num_proc=8)
val_data = val_data.filter(lambda x: x['text_length'] <= 4096, num_proc=8)
test_data = test_data.filter(lambda x: x['text_length'] <= 4096, num_proc=8)


print(f"Number of train samples: {len(train_data)}")
print(f"Number of validation samples: {len(val_data)}")
print(f"Number of test samples: {len(test_data)}")

# 評価データの一部をトレーニングデータに移動
def move_random_samples(eval_dataset, train_dataset, num_samples=4500):
    eval_indices = list(range(len(eval_dataset)))
    random.seed(42)
    selected_indices = random.sample(eval_indices, num_samples)
    
    selected_subset = eval_dataset.select(selected_indices)
    remaining_eval_subset = eval_dataset.select([i for i in eval_indices if i not in selected_indices])
    selected_subset = Dataset.from_dict(selected_subset.to_dict())
    train_dataset = concatenate_datasets([train_dataset, selected_subset])
    
    return train_dataset, remaining_eval_subset

train_data, eval_data = move_random_samples(val_data, train_data, num_samples=4000)

# データセットの分割
num_train_samples = int(0.6 * len(train_data))
train_data_used = train_data.select(range(num_train_samples))
train_data_unused = train_data.select(range(num_train_samples, len(train_data)))

num_eval_samples = int(0.6 * len(eval_data))
eval_data_used = eval_data.select(range(num_eval_samples))
eval_data_unused = eval_data.select(range(num_eval_samples, len(eval_data)))

# 長さでソート
train_data_sorted = train_data_used.sort('length')

# DeepSpeed設定
ds_config = {
    "train_batch_size": 64,
    "gradient_accumulation_steps": 64,
    "gradient_clipping": 1.0,
    "bf16": {
        "enabled": True
    },
    "zero_optimization": {
        "stage": 0
    },
    "optimizer": {
        "type": "AdamW",
        "params": {
            "lr": 1e-3,
            "betas": [0.9, 0.95],
            "eps": 1e-8,
            "weight_decay": 0.0
        }
    },
    "pipeline": {
        "stages": 32,
        "pipe_chunk_size": 1,
        "activation_checkpoint_interval": 0,
        "pipe_schedule": "interleaved"
    },
    "data_parallel_size": 1,
    "activation_checkpointing": {
        "partition_activations": True,
        "contiguous_memory_optimization": True
    }
}


# パイプラインモデルの作成と初期化
model = CoEncoderPipeline(base_model, num_stages=32)

# パラメータの凍結
for name, param in model.named_parameters():
    if 'context_tower' in name or 'language_model' in name:
        param.requires_grad = True


# 学習するパラメータ（connector部分）
trainable_params = []
# 学習しないパラメータ（他の部分）
non_trainable_params = []

for name, param in model.named_parameters():
    if 'connector' in name:
        trainable_params.append(param)
    else:
        non_trainable_params.append(param)

# パラメータグループを定義
param_groups = [
    {'params': trainable_params},  # 学習するパラメータ
    {'params': non_trainable_params, 'lr': 0.0}  # 学習しないパラメータ
]

# トレーニング引数の設定
training_args = TrainingArguments(
    num_train_epochs=1,
    per_device_train_batch_size=1,
    per_device_eval_batch_size=1,
    gradient_accumulation_steps=2,
    learning_rate=1e-3,
    adam_beta2=0.95,
    weight_decay=0.0,
    lr_scheduler_type="cosine",
    warmup_ratio=0.03,
    disable_tqdm=False,
    logging_steps=1,
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
    bf16=True,
    bf16_full_eval=True,
    deepspeed="ds_config_mn_pp.json",
    local_rank=local_rank,
    gradient_checkpointing=True,
    optim="adamw_torch_fused",
    dataloader_pin_memory=True,
    dataloader_num_workers=4,
)

optimizer = torch.optim.AdamW(param_groups, lr=training_args.learning_rate)
# DeepSpeedエンジンの初期化
model_engine, optimizer, _, _ = deepspeed.initialize(
    model=model,
    optimizer=optimizer,
    # model_parameters=model.parameters(),
    config=ds_config
)

# トレーナーの初期化
trainer = DeepSpeedPipelineTrainer(
    model_engine=model_engine,
    model=model,
    args=training_args,
    train_dataset=train_data_sorted,
    eval_dataset=eval_data_used,
    data_collator=data_collator,
)

# トレーニングの実行
trainer.train()

# モデルの保存
if trainer.is_world_process_zero():
    model_engine.save_checkpoint("co_output_model")

    # Pipeline形式からオリジナルのモデル形式に戻す
    original_model = base_model
    
    # Connectorの重みを更新
    connector_state_dict = {}
    for name, param in model.named_parameters():
        if 'connector' in name:
            # Pipeline形式の名前をオリジナルの名前に変換
            original_name = name.replace('module.', '')
            connector_state_dict[original_name] = param.data
    
    # オリジナルモデルのConnector部分のみを更新
    original_model_dict = original_model.state_dict()
    original_model_dict.update(connector_state_dict)
    original_model.load_state_dict(original_model_dict)
    
    # Transformers形式で保存
    original_model.save_pretrained(
        "co_output_model_transformers",
        # save_function=torch.save,
        state_dict=original_model_dict,
        max_shard_size="5GB"
    )

# テストデータでの評価
test_results = trainer.predict(test_dataset=test_data)
predictions = test_results.predictions

# 予測結果の保存
if trainer.is_world_process_zero():
    decoded_preds = tokenizer.batch_decode(predictions, skip_special_tokens=True)
    with open("test_predictions.txt", "w", encoding="utf-8") as f:
        for pred in decoded_preds:
            f.write(pred + "\n")