import random
from torch.utils.data import Dataset, Subset, DataLoader
from transformers import Trainer, TrainingArguments, logging
import torch
import torch.distributed as dist
from torch.distributed.pipelining import pipe_split, pipeline, PipelineStage, ScheduleGPipe
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

phase = 1

# Initialize distributed environment for pipeline parallel
if not dist.is_initialized():
    dist.init_process_group(backend='nccl')

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

# パイプライン並列化の準備
device_ids = list(range(torch.cuda.device_count()))
if len(device_ids) < 4:
    raise ValueError("4-GPU pipeline parallelismには4台以上のGPUが必要です")


# モデルの各ステージを手動で定義
class FirstStage(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.embed_tokens = model.context_tower.tower.model.embed_tokens
        self.layers = nn.ModuleList(model.context_tower.tower.model.layers[:12])
        
    def forward(self, context_input_ids, context_attention_mask=None):
        x = self.embed_tokens(context_input_ids)
        for layer in self.layers:
            x = layer(x)
        return x

class SecondStage(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.layers = nn.ModuleList(model.context_tower.tower.model.layers[12:])
        self.norm = model.context_tower.tower.model.norm
        self.lm_head = model.context_tower.tower.lm_head
        
    def forward(self, hidden_states):
        for layer in self.layers:
            hidden_states = layer(hidden_states)
        hidden_states = self.norm(hidden_states)
        return self.lm_head(hidden_states)

class ThirdStage(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.connector = model.connector
        
    def forward(self, hidden_states):
        return self.connector(hidden_states)

class FourthStage(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.language_model = model.language_model
        
    def forward(self, past_key_values, input_ids=None, attention_mask=None, labels=None):
        return self.language_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
            past_key_values=past_key_values
        )

# パイプラインステージの作成
def create_pipeline_stage(model, stage_index):
    device = f"cuda:{stage_index}"
    if stage_index == 0:
        stage_module = FirstStage(model)
    elif stage_index == 1:
        stage_module = SecondStage(model)
    elif stage_index == 2:
        stage_module = ThirdStage(model)
    elif stage_index == 3:
        stage_module = FourthStage(model)
    else:
        raise ValueError(f"Invalid stage index: {stage_index}")
        
    # 例の入力を作成
    if stage_index == 0:
        example_input = torch.ones(1, 512, dtype=torch.long)
        stage = PipelineStage(
            stage_module, 
            stage_index, 
            4,  # num_stages 
            device,
            input_args=(example_input,),
        )
    else:
        example_input = torch.ones(1, 512, model.config.hidden_size)
        stage = PipelineStage(
            stage_module,
            stage_index,
            4,  # num_stages
            device,
            input_args=(example_input,),
        )
    
    return stage

# パイプラインステージの初期化
stage_index = dist.get_rank()
stage = create_pipeline_stage(model, stage_index)

# スケジューラーの設定
schedule = ScheduleGPipe(
    stage=stage,
    n_microbatches=4,  # マイクロバッチの数
)

if phase == 1:
    dataset = load_dataset("sudy-super/c_cubed_restoration_tokenized_98304")
    train_data_phase1 = dataset["train"]
    val_data_phase1 = dataset["validation"]
    train_data_phase1 = train_data_phase1.shuffle(seed=42)
    val_data_phase1 = val_data_phase1.shuffle(seed=42)
    print(f"Number of train samples (phase1): {len(train_data_phase1)}")
    print(f"Number of validation samples (phase1): {len(val_data_phase1)}")
elif phase == 2:
    dataset = load_dataset("sudy-super/c_cubed_finetune_tokenized")
    train_data_phase2 = dataset["train"]
    val_data_phase2 = dataset["validation"]
    train_data_phase2 = train_data_phase2.shuffle(seed=42)
    val_data_phase2 = val_data_phase2.shuffle(seed=42)
    print(f"Number of train samples (phase2): {len(train_data_phase2)}")
    print(f"Number of validation samples (phase2): {len(val_data_phase2)}")

class DataCollatorAssistantWithContext:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        self.start_token_id = tokenizer.text_tokenizer.convert_tokens_to_ids("<|im_start|>")
        self.end_token_id = tokenizer.text_tokenizer.convert_tokens_to_ids("<|im_end|>")
        self.assistant_token_id = 77091
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

class PipelineParallelDataCollator(DataCollatorAssistantWithContext):
    def __call__(self, features):
        batch = super().__call__(features)
        # テンソルを適切なGPUに配置
        for key, value in batch.items():
            if isinstance(value, torch.Tensor):
                if 'context' in key:
                    if dist.get_rank() in [0, 1]:  # context tower stages
                        batch[key] = value.cuda(dist.get_rank())
                elif key in ['input_ids', 'attention_mask']:
                    if dist.get_rank() == 2:  # connector stage
                        batch[key] = value.cuda(2)
                else:  # labels
                    if dist.get_rank() == 3:  # language model stage
                        batch[key] = value.cuda(3)
        return batch

# マイクロバッチの例を作成
example_batch = next(iter(DataLoader(train_data_phase1 if phase == 1 else train_data_phase2, batch_size=1)))

# パイプラインの作成
pipe = pipeline(
    module=model,
    mb_args=(example_batch,),
)

# 各ランクのステージモジュールを取得とGPipeスケジュールの作成
stage_idx = dist.get_rank()
stage = pipe.build_stage(stage_idx, f"cuda:{stage_idx}")
schedule = ScheduleGPipe(
    stage=stage,
    n_microbatches=4,  # マイクロバッチの数を4に設定
)

class PipelineParallelTrainer(Trainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.args.pipeline_parallel = True
    
    def training_step(self, model, inputs, optimizer=None):
        try:
            # パイプラインスケジュールによる処理
            if dist.get_rank() == 0:
                outputs = self.schedule.step(inputs)
            else:
                outputs = self.schedule.step()
            
            loss = outputs.loss
            if self.args.gradient_accumulation_steps > 1:
                loss = loss / self.args.gradient_accumulation_steps
            
            # メモリ使用量のログ記録
            if self.args.local_rank == 0 and self.state.global_step % 100 == 0:
                for i in range(torch.cuda.device_count()):
                    mem = torch.cuda.memory_allocated(i) / 1024**3
                    print(f"GPU {i} memory usage: {mem:.2f} GB")
                    
            return loss
            
        except Exception as e:
            print(f"Pipeline parallel training step error: {str(e)}")
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
            raise e

# Hugging Faceの進捗バーを強制的に有効化
logging.set_verbosity_info()
logging.enable_progress_bar()

# トレーニング引数の設定
args = TrainingArguments(
    num_train_epochs=1,
    per_device_train_batch_size=1,
    per_device_eval_batch_size=1,
    gradient_accumulation_steps=2 if phase==2 else 4,
    learning_rate=2e-5 if phase==2 else 1e-3,
    adam_beta2=0.95,
    weight_decay=0.1,
    lr_scheduler_type="cosine",
    warmup_ratio=0.03,
    disable_tqdm=False,
    logging_steps=1,
    log_level="info",
    logging_strategy="steps",
    eval_strategy="steps",
    save_strategy="steps",
    eval_steps=50,
    save_steps=2000,
    output_dir="output",
    report_to="wandb",
    save_total_limit=3,
    push_to_hub=False,
    seed=42,
    bf16=True,
    bf16_full_eval=True,
    deepspeed="ds_config_mn.json",
    gradient_checkpointing=False,  # パイプライン並列を使用するため無効化
    pipeline_parallel_degree=4,    # パイプライン並列のステージ数
    distributed_state="pipeline_parallel",
    optim="adamw_torch_fused",
    dataloader_pin_memory=False,
    dataloader_num_workers=2,
    local_rank=int(os.environ.get("LOCAL_RANK", -1)),
)

# トレーナーの初期化
trainer = PipelineParallelTrainer(
    model=stage,  # パイプラインステージを使用
    args=args,
    train_dataset=train_data_phase2 if phase == 2 else train_data_phase1,
    eval_dataset=val_data_phase2 if phase == 2 else val_data_phase1,
    data_collator=PipelineParallelDataCollator(tokenizer),
)

print("[INFO] Trainer initialized successfully.")
trainer.train()

# 学習済みのパラメータ確認とモデルの保存は最後のステージでのみ実行
if dist.get_rank() == 3:
    for name, param in model.connector.named_parameters():
        if param.requires_grad:
            print(f"trained param - {name}: {param.shape}")
    
    model.save_pretrained("c3_output_model", safe_serialization=True)