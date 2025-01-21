from c_cubed_src.modeling_c_cubed import CcubedForConditionalGeneration
from c_cubed_src.configuration_c_cubed import CcubedConfig
from c_cubed_src.tokenization_c_cubed import CcubedDualTokenizer
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
from huggingface_hub import HfApi, HfFolder
import os

CcubedConfig.register_for_auto_class("AutoConfig")
CcubedForConditionalGeneration.register_for_auto_class("AutoModelForCausalLM")
CcubedDualTokenizer.register_for_auto_class("AutoTokenizer")

tokenizer = CcubedDualTokenizer.from_pretrained("./tokenizer_production", trust_remote_code=True)

model = CcubedForConditionalGeneration.from_pretrained("./output/checkpoint-1209-consolidated", torch_dtype=torch.bfloat16, trust_remote_code=True)

"""
# カスタムトークナイザを作成
tokenizer = CcubedDualTokenizer(
    context_tokenizer=context_tokenizer,
    text_tokenizer=text_tokenizer
)
"""

# 2. モデルとトークナイザを保存
model_dir = "c_cubed_phase1"
os.makedirs(model_dir, exist_ok=True)

# モデルを保存
model.save_pretrained(model_dir)

# トークナイザを保存
tokenizer.save_pretrained(model_dir)

# カスタムトークナイゼーションファイルをモデルディレクトリにコピー
# 'tokenization_Ccubed.py'を'model_dir'に保存してください

# 3. Hugging Face Hubにプッシュ

# Hugging Face Hubにログイン
api = HfApi()
token=HfFolder.get_token()

if token is None:
    from huggingface_hub import notebook_login
    notebook_login()
    token = HfFolder.get_token()

# レポジトリ名とユーザー名を設定
repo_name = "c_cubed_phase1"
username = api.whoami(token)['name']
full_repo_name = f"{username}/{repo_name}"

# レポジトリを作成（存在しない場合）
try:
    api.create_repo(repo_id=repo_name)
except Exception as e:
    print(f"Repo {full_repo_name} already exists.")

# モデルディレクトリをアップロード
api.upload_folder(
    folder_path=model_dir,
    repo_id=full_repo_name,
    commit_message="Upload model and tokenizers",
)