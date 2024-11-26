from coencoder_src.modeling_co_encoder import CoEncoderForConditionalGeneration
from coencoder_src.configuration_co_encoder import CoEncoderConfig
from coencoder_src.tokenization_co_encoder import CoEncoderDualTokenizer
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
from huggingface_hub import HfApi, HfFolder
import os

CoEncoderConfig.register_for_auto_class("AutoConfig")
CoEncoderForConditionalGeneration.register_for_auto_class("AutoModelForCausalLM")
CoEncoderDualTokenizer.register_for_auto_class("AutoTokenizer")

tokenizer = AutoTokenizer.from_pretrained("./co_model", trust_remote_code=True)

model = AutoModelForCausalLM.from_pretrained("./co_model", torch_dtype=torch.bfloat16, trust_remote_code=True)

"""
# カスタムトークナイザを作成
tokenizer = CoEncoderDualTokenizer(
    context_tokenizer=context_tokenizer,
    text_tokenizer=text_tokenizer
)
"""

# 2. モデルとトークナイザを保存
model_dir = "coencoder_test3"
os.makedirs(model_dir, exist_ok=True)

# モデルを保存
model.save_pretrained(model_dir)

# トークナイザを保存
tokenizer.save_pretrained(model_dir)

# カスタムトークナイゼーションファイルをモデルディレクトリにコピー
# 'tokenization_coencoder.py'を'model_dir'に保存してください

# 3. Hugging Face Hubにプッシュ

# Hugging Face Hubにログイン
api = HfApi()
token = HfFolder.get_token()

if token is None:
    from huggingface_hub import notebook_login
    notebook_login()
    token = HfFolder.get_token()

# レポジトリ名とユーザー名を設定
repo_name = "coencoder_test3"
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