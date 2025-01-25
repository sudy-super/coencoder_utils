from huggingface_hub import snapshot_download
import os

os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"

# Hugging Face上のモデル(リポジトリ)ID: "ユーザ名/リポジトリ名"
repo_id = "sudy-super/c_cubed_2000_raw"

# ダウンロード先のローカルフォルダ名
local_dir = "/workspace/coencoder_test/coencoder_utils/output/checkpoint-2000"

# デフォルトブランチ(mainなど)を指定したい場合
revision = "main"

# 公開モデルの場合は use_auth_token=False (認証不要)
# 非公開モデルの場合は use_auth_token=True にして、トークンを渡す必要があります
snapshot_download(
    repo_id=repo_id,
    local_dir=local_dir,
    revision=revision,
    use_auth_token=False
)

print(f"Downloaded {repo_id} into ./{local_dir}")