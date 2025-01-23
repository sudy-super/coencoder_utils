import os
from huggingface_hub import HfApi, HfFolder

def upload_folder_to_huggingface(model_id, folder_path, token=None):
    """
    指定したフォルダ内のファイルをHugging Faceのモデルリポジトリにアップロードします。

    :param model_id: Hugging FaceのモデルID（例: "username/model_name"）
    :param folder_path: アップロードするローカルフォルダのパス
    :param token: Hugging Faceのアクセストークン（指定しない場合は環境変数から取得）
    """
    api = HfApi()
    
    if token is None:
        token = HfFolder.get_token()
        if token is None:
            raise ValueError("Hugging Faceのアクセストークンが提供されていません。トークンを指定するか、環境変数に設定してください。")
    
    # フォルダ内の全ファイルを再帰的に取得
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            local_file_path = os.path.join(root, file)
            # フォルダパスからの相対パスを取得
            relative_path = os.path.relpath(local_file_path, folder_path)
            # ファイルをアップロード
            print(f"アップロード中: {relative_path} を {model_id} に")
            api.upload_file(
                path_or_fileobj=local_file_path,
                path_in_repo=relative_path,
                repo_id=model_id,
                repo_type="model",
                token=token
            )

if __name__ == "__main__":
    # 例として以下を設定
    model_id = "sudy-super/c_cubed_2000"
    folder_path = "/workspace/coencoder_test/coencoder_utils/output/checkpoint-2000"
    token = "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"

    upload_folder_to_huggingface(model_id, folder_path, token)