import os
from huggingface_hub import HfApi, HfFolder

def upload_folder_to_huggingface(model_id, folder_path, token=None, private=False):
    """
    指定したフォルダ内のファイルをHugging Faceのモデルリポジトリにアップロードします。
    リポジトリが存在しない場合は作成します。

    :param model_id: Hugging FaceのモデルID（例: "username/model_name"）
    :param folder_path: アップロードするローカルフォルダのパス
    :param token: Hugging Faceのアクセストークン（指定しない場合は環境変数から取得）
    :param private: リポジトリをプライベートにするかどうか（デフォルトはFalse）
    """
    api = HfApi()
    
    if token is None:
        token = HfFolder.get_token()
        if token is None:
            raise ValueError("Hugging Faceのアクセストークンが提供されていません。トークンを指定するか、環境変数に設定してください。")
    
    # リポジトリの存在確認と作成
    try:
        repo_info = api.repo_info(repo_id=model_id, token=token)
        print(f"リポジトリ '{model_id}' は既に存在します。")
    except Exception as e:
        if "404" in str(e):
            print(f"リポジトリ '{model_id}' が存在しないため、新規作成します。")
            api.create_repo(
                name=model_id.split('/')[-1],
                token=token,
                private=private,
                repo_type="model",
                exist_ok=True  # 既に存在する場合はエラーを出さない
            )
        else:
            raise e

    # フォルダ内の全ファイルを再帰的に取得
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            local_file_path = os.path.join(root, file)
            # フォルダパスからの相対パスを取得
            relative_path = os.path.relpath(local_file_path, folder_path)
            # ファイルをアップロード
            print(f"アップロード中: {relative_path} を {model_id} に")
            try:
                api.upload_file(
                    path_or_fileobj=local_file_path,
                    path_in_repo=relative_path,
                    repo_id=model_id,
                    repo_type="model",
                    token=token
                )
            except Exception as upload_error:
                print(f"ファイル '{relative_path}' のアップロード中にエラーが発生しました: {upload_error}")


if __name__ == "__main__":
    # 例として以下を設定
    model_id = "sudy-super/c_cubed_2000"
    folder_path = "/workspace/coencoder_test/coencoder_utils/output/checkpoint-2000"
    token = "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"
    private = False

    upload_folder_to_huggingface(model_id, folder_path, token)