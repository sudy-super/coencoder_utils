from datasets import load_dataset, Dataset, DatasetDict
from sklearn.model_selection import train_test_split
import pandas as pd

# OASST2のオリジナルデータをロード
ds = load_dataset("OpenAssistant/oasst2")
train = ds["train"].to_pandas()
val = ds["validation"].to_pandas()
df_origin = pd.concat([train, val], axis=0).reset_index(drop=True)

# 英語データのみを抽出
df_origin_en = df_origin[df_origin["lang"] == "en"].copy()

# 日本語翻訳データをロード（'train'スプリットのみ）
ds_ja = load_dataset("kunishou/oasst2-135k-ja")
train_ja = ds_ja["train"].to_pandas()
df_ja_translations = train_ja.reset_index(drop=True)

# オリジナルデータと日本語翻訳データを結合
df = pd.merge(df_origin, df_ja_translations[["message_id", "text_ja"]], on="message_id", how="left")

# 日本語のデータフレームを作成
df_ja = df.copy()
df_ja["text"] = df_ja["text_ja"]

# 日本語データのみを抽出（text_jaが存在するもの）
df_ja = df_ja[~df_ja["text_ja"].isnull()].copy()

# 英語のアシスタントとプロンプターのメッセージを抽出
df_assistant_en = df_origin_en[df_origin_en["role"] == "assistant"].copy()

# df_prompter_enの重複を削除し、インデックスを設定
df_prompter_en = df_origin_en[df_origin_en["role"] == "prompter"].copy()
df_prompter_en = df_prompter_en.drop_duplicates(subset="message_id")
df_prompter_en = df_prompter_en.set_index("message_id")

# 日本語のアシスタントとプロンプターのメッセージを抽出
df_assistant_ja = df_ja[df_ja["role"] == "assistant"].copy()

# df_prompter_jaの重複を削除し、インデックスを設定
df_prompter_ja = df_ja[df_ja["role"] == "prompter"].copy()
df_prompter_ja = df_prompter_ja.drop_duplicates(subset="message_id")
df_prompter_ja = df_prompter_ja.set_index("message_id")

# データ準備用の関数（'message_id'を'id'にリネーム）
def prepare_data(df_assistant, df_prompter):
    # parent_idがdf_prompterのインデックスに存在する行のみを残す
    df_assistant = df_assistant[df_assistant["parent_id"].isin(df_prompter.index)].copy()
    df_assistant["output"] = df_assistant["text"].values

    # インデックスがユニークであることを確認
    assert df_prompter.index.is_unique, "df_prompterのインデックスがユニークではありません。"

    # instructionを取得
    df_assistant["instruction"] = df_prompter.loc[df_assistant["parent_id"], "text"].values

    # 'message_id' を 'id' にリネーム
    df_assistant = df_assistant.rename(columns={"message_id": "id"})
    df_assistant = df_assistant[
        ["instruction", "output", "id", "parent_id", "lang", "rank"]
    ]
    return df_assistant

# 英語と日本語のデータを準備
df_assistant_en = prepare_data(df_assistant_en, df_prompter_en)
df_assistant_ja = prepare_data(df_assistant_ja, df_prompter_ja)

# データ数の確認
print("英語アシスタントデータの総数（parent_idが存在する）：", len(df_assistant_en))
print("日本語アシスタントデータの総数（parent_idが存在する）：", len(df_assistant_ja))

# 指定された件数をランダムに抽出（データ数を超えないように調整）
n_en_samples = min(37080, len(df_assistant_en))
n_ja_samples = min(4120, len(df_assistant_ja))

df_en_sampled = df_assistant_en.sample(n=n_en_samples, random_state=42)
df_ja_sampled = df_assistant_ja.sample(n=n_ja_samples, random_state=42)

# 英語データの分割
test_size_en = 1080 if n_en_samples >= 37080 else int(n_en_samples * (1080 / 37080))
df_en_train, df_en_temp = train_test_split(df_en_sampled, test_size=test_size_en, random_state=42)
df_en_val, df_en_test = train_test_split(df_en_temp, test_size=270/1080, random_state=42)

# 日本語データの分割
test_size_ja = 120 if n_ja_samples >= 4120 else int(n_ja_samples * (120 / 4120))
df_ja_train, df_ja_temp = train_test_split(df_ja_sampled, test_size=test_size_ja, random_state=42)
df_ja_val, df_ja_test = train_test_split(df_ja_temp, test_size=30/120, random_state=42)

# データリスト作成用の関数（指定されたデータ構造に対応）
def create_data_list(df):
    return [
        {
            "conversations": [
                {"from": "user", "value": str(row["instruction"])},
                {"from": "assistant", "value": str(row["output"])}
            ]
        }
        for _, row in df.iterrows()
    ]

# 英語データセットの作成
en_train_data = create_data_list(df_en_train)
en_val_data = create_data_list(df_en_val)
en_test_data = create_data_list(df_en_test)

ds_en = DatasetDict({
    "train": Dataset.from_list(en_train_data),
    "validation": Dataset.from_list(en_val_data),
    "test": Dataset.from_list(en_test_data),
})

# 日本語データセットの作成
ja_train_data = create_data_list(df_ja_train)
ja_val_data = create_data_list(df_ja_val)
ja_test_data = create_data_list(df_ja_test)

ds_ja = DatasetDict({
    "train": Dataset.from_list(ja_train_data),
    "validation": Dataset.from_list(ja_val_data),
    "test": Dataset.from_list(ja_test_data),
})

# データセットをローカルに保存（オプション）
# ds_en.save_to_disk("oasst2_en")
# ds_ja.save_to_disk("oasst2_ja")

# Hugging Faceにアップロード
# 事前にhuggingface-cli login または from huggingface_hub import login でログイン
ds_en.push_to_hub("sudy-super/coencoder_oasst2_en", private=False)
ds_ja.push_to_hub("sudy-super/coencoder_oasst2_ja", private=False)
