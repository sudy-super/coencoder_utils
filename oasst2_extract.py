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

# 日本語翻訳データをロード（スプリットを指定）
ds_ja = load_dataset("kunishou/oasst2-135k-ja")
train_ja = ds_ja["train"].to_pandas()
val_ja = ds_ja["validation"].to_pandas()
df_ja_translations = pd.concat([train_ja, val_ja], axis=0).reset_index(drop=True)

# オリジナルデータと日本語翻訳データを結合
df = pd.merge(df_origin, df_ja_translations[["message_id", "text_ja"]], on="message_id", how="left")

# 日本語のデータフレームを作成
df_ja = df.copy()
df_ja["text"] = df_ja["text_ja"]

# 日本語データのみを抽出（text_jaが存在するもの）
df_ja = df_ja[~df_ja["text_ja"].isnull()].copy()

# 英語のアシスタントとプロンプターのメッセージを抽出
df_assistant_en = df_origin_en[df_origin_en["role"] == "assistant"].copy()
df_prompter_en = df_origin_en[df_origin_en["role"] == "prompter"].copy().set_index("message_id")

# 日本語のアシスタントとプロンプターのメッセージを抽出
df_assistant_ja = df_ja[df_ja["role"] == "assistant"].copy()
df_prompter_ja = df_ja[df_ja["role"] == "prompter"].copy().set_index("message_id")

# データ準備用の関数
def prepare_data(df_assistant, df_prompter):
    df_assistant["output"] = df_assistant["text"].values
    instructions = []
    parent_ids = []
    for _, row in df_assistant.iterrows():
        parent = df_prompter.loc[row["parent_id"]]
        instructions.append(parent["text"])
        parent_ids.append(parent["parent_id"])
    df_assistant["instruction"] = instructions
    df_assistant["parent_id"] = parent_ids
    df_assistant = df_assistant[
        ["instruction", "output", "id", "parent_id", "lang", "rank"]
    ]
    return df_assistant

# 英語と日本語のデータを準備
df_assistant_en = prepare_data(df_assistant_en, df_prompter_en)
df_assistant_ja = prepare_data(df_assistant_ja, df_prompter_ja)

# 指定された件数をランダムに抽出
df_en_sampled = df_assistant_en.sample(n=37080, random_state=42)
df_ja_sampled = df_assistant_ja.sample(n=4120, random_state=42)

# 英語データの分割
df_en_train, df_en_temp = train_test_split(df_en_sampled, test_size=1080, random_state=42)
df_en_val, df_en_test = train_test_split(df_en_temp, test_size=270/1080, random_state=42)

# 日本語データの分割
df_ja_train, df_ja_temp = train_test_split(df_ja_sampled, test_size=120, random_state=42)
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
