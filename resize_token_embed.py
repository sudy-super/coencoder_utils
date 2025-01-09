from transformers import LlamaTokenizer, LlamaForCausalLM
import torch

# 1. トークナイザーをロード
tokenizer = LlamaTokenizer.from_pretrained("sbintuitions/tiny-lm-chat", use_fast=False)

# 2. 特殊トークンを定義
special_tokens = {"additional_special_tokens": ["<|start_of_context|>", "<|end_of_context|>"]}

# 3. 特殊トークンをトークナイザーに追加
tokenizer.add_special_tokens(special_tokens)

# 4. モデルをロード
model = LlamaForCausalLM.from_pretrained("sbintuitions/tiny-lm-chat", torch_dtype=torch.bfloat16)

# 5. モデルにトークナイザーの新しいボキャブラリーサイズを適用
model.resize_token_embeddings(len(tokenizer))

# 6. 確認 (特殊トークンのインデックスを表示)
special1_id = tokenizer.convert_tokens_to_ids("<|start_of_context|>")
special2_id = tokenizer.convert_tokens_to_ids("<|end_of_context|>")

print(f"Token ID for <|start_of_context|>: {special1_id}")
print(f"Token ID for <|end_of_context|>: {special2_id}")

# 7. 保存 (必要に応じて)
tokenizer.save_pretrained("./co_model_debug/text_tokenizer")
model.save_pretrained("../tiny-lm-chat-resized")
