from  transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("tokenizer_production/text_tokenizer", token="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", use_fast=False)

messages = [
    # {"role": "system", "content": "You are an honest and talented Japanese assistant."},
    {"role": "user", "content": "Who are you?"},
    {"role": "assistant", "content": "I am Japanese assistant."},
    {"role": "user", "content": "What can you do?"}
]

print(tokenizer.apply_chat_template(messages, tokenize=True))
eos_num = tokenizer(tokenizer.eos_token)["input_ids"]
print(eos_num)
print(tokenizer.decode(eos_num))
print(tokenizer.decode([151665]))
# test_num = tokenizer("<|begin_of_text|>test", add_special_tokens=False)["input_ids"]
# print(tokenizer.decode(test_num))
