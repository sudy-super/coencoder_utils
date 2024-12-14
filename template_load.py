from  transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2-0.5B", token="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")

messages = [
    # {"role": "system", "content": "You are an honest and talented Japanese assistant."},
    {"role": "user", "content": "Who are you?"},
    {"role": "assistant", "content": "I am Japanese assistant."},
    {"role": "user", "content": "What can you do?"}
]

print(tokenizer.apply_chat_template(messages, tokenize=False))
eos_num = tokenizer(tokenizer.eos_token)["input_ids"]
print(tokenizer.decode(eos_num))
# test_num = tokenizer("<|begin_of_text|>test", add_special_tokens=False)["input_ids"]
# print(tokenizer.decode(test_num))
