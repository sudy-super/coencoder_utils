from c_cubed_src.modeling_c_cubed import CoEncoderForConditionalGeneration
from c_cubed_src.configuration_c_cubed import CoEncoderConfig
from c_cubed_src.tokenization_c_cubed import CoEncoderDualTokenizer
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

CoEncoderConfig.register_for_auto_class("AutoConfig")
CoEncoderForConditionalGeneration.register_for_auto_class("AutoModelForCausalLM")
CoEncoderDualTokenizer.register_for_auto_class("AutoTokenizer")


tokenizer = AutoTokenizer.from_pretrained("./co_model", trust_remote_code=True)

model = AutoModelForCausalLM.from_pretrained("./co_model", torch_dtype=torch.bfloat16, trust_remote_code=True)


model.push_to_hub("coencoder_test2")

tokenizer.push_to_hub("coencoder_test2")

