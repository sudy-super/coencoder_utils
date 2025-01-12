from c_cubed_src.building_c_cubed import CcubedModelBuilder, CcubedTokenizerBuilder

import torch

builder = CcubedModelBuilder(
            context_model_name="Qwen/Qwen2-0.5B",
            text_model_name="meta-llama/Llama-3.1-8B-Instruct",
            output_path="./co_model",
            auth_token="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"
)

"""
tokenizer_builder = CcubedTokenizerBuilder(
            context_model_name="Qwen/Qwen2-0.5B",
            text_model_name="meta-llama/Llama-3.1-8B-Instruct",
            output_path="./co_model",
            auth_token="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"
)
"""

builder.build_and_save_model(
    start_of_context_token_id=128002,
    end_of_context_token_id=128003
)
"""
tokenizer_builder.build_and_save_tokenizer()
"""