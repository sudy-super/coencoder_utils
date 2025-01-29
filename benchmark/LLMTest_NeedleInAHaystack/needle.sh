needlehaystack.run_test \
    --provider local \
    --evaluator openai \
    --model_name "Qwen/Qwen2.5-7B-Instruct" \
    --evaluator_model_name "chatgpt-4o-latest" \
    --context_lengths_min 1024 \
    --context_lengths_max 131072 \
    --context_lengths_num_intervals 33 \
    --document_depth_percent_intervals 33 \
    --seconds_to_sleep_between_completions 0.5 \
