needlehaystack.run_test \
    --provider local \
    --evaluator openai \
    --model_name "Qwen/Qwen2.5-7B-Instruct" \
    --evaluator_model_name "chatgpt-4o-latest" \
    --context_lengths_min 1000 \
    --context_lengths_max 131000 \
    --context_lengths_num_intervals 20 \
    --document_depth_percent_intervals 20 \
    --seconds_to_sleep_between_completions 0.5 \
