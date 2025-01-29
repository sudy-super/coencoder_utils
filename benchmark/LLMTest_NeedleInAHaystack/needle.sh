needlehaystack.run_test \
    --provider local \
    --evaluator openai \
    --model_name "sudy-super/C-cubed-8B-128k" \
    --evaluator_model_name "chatgpt-4o-latest" \
    --context_lengths_min 1024 \
    --context_lengths_max 131072 \
    --context_lengths_num_intervals 33 \
    --document_depth_percent_intervals 33 \
