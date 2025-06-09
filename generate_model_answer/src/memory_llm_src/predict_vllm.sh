#!/bin/bash

# gpu number for vllm inference
devices="4,5,6,7"

# "summarization", "short_answer", "true_false", "multiple_choice", "multiple_select"
task_list='["multiple_choice", "true_false"]'

# original, topic, paragraph, cot
exp_type="original"

save_path="YOUR_SAVE_PATH"
data_path="YOUR_DATA_PATH"

# 'llama-70b', 'llama-8b', 'Mistral-24b', 'Mistral-8b', 'Qwen-72b', 'Qwen-32b-reasoning', 'Qwen-7b', 'Trilion-7b', 'Gemma2-27b', 'Gemma2-9b', 'Bllossom-70b', 'Bllossom-8b', 'Koni-8b', 'Exaone-32b-reasoning', 'Exaone-32b', 'Exaone-8b'
CUDA_VISIBLE_DEVICES=$devices python3 main.py --model_nick llama-8b --task_list "$task_list" --exp_type $exp_type --save_path $save_path --data_path $data_path
