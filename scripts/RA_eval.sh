cd evaluation

CUDA_VISIBLE_DEVICES=3 python -m RedCode_Exec.main RA --model deepseek-coder-6.7b-instruct --temperature 0 --max_tokens 1024 --python_eval

