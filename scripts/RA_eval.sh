cd evaluation

# python -m RedCode_Exec.main RA --model gemma-2-9b-it --temperature 0 --max_tokens 1024 --python_eval
# python -m RedCode_Exec.main RA --model deepseek-coder-6.7b-instruct --temperature 0 --max_tokens 1024 --python_eval
python -m RedCode_Exec.main RA --model gpt-4o --temperature 0 --max_tokens 1024 --python_eval

