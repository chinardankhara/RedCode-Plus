
cd evaluation

CUDA_VISIBLE_DEVICES=3 python -m RedCode_Exec.main CA --model CodeActAgent-Mistral-7b-v0.1 --max_exec 5 --max_token 512

