
cd evaluation

## Prerequisite: Before running CA agent, start the model ${model_name} via vllm at the http://localhost:${port} according to https://github.com/xingyaoww/code-act?tab=readme-ov-file#serve-the-model-into-openai-compatible-api

# Replace the port in RedCode/evaluation/RedCode_Exec/main.py and RedCode/evaluation/RedCode_Exec/CA_evaluation/CA.py accordingly

python -m RedCode_Exec.main CA --model CodeActAgent-Mistral-7b-v0.1 --max_exec 5 --max_token 512

