port=8885


model_name=CodeLlama-7b-Instruct-hf
# model_name=CodeActAgent-Mistral-7b-v0.1


## prerequisite: start the model ${model_name} via vllm at the http://localhost:${port} according to https://github.com/xingyaoww/code-act?tab=readme-ov-file#serve-the-model-into-openai-compatible-api


for ids in 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25
do
python evaluate.py  --openai_base_url http://localhost:${port}/v1 --ids ${ids} --model_name ${model_name} 
done 

