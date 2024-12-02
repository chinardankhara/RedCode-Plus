import json
import logging
import os
import signal
import time

import docker
import torch
from openai import AzureOpenAI, OpenAI
from transformers import AutoModelForCausalLM, AutoTokenizer

from .sysprompt import (SAFETY_SYS_SUFFIX0, SAFETY_SYS_SUFFIX1,
                       SAFETY_SYS_SUFFIX2, SAFETY_SYS_SUFFIX3,
                       SAFETY_SYS_SUFFIX4)


class BaseModel:
    def __init__(self, model, temperature, top_p, seed, max_tokens, dry_run):
        self.model = model
        self.temperature = temperature
        self.top_p = top_p
        self.seed = seed
        self.max_tokens = max_tokens
        self.dry_run = dry_run
        if model=='deepseek-coder-6.7b-instruct':
            self.tokenizer = AutoTokenizer.from_pretrained("deepseek-ai/deepseek-coder-6.7b-instruct", trust_remote_code=True)
            self.deepseek = AutoModelForCausalLM.from_pretrained(
                "deepseek-ai/deepseek-coder-6.7b-instruct", 
                trust_remote_code=True, 
                torch_dtype=torch.bfloat16
            ).cuda()
        if model=='deepseek-coder-v2-lite-instruct':
            self.tokenizer = AutoTokenizer.from_pretrained("deepseek-ai/deepseek-coder-v2-lite-instruct", trust_remote_code=True)
            self.deepseek_v2 = AutoModelForCausalLM.from_pretrained(
                "deepseek-ai/deepseek-coder-v2-lite-instruct", 
                trust_remote_code=True, 
                torch_dtype=torch.bfloat16,
                device_map="auto"
            )
        if model=='CodeLlama-7b-Instruct-hf':
            self.client = OpenAI(base_url="http://localhost:7878/v1", api_key='a_demo_api_key')
        if model=='CodeLlama-13b-Instruct-hf':
            self.client = OpenAI(base_url="http://localhost:7879/v1", api_key='a_demo_api_key')
        if model=='meta-llama-3-8B-instruct':
            self.tokenizer = AutoTokenizer.from_pretrained("meta-llama/meta-llama-3-8B-instruct")
            self.llama3 = AutoModelForCausalLM.from_pretrained(
                "meta-llama/meta-llama-3-8B-instruct",
                torch_dtype=torch.bfloat16,
                device_map="auto",
            )
        if model=='meta-llama-3.1-8b-instruct':
            self.tokenizer = AutoTokenizer.from_pretrained("meta-llama/meta-llama-3.1-8b-instruct")
            self.llama3 = AutoModelForCausalLM.from_pretrained(
                "meta-llama/meta-llama-3.1-8b-instruct",
                torch_dtype=torch.bfloat16,
                device_map="auto",
            )
        if model=='meta-llama-3-70b-instruct':
            self.tokenizer = AutoTokenizer.from_pretrained("meta-llama/meta-llama-3-70b-instruct")
            self.llama3 = AutoModelForCausalLM.from_pretrained(
                "meta-llama/meta-llama-3-70b-instruct",
                torch_dtype=torch.bfloat16,
                device_map="auto",
            )
        if model=='meta-llama-3.1-70b-instruct':
            self.tokenizer = AutoTokenizer.from_pretrained("meta-llama/meta-llama-3.1-70b-instruct")
            self.llama3 = AutoModelForCausalLM.from_pretrained(
                "meta-llama/meta-llama-3.1-70b-instruct",
                torch_dtype=torch.bfloat16,
                device_map="auto",
            )
        if model=='llama-2-7b-chat-hf':
            self.tokenizer = AutoTokenizer.from_pretrained("meta-llama/llama-2-7b-chat-hf")
            self.llama2 = AutoModelForCausalLM.from_pretrained(
                "meta-llama/llama-2-7b-chat-hf",
                torch_dtype=torch.bfloat16,
                device_map="auto",
            )
        if model=='llama-2-13b-chat-hf':
            self.tokenizer = AutoTokenizer.from_pretrained("meta-llama/llama-2-13b-chat-hf")
            self.llama2 = AutoModelForCausalLM.from_pretrained(
                "meta-llama/llama-2-13b-chat-hf",
                torch_dtype=torch.bfloat16,
                device_map="auto",
            )
        if model=='llama-2-70b-chat-hf':
            self.tokenizer = AutoTokenizer.from_pretrained("meta-llama/llama-2-70b-chat-hf")
            self.llama2 = AutoModelForCausalLM.from_pretrained(
                "meta-llama/llama-2-70b-chat-hf",
                torch_dtype=torch.bfloat16,
                device_map="auto",
            )
        if model=='gemma-2-9b-it':
            self.tokenizer = AutoTokenizer.from_pretrained("google/gemma-2-9b-it")
            self.gemma2 = AutoModelForCausalLM.from_pretrained(
                "google/gemma-2-9b-it",
                torch_dtype=torch.bfloat16,
                device_map="auto",
            )
        if model=='codeqwen1.5-7b-chat':
            model_id = "qwen/codeqwen1.5-7b-chat"
            self.tokenizer = AutoTokenizer.from_pretrained(model_id)
            self.codeqwen = AutoModelForCausalLM.from_pretrained(
                model_id,
                torch_dtype="auto",
                device_map="auto"
            )
        if 'claude' in model:
            import anthropic
            api_key = os.environ['ANTHROPIC_API_KEY']
            self.client = anthropic.Anthropic(api_key=api_key)

        if 'gpt' in model:
            if model=='gpt-4o-2024-05-13':
                api_key = os.environ['OPENAI_PROJ_API_KEY']
                self.client = OpenAI(api_key=api_key)
            elif model=='gpt-35-turbo':
                deploymentName = 'gpt-35-turbo'
                endpoint = 'redmond/interactive'

                API_KEY = os.environ['OPENAI_AZURE_API_KEY']
                API_VERSION = '2024-02-01'
                BASE_URL = 'https://trapi.research.microsoft.com/' + endpoint 
                DEPLOYMENT_ID = deploymentName

                self.client = AzureOpenAI(
                    api_key=API_KEY,
                    api_version=API_VERSION,
                    azure_endpoint=BASE_URL,
                    azure_deployment=DEPLOYMENT_ID
                )
            else:
                self.client = OpenAI(base_url="https://tnrllmproxy.azurewebsites.net/v1")

    def generate_deepseek(self, messages):
        inputs = self.tokenizer.apply_chat_template(messages, add_generation_prompt=True, return_tensors="pt").to(self.deepseek.device)
        # tokenizer.eos_token_id is the id of <|EOT|> token
        outputs = self.deepseek.generate(inputs, max_new_tokens=self.max_tokens, top_p=self.top_p, num_return_sequences=1, eos_token_id=self.tokenizer.eos_token_id)
        ans = self.tokenizer.decode(outputs[0][len(inputs[0]):], skip_special_tokens=True)
        return ans
    def generate_deepseek_v2(self, messages):
        inputs = self.tokenizer.apply_chat_template(messages, add_generation_prompt=True, return_tensors="pt").to(self.deepseek_v2.device)
        # tokenizer.eos_token_id is the id of <|EOT|> token
        outputs = self.deepseek_v2.generate(inputs, max_new_tokens=self.max_tokens, do_sample=False, top_p=self.top_p, num_return_sequences=1, eos_token_id=self.tokenizer.eos_token_id)
        ans = self.tokenizer.decode(outputs[0][len(inputs[0]):], skip_special_tokens=True)
        return ans
    def generate_llama3(self, messages):
        input_ids = self.tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            return_tensors="pt"
        ).to(self.llama3.device)
        terminators = [
            self.tokenizer.eos_token_id,
            self.tokenizer.convert_tokens_to_ids("<|eot_id|>")
        ]
        outputs = self.llama3.generate(
            input_ids,
            max_new_tokens=self.max_tokens,
            eos_token_id=terminators,
            do_sample=False,
            top_p=self.top_p,
        )
        ans = self.tokenizer.decode(outputs[0][input_ids.shape[-1]:], skip_special_tokens=True)
        return ans
    def generate_llama2(self, messages):
        input_ids = self.tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            return_tensors="pt"
        ).to(self.llama2.device)
        outputs = self.llama2.generate(
            input_ids,
            max_new_tokens=self.max_tokens,
            do_sample=False,
            top_p=self.top_p,
        )
        ans = self.tokenizer.decode(outputs[0][input_ids.shape[-1]:], skip_special_tokens=True)
        return ans
    def generate_gemma2(self, messages):
        prompt = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        input_ids = self.tokenizer.encode(prompt, add_special_tokens=False, return_tensors="pt").to(self.gemma2.device)
        outputs = self.gemma2.generate(
            input_ids, 
            max_new_tokens=self.max_tokens, 
            do_sample=False, 
            top_p=self.top_p,
        )
        ans = self.tokenizer.decode(outputs[0][input_ids.shape[-1]:], skip_special_tokens=True)
        return ans
    def generate_codeqwen(self, messages):
        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        model_inputs = self.tokenizer([text], return_tensors="pt").to(self.codeqwen.device)
        generated_ids = self.codeqwen.generate(
            model_inputs.input_ids,
            max_new_tokens=self.max_tokens,
            do_sample=False,
            top_p=self.top_p,
        )
        generated_ids = [
            output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
        ]
        response = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
        return response
    def generate_claude(self, messages, max_retries=10, backoff_factor=1):
        attempt = 0
        system_prompt = ""
        new_messages = []
        for message in messages:
            if message['role']=='system':
                system_prompt = message['content']
            else:
                new_messages.append(message)

        if not system_prompt:
            while attempt < max_retries:
                try:
                    response = self.client.messages.create(
                        model=self.model,
                        temperature=self.temperature,
                        top_p=self.top_p,
                        max_tokens=self.max_tokens,
                        messages=new_messages
                    )
                    return response.content[0].text
                except Exception as e:
                    attempt += 1
                    wait_time = backoff_factor * (2 ** attempt)
                    print(f"Error: {e}. Retrying in {wait_time} seconds...")
                    time.sleep(wait_time)
        else:
            while attempt < max_retries:
                try:
                    response = self.client.messages.create(
                        model=self.model,
                        temperature=self.temperature,
                        top_p=self.top_p,
                        system=system_prompt,
                        max_tokens=self.max_tokens,
                        messages=new_messages
                    )
                    return response.content[0].text
                except Exception as e:
                    attempt += 1
                    wait_time = backoff_factor * (2 ** attempt)
                    print(f"Error: {e}. Retrying in {wait_time} seconds...")
                    time.sleep(wait_time)

        raise Exception(f"Failed to get a response from the Claude API after {max_retries} attempts.")
        # except Exception as e:


class EvalTask:
    def __init__(self, task, version, safety_sys_choice):
        self.task = task
        self.version = version

        # Options: ['none', '0_safety_sys', '1_safety_sys_awareness', '2_safety_sys_detection', '3_safety_sys_modification']
        if safety_sys_choice=='none':
            self.safety_sys = ""
        elif safety_sys_choice=='0_safety_sys':
            self.safety_sys = SAFETY_SYS_SUFFIX0
        elif safety_sys_choice=='1_safety_sys_awareness':
            self.safety_sys = SAFETY_SYS_SUFFIX1
        elif safety_sys_choice=='2_safety_sys_detection':
            self.safety_sys = SAFETY_SYS_SUFFIX2
        elif safety_sys_choice=='3_safety_sys_modification':
            self.safety_sys = SAFETY_SYS_SUFFIX3
        elif safety_sys_choice=='4_safety_integration':
            self.safety_sys = SAFETY_SYS_SUFFIX4
        else:
            raise ValueError(f"Unsupported safety system prompt: {safety_sys_choice}")

class Container:
    def __init__(self, image):
        self.image = image
        self.client = docker.from_env()
        self.container = None

    def __del__(self):
        try:
            self.container.stop()
        except:
            pass
    
    def __enter__(self):
        self.container = self.client.containers.run(
            self.image,
            labels={"created_by": "code-agent-evaluation"},
            detach=True, tty=True, stdin_open=True
        )
        output = self.container.exec_run("bash /app/init.sh")
        output_str = output.output.decode()
        # print("Container DEBUG info:", output_str)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.container.stop()
        self.container.remove()

    def execute_with_retries(self, cmd, retries=3, timeout=1*60):
        def handler(signum, frame):
            raise TimeoutError("Execution timed out")
        
        signal.signal(signal.SIGALRM, handler)
        
        for attempt in range(retries):

            # # Naive exec_run
            # output = self.container.exec_run(cmd, stdin=False, stdout=True, stderr=True, demux=False)
            # output_str = output.output.decode()
            # return output_str

            try:
                logging.debug(f"Attempt {attempt+1}: Starting exec_run in {cmd}")
                signal.alarm(timeout)
                output = self.container.exec_run(cmd, stdin=False, stdout=True, stderr=True, demux=False)
                signal.alarm(0)
                output_str = output.output.decode()
                logging.debug("Finished exec_run")

                if "EOFError" in output_str:
                    logging.error(f"EOFError detected in command output: {cmd}\nOutput:\n{output_str}")
                    return "Execution blocked due to EOFError"
                
                return output_str
            except TimeoutError:
                logging.warning(f"Attempt {attempt+1}: Command execution timed out in docker container with cmd:\n```{cmd}```")
            except Exception as e:
                logging.error(f"Attempt {attempt+1}: An error occurred in docker container: {e}")
            
            logging.info(f"Retrying in 2 seconds...")
            time.sleep(2)
        
        return "Execution timed out"

    def execute_independent(self, command, timeout=1*60, *params):
        # print("=== EXECUTING INDEPENDENT ===\n", command)
        language, command = command
        # if params:
        #     print("== Parameters ==\n", params)
        if language == "bash":
            cmd = ["bash", "-c", command]
            if params:
                cmd.append("--")
                cmd.extend(params)
        elif language == "python":
            cmd = ["python", "-c", command, *params]
        elif language == "c++":
            self.execute_independent(
                (
                    "bash",
                    f'echo "{json.dumps(command)}" > /tmp/main.cpp && '
                    f"g++ -o /tmp/a.out /tmp/main.cpp",
                ),
                None,
            )
            cmd = ["/tmp/a.out", *params]
        elif language == "c":
            self.execute_independent(
                (
                    "bash",
                    f'echo "{json.dumps(command)}" > /tmp/main.cpp && '
                    f"gcc -o /tmp/a.out /tmp/main.cpp",
                ),
                None,
            )
            cmd = ["/tmp/a.out", *params]
        else:
            raise ValueError(f"Unsupported language {language}")
        
        return self.execute_with_retries(cmd, timeout=timeout)