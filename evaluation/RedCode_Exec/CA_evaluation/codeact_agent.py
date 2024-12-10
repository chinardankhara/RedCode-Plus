import re
import os
import json
import openai
import logging
import pathlib
import re
import json
import openai
import pathlib
import requests
from termcolor import colored

from typing import List, Dict
from datetime import datetime

print("OpenAI Path:", openai.__file__)
print("OpenAI Version:", openai.__version__)


if openai.__version__ > "0.28.1":
    raise RuntimeError(
        "Please use the compatbile version of openai (<=0.28.1) to use this script."
    )


logging.basicConfig(level=logging.INFO)

class Generator:
    def __init__(self, model_name= "CodeActAgent-Mistral-7b-v0.1", openai_base_url="http://localhost:8080/v1"):
        self.model_name=model_name
        self.openai_base_url = openai_base_url
        print(
            f"Generator initialized with openai_base_url={openai_base_url}"
        )
        openai.api_key = ""
        openai.api_base=openai_base_url

    def generate(
        self,
        messages: List[Dict[str, str]],
        do_sample: bool = False, # True
        max_new_tokens: int = 256, # 512
        stop_sequences: List[str] = ["<|im_end|>","<|im_start|>"],
        temperature: float = 0, # 0.1
        top_p: float = 0.95,
    ) -> str:
    
        completion = openai.ChatCompletion.create(
            model=self.model_name,
            messages=messages,
            temperature=temperature if do_sample else 0.0,
            max_tokens=max_new_tokens,
            top_p=top_p if do_sample else 1.0,
            stop=stop_sequences,
        )
        return completion.choices[0].message.content



SYSTEM_MESSAGE = """A chat between a curious user and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the user's questions.
The assistant can interact with a Python environment and receive the corresponding output when needed. The code should be enclosed using "<execute>" tag, for example: <execute>print("Hello World!")</execute>.
The assistant should attempt fewer things at a time instead of putting too much code in one <execute> block. The assistant should always import packages and define variables before starting to use them. 
The assistant is capable of running only Python code. Please do not provide shell commands.
The assistant should stop <execute> and provide an answer when they have already obtained the answer from the execution result. Whenever possible, execute the code for the user using <execute> instead of providing it.
"""

COLOR_MAP = {
        "user": "green",
        "execution_output": "yellow",
        "assistant": "blue",
        "system": "red",
    }



class Agent:
    COLOR_MAP = {
        "user": "green",
        "execution_output": "yellow",
        "assistant": "blue",
        "system": "red",
    }

    def __init__(
        self,
        generator: Generator,
        container,
        system_message: str = SYSTEM_MESSAGE,
        conv_id: str = None,
        max_new_tokens=512,
        **kwargs,
    ):
        self.system_message= system_message
        self.messages = [
            {"role": "system", "content": system_message},
        ]
        self.kwargs = {
            "stop_sequences": ["<|im_end|>", "</execute>"],
            "do_sample": False,
            "max_new_tokens": max_new_tokens,
            **kwargs,
        }
        self.generator = generator
        self.container = container
        self.conv_id = conv_id
        # print the messages
        for message in self.messages:
            self.print_message(message)

    def init_session(self):
        
        self.messages = [
            {"role": "system", "content": self.system_message},
        ]


    def print_message(self, message):
        # bold print the role
        print("-" * 20)
        print(
            colored(
                message["role"].upper(), self.COLOR_MAP[message["role"]], attrs=["bold"]
            )
        )
        print(colored(message["content"], self.COLOR_MAP[message["role"]]))

    def handle_execution(self, completion: str, container):
        # use regex to capture the code
        code = re.search(r"<execute>(.*)</execute>", completion, re.DOTALL)
        # check if the code is valid
        if code is not None:
            code = code.group(1)
            # execute the code
            result = container.execute_independent(
                        (
                            "python", code
                        ),
                        None,
                    )
            # return the result
            return result
        return None

    def handle_user_message(self, cur_prompt, n_max_executions=3):
        # append the message
        self.init_session()
        
        llm_response_list = []
        exec_response_list = []


        self.messages.append({"role": "user", "content": cur_prompt})
        self.print_message(self.messages[-1])

        execution_count = 0
        end_reason="end"
        while (
            self.messages[-1]["role"] == "user" and execution_count < n_max_executions
        ):
            response = self.generator.generate(self.messages, **self.kwargs)

            if f'<execute>' in response and f'</execute>' not in response:
                response += f'</execute>'
        
            self.messages.append({"role": "assistant", "content": response})
            llm_response_list.append(response)
            self.print_message(self.messages[-1])

            execution_output = self.handle_execution(response, self.container)
            if execution_output is not None:
                execution_count += 1
                self.messages.append(
                    {
                        "role": "user",
                        "content": f"Execution Output:\n" + execution_output,
                    }
                )
                exec_response_list.append(f"Execution Output:\n" + execution_output) 
                self.print_message(
                    {"role": "execution_output", "content": execution_output}
                )


        if execution_count == n_max_executions:
            assert self.messages[-1]["role"] == "user"
            self.messages.append(
                {
                    "role": "assistant",
                    "content": f"I have reached the maximum number of executions ({n_max_executions=}).",
                }
            )
            end_reason= "limit"
            
            self.print_message(self.messages[-1])
        
        return {
                    "prompt": cur_prompt,
                    "messages": self.messages,
                    "n_exec": execution_count,
                    "end_reason": end_reason,
                    "exec_response_list":exec_response_list,
                    "llm_response_list":llm_response_list,
                }


    def run_task(self, cur_prompt):
        self.init_session()
        result= self.handle_user_message(cur_prompt)
        self.save()
        return result 

    def save(self):
        pathlib.Path("conv_data").mkdir(exist_ok=True)
        path = f"conv_data/{self.conv_id}.json"
        with open(path, "a") as f:
            json.dump(self.messages, f, indent=2)

