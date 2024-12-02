# from evaluate import *
# from interaction import *
# import argparse
# import datetime
# import os

# # Eval commands example:
# # python main.py gpt-3.5-turbo-1106 --python_eval

# if __name__=='__main__':

#     if not os.path.exists('../logs/Agents'):
#         os.makedirs('../logs/Agents')
#     if not os.path.exists('../results'):
#         os.makedirs('../results')

#     time_now = datetime.datetime.now()+datetime.timedelta(hours=8)
#     time_stamp = str(time_now).replace(' ', '_')
#     resfile = f'../results/{time_stamp}.json'
#     logfile = f'../logs/Agents/Agents_{time_stamp}.json'

#     parser = argparse.ArgumentParser(description='Select base model for CodeAgent and set parameters.')
#     parser.add_argument('model', choices=[
#                             'gpt-3.5-turbo-1106', 'gpt-4-1106-preview', 'gpt-4o-2024-05-13',
#                             'deepseek-coder-6.7b-instruct', 'deepseek-coder-v2-lite-instruct',
#                             'CodeLlama-7b-Instruct-hf', 'CodeLlama-13b-Instruct-hf',
#                             'meta-llama-3-8B-instruct', 'meta-llama-3.1-8b-instruct',
#                             'meta-llama-3-70b-instruct', 'meta-llama-3.1-70b-instruct',
#                             'gemma-2-9b-it',
#                             'claude-3-5-sonnet-20240620',
#                             'llama-2-7b-chat-hf', 'llama-2-13b-chat-hf', 'llama-2-70b-chat-hf',
#                             'codeqwen1.5-7b-chat',
#                             'gpt-35-turbo'],
#                         help='The base model to use in CodeAgent.')
#     parser.add_argument('--temperature', type=float, default=0, help='Temperature setting for the base model.')
#     parser.add_argument('--top_p', type=float, default=1.0, help='Top-p (nucleus sampling) parameter for the base model.')
#     parser.add_argument('--seed', type=int, default=998244353, help='Random seed for reproducibility.')
#     parser.add_argument('--max_tokens', type=int, default=1024, help='Token length limitation.')
#     parser.add_argument('--dry_run', type=bool, default=False)
#     parser.add_argument('--version', type=str, default='v1', help='Version of the dataset to evaluate.')

#     parser.add_argument('--safety_sys_choice', choices=[
#                             'none', '0_safety_sys', '1_safety_sys_awareness', '2_safety_sys_detection', '3_safety_sys_modification',
#                             '4_safety_integration'
#                             ],
#                         default='none',
#                         help='The safety system prompt for risk mitigation.')

#     group = parser.add_mutually_exclusive_group(required=True)
#     group.add_argument('--python_eval', action='store_true', help='Flag to enable Python evaluation.')
#     group.add_argument('--bash_eval', action='store_true', help='Flag to enable Bash evaluation.')

#     args = parser.parse_args()

#     if args.python_eval:
#         task = 'python_eval'
#         ids = [1, 2, 3, 4, 5, 6, 7, 8, 9, 12, 13, 14, 15, 18, 19, 20, 22, 23, 24, 25, 11, 16, 17, 21, 10]
#     if args.bash_eval:
#         task = 'bash_eval'
#         ids = [1, 2, 3, 4, 5, 6, 7, 8, 9, 12, 13, 14, 15, 16, 18, 20, 21, 22, 23, 25]


#     base_model  = BaseModel(model=args.model, temperature=args.temperature, top_p=args.top_p, seed=args.seed, max_tokens=args.max_tokens, dry_run=args.dry_run)
#     eval_task   = EvalTask(task=task.split('_')[0], version=args.version, safety_sys_choice=args.safety_sys_choice)

#     for id in ids:
#         image = "redcode"
#         with Container(image) as container:

#             if task=='python_eval':
#                 jsonfile = f'../Trans_code/RedCode-Exec-{args.version}/py2text_dataset_json/index{id}_30_codes_full.json'
            
#             if task=='bash_eval':
#                 jsonfile = f'../Trans_code/RedCode-Exec-{args.version}/bash2text_dataset_json/index{id}_30_codes_full_upd.json'

#             eval_dataset(id, container, base_model, jsonfile, resfile, logfile, eval_task)


from RedCode_Exec.RA_evaluation.evaluate import *
from RedCode_Exec.RA_evaluation.interaction import *
import argparse
import datetime
import os

def RA(model, temperature=0, top_p=1.0, seed=998244353, max_tokens=1024, dry_run=False, version='v1', 
       safety_sys_choice='none', task_type='python_eval'):

    # Create directories if not exist
    # if not os.path.exists('../logs/Agents'):
    #     os.makedirs('../logs/Agents')
    # if not os.path.exists('../results'):
    #     os.makedirs('../results')

    # Create directories if not exist
    if not os.path.exists('../results/RA/log'):
        os.makedirs('../results/RA/log')
    if not os.path.exists('../results/RA'):
        os.makedirs('../results/RA')

    # Timestamp for file naming
    time_now = datetime.datetime.now() + datetime.timedelta(hours=8)
    time_stamp = str(time_now).replace(' ', '_')
    # resfile = f'../results/RA/{time_stamp}.json'
    # logfile = f'../logs/Agents/Agents_{time_stamp}.json'

    resfile = f'../results/RA/{time_stamp}.json'
    logfile = f'../results/RA/log/RA_log_{time_stamp}.json'

    # Define task ids based on task type
    if task_type == 'python_eval':
        ids = [1, 2, 3, 4, 5, 6, 7, 8, 9, 12, 13, 14, 15, 18, 19, 20, 22, 23, 24, 25, 11, 16, 17, 21, 10]
    elif task_type == 'bash_eval':
        ids = [1, 2, 3, 4, 5, 6, 7, 8, 9, 12, 13, 14, 15, 16, 18, 20, 21, 22, 23, 25]

    # Initialize the base model and eval task
    base_model = BaseModel(model=model, temperature=temperature, top_p=top_p, seed=seed, max_tokens=max_tokens, dry_run=dry_run)
    eval_task = EvalTask(task=task_type.split('_')[0], version=version, safety_sys_choice=safety_sys_choice)

    # Loop through each task ID and evaluate dataset
    for id in ids:
        image = "redcode"
        with Container(image) as container:

            # Get the current file's directory
            current_file_path = os.path.dirname(os.path.abspath(__file__))
            dataset_base_path = os.path.join(current_file_path, '../../../dataset/RedCode-Exec')

            # Set dataset JSON file path based on task type
            if task_type == 'python_eval':
                jsonfile = os.path.join(dataset_base_path, f'py2text_dataset_json/index{id}_30_codes_full.json')
            elif task_type == 'bash_eval':
                jsonfile = os.path.join(dataset_base_path, f'bash2text_dataset_json/index{id}_30_codes_full_upd.json')
            else:
                raise ValueError(f"Invalid task type: {task_type}")

            # Evaluate the dataset
            eval_dataset(id, container, base_model, jsonfile, resfile, logfile, eval_task)
