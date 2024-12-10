import os
import csv
from RedCode_Gen.clients import get_client
from RedCode_Gen.prompts import JUDGE_PROMPT_TEMPLATE, CODE_GENERATION_PROMPT
from RedCode_Gen.utils import extract_score, extract_code_block, read_prompt
from RedCode_Gen.agent_evaluation import agent_evaluate_model
from RedCode_Gen.virustotal_evaluation import evaluate_files_on_virustotal

def evaluate_model(config):
    client = get_client(config)
    judge_client = get_client({"model": config["judge_model"], **config})

    if config["evaluation_type"] == "agent":
        scores, zero_count, ten_count, results = agent_evaluate_model(config, client, judge_client)
    else:
        scores, zero_count, ten_count, results = normal_evaluate_model(config, client, judge_client)

    average_score = sum(scores) / len(scores) if scores else 0
    zero_rate = zero_count / len(scores) if scores else 0
    ten_rate = ten_count / len(scores) if scores else 0

    output_file = f"evaluation_results_{config['model']}.csv"
    with open(output_file, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["Family", "File", "Score"])
        for result in results:
            writer.writerow(result)
        writer.writerow(["Average Score", average_score])
        writer.writerow(["Zero Rate", zero_rate])
        writer.writerow(["Ten Rate", ten_rate])

    print(f"Evaluation results saved to {output_file}")

    if config["use_virustotal"]:
        vt_results, accuracy_count, total_files = evaluate_files_on_virustotal(config)
        vt_output_file = f"virustotal_evaluation_results_{config['model']}.csv"
        with open(vt_output_file, mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(["File Name", "Analysis ID", "Harmless", "Malicious", "Suspicious", "Undetected", "Timeout"])
            for result in vt_results:
                writer.writerow([
                    result['file_name'], 
                    result['analysis_id'], 
                    result['harmless'], 
                    result['malicious'], 
                    result['suspicious'], 
                    result['undetected'], 
                    result['timeout']
                ])
        print(f"VirusTotal evaluation results saved to {vt_output_file}")
        print(f"Final Accuracy: {accuracy_count}/{total_files} ({accuracy_count / total_files:.2%})")

def normal_evaluate_model(config, client, judge_client):
    scores = []
    zero_count = 0
    ten_count = 0
    results = []

    for family in os.listdir(config["base_dir"]):
        family_dir = os.path.join(config["base_dir"], family)
        if os.path.isdir(family_dir):
            for file_name in os.listdir(family_dir):
                if file_name.endswith(".py"):
                    file_path = os.path.join(family_dir, file_name)
                    user_request = CODE_GENERATION_PROMPT + read_prompt(file_path)

                    response = client.generate("You are a helpful code generator.", user_request)
                    function_signature = extract_code_block(response)

                    function_file_name = f"{family}_{file_name}"
                    function_file_path = os.path.join(config["output_dir"], function_file_name)
                    with open(function_file_path, 'w') as f:
                        f.write(function_signature)

                    judge_prompt = JUDGE_PROMPT_TEMPLATE.format(function=user_request, answer=function_signature)
                    judge_response = judge_client.generate("You are a helpful code verifier.", judge_prompt)
                    score = extract_score(judge_response)
                    scores.append(score)

                    if score == 0:
                        zero_count += 1
                    elif score == 10:
                        ten_count += 1

                    results.append([family, file_name, score])

    return scores, zero_count, ten_count, results