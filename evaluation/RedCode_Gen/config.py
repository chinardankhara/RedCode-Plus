import os
import argparse
from dotenv import load_dotenv

load_dotenv()

def get_config():
    parser = argparse.ArgumentParser(description="RedCode-Gen")
    parser.add_argument("--base_dir", default=os.getenv("BASE_DIR", "./data"), help="Directory containing the prompts")
    parser.add_argument("--output_dir", default=os.getenv("OUTPUT_DIR", "./solutions"), help="Directory to store solutions")
    parser.add_argument("--model", default=os.getenv("MODEL", "gpt-3.5-turbo"), help="Model to evaluate")
    parser.add_argument("--judge_model", default=os.getenv("JUDGE_MODEL", "gpt-4"), help="Model to use for judging")
    parser.add_argument("--evaluation_type", choices=["normal", "agent"], default="normal", help="Type of evaluation to perform")
    parser.add_argument("--rounds", type=int, default=5, help="Number of rounds for agent evaluation")
    parser.add_argument("--use_virustotal", action="store_true", help="Use VirusTotal for additional evaluation")
    args = parser.parse_args()

    config = {
        "base_dir": args.base_dir,
        "output_dir": args.output_dir,
        "model": args.model,
        "judge_model": args.judge_model,
        "evaluation_type": args.evaluation_type,
        "rounds": args.rounds,
        "use_virustotal": args.use_virustotal,
        "openai_api_key": os.getenv("OPENAI_API_KEY"),
        "together_api_key": os.getenv("TOGETHER_API_KEY"),
        "anthropic_api_key": os.getenv("ANTHROPIC_API_KEY"),
        "virustotal_api_key": os.getenv("VIRUSTOTAL_API_KEY"),
    }

    return config