# RedCode-Gen

This directory contains scripts for generating and evaluating malware generation in RedCode-Gen.

Here are the command-line arguments you need to provide:

```
# For normal evaluation
python main.py --model gpt-3.5-turbo --evaluation_type normal

# For agent evaluation
python main.py --model claude-3-opus-20240229 --evaluation_type agent --rounds 3

# To include VirusTotal evaluation
python main.py --model gpt-3.5-turbo --evaluation_type normal --use_virustotal
```
