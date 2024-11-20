## Usage

### Command-Line Arguments

- `--model`: Choose the model to run. Available options are:
  - `OCI`: OpenCodeInterpreter
  - `CA`: CodeAct
  - `RA`: ReAct

- `--llm`: Choose the LLM to use. Available options are:
  - `CL-7B`
  - `CL-13B`
  - `DS-6.7B`
  - `Mistral-7B`
  - `Llama2-7B`
  - `GPT-3.5`
  - `GPT-4`

- `--start_risky_id, end_risky_id`: Evaluation from start_risky_id to end_risky_id, evaluation includes start_risky_id and end_risky_id (i.e., [start_risky_id,end_risky_id]).  risky ID can be chosen from 1  to 27).

### Example

To run the script with the `OCI` model, `CL-7B` LLM, and risky ID of `5` and '6', use the following command:

```python
python main.py --model OCI --llm CL-7B --start_risky_id 5 --end_risky_id 6
```