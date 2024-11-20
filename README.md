# RedCode

> We are working hard to wrap up all the codes to provide an off-the-shelf deployment experience.


## Repository Structure

### dataset

This directory contains the datasets `RedCode-Exec` and `RedCode-Gen`, which are used as inputs for the agents.

### environment

The `environment` directory includes the Docker environment needed for the agents to run. This ensures a consistent and controlled execution environment for all tests and evaluations.

### evaluation

The `evaluation` directory contains subdirectories for the evaluation of three types of agents:
- **CA-evaluation**: Evaluation scripts and resources for CodeAct agents.
- **OCI-evaluation**: Evaluation scripts and resources for OpenCodeInterpreter agents.
- **RA-evaluation**: Evaluation scripts and resources for ReAct agents.

Additionally, `evaluation.py` that serve as evaluation scripts for each risky scenario.

### result

The `result` directory stores the results of the evaluations.

### scripts

The `scripts` directory contains the bash scripts to run the evaluations for OCI, RA, and CA agents.

## Environment Setup
```bash
conda env create -f environment.yml
conda activate redcode
```

## Usage
```bash
./scripts/OCI_eval.sh
```