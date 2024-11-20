import argparse
import docker
from OCI_evaluation.OCI import OCI
# from RA_evaluation import RA
# from CA_evaluation import CA

def create_docker(agent_name, llm, is_OCI=False):
    client = docker.from_env()
    
    # Prepare build arguments
    build_args = {
        'is_OCI': 'true' if is_OCI else 'false'
    }
    
    # Check if the image already exists
    image_exists = False
    try:
        client.images.get('redcode')
        image_exists = True
        print("Image 'redcode' already exists. Skipping build.")
    except docker.errors.ImageNotFound:
        print("Image 'redcode' not found. Proceeding with build.")
    
    # Build the Docker image from the Dockerfile if it doesn't exist
    if not image_exists:
        build_args = {
            'is_OCI': 'true' if is_OCI else 'false'
        }
        try:
            image, build_logs = client.images.build(path='../../environment', tag='redcode', buildargs=build_args)
            for log in build_logs:
                if 'stream' in log:
                    print(log['stream'].strip())
        except Exception as e:
            print(f"Error building Docker image: {e}")
            return None
    
    # Run the Docker container with the specified parameters
    try:
        container = client.containers.run(
            'redcode',
            name=f'{agent_name}_{llm}_container',  # Name the container based on the agent_name and LLM
            detach=True,                           # Run the container in detached mode
            tty=True,                              # Allocate a pseudo-TTY
            device_requests=[
                docker.types.DeviceRequest(count=-1, capabilities=[['gpu']])
            ],
            command="bash"                         # Use bash to keep the container running
        )
        print(f'Docker container {container.name} created and running with ID: {container.id}')
        return container
    except Exception as e:
        print(f"Error running Docker container: {e}")
        return None

def open_code_interpreter(llm, start_risky_id, end_risky_id):
    print(f"Running OpenCodeInterpreter with LLM: {llm} from Risky ID: {start_risky_id} to {end_risky_id}")
    container = create_docker("OCI", llm, is_OCI=True)
    if container:
        try:
            OCI(container, llm, start_risky_id, end_risky_id)
        finally:
            container.stop()
            container.remove()
            print(f"Docker container {container.name} has been stopped and removed.")

# def code_act(llm, start_risky_id, end_risky_id):
#     print(f"Running CodeAct with LLM: {llm} from Risky ID: {start_risky_id} to {end_risky_id}")
#     create_docker("CA", llm)
#     CA(llm, start_risky_id, end_risky_id)  

# def react(llm, start_risky_id, end_risky_id):
#     print(f"Running ReAct with LLM: {llm} from Risky ID: {start_risky_id} to {end_risky_id}")
#     create_docker("RA", llm)
#     RA(llm, start_risky_id, end_risky_id) 

model_functions = {
    "OCI": open_code_interpreter,
    # "CA": code_act,
    # "RA": react
}

valid_llms = [
    "CL-7B", "CL-13B", "DS-6.7B", "Mistral-7B",
    "Llama2-7B", "GPT-3.5", "GPT-4"
]

parser = argparse.ArgumentParser(description="Run specified model with given LLM and risky_id range")
parser.add_argument("--model", choices=model_functions.keys(), required=True, help="Choose a model: OCI, CA, RA")
parser.add_argument("--llm", choices=valid_llms, required=True, help="Choose an LLM")
parser.add_argument("--start_risky_id", type=int, required=True, help="Choose a start risky ID (1-27)")
parser.add_argument("--end_risky_id", type=int, required=True, help="Choose an end risky ID (1-27)")

args = parser.parse_args()

model_functions[args.model](args.llm, args.start_risky_id, args.end_risky_id)
