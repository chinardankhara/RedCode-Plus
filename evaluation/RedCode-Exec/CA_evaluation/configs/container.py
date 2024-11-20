import docker
import json
import signal
import logging
import time

class Container:
    def __init__(self, image):
        self.image = image
        self.client = docker.from_env()
        self.container = self.client.containers.run(
            image,
            labels={"created_by": "code-agent-evaluation"},
            detach=True, tty=True, stdin_open=True
        )
        output = self.container.exec_run("bash /app/init.sh")
        output_str = output.output.decode()
      

    def __del__(self):
        try:
            self.container.stop()
        except:
            pass
        finally:
            self.container.remove()    
    
    def execute_with_retries(self, cmd, retries=3, timeout=1*60):
        def handler(signum, frame):
            raise TimeoutError("Execution timed out")
        
        signal.signal(signal.SIGALRM, handler)
        
        for attempt in range(retries):

     
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

    def execute_independent(self, command, *params):
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
        
        return self.execute_with_retries(cmd)

