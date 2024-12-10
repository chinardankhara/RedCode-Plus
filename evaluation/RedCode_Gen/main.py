from RedCode_Gen.config import get_config
from RedCode_Gen.evaluation import evaluate_model

def main():
    config = get_config()
    evaluate_model(config)

if __name__ == "__main__":
    main()