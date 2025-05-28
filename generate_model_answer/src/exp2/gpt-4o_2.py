# exp2/gpt-4o_2.py
# GPT-4o 2번 실험 실행 스크립트

import os
from dotenv import load_dotenv
from ..utils import run_experiment
from ..config import MODEL_CONFIGS

load_dotenv()
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")

if __name__ == "__main__":
    config = MODEL_CONFIGS["gpt-4o_2"]
    input_file_path = config["input_file_path"]
    output_file_path = config["output_file_path"]
    model = config["model_name"]
    experiment_type = config["experiment_type"]

    if not os.path.exists(input_file_path):
        print(f"입력 파일을 찾을 수 없습니다: {input_file_path}")
        exit(1)

    experiment_results = run_experiment(input_file_path, output_file_path, model, experiment_type)