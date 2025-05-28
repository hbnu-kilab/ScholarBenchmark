import os
import json
import glob

def convert_json_to_jsonl_in_model_dirs(base_dir):
    for model_dir_name in os.listdir(base_dir):
        model_dir_path = os.path.join(base_dir, model_dir_name)

        if not os.path.isdir(model_dir_path):
            continue  

        json_files = glob.glob(os.path.join(model_dir_path, "*.json"))
        if not json_files:
            continue  

        for json_file in json_files:
            try:
                with open(json_file, "r", encoding="utf-8") as f:
                    data = json.load(f)

                jsonl_path = os.path.splitext(json_file)[0] + ".jsonl"

                with open(jsonl_path, "w", encoding="utf-8") as jsonl_file:
                    for item in data:
                        jsonl_file.write(json.dumps(item, ensure_ascii=False) + "\n")

                os.remove(json_file)  # 원본 json 파일 삭제
                print(f"Converted and replaced: {json_file} -> {jsonl_path}")
            except Exception as e:
                print(f"Error processing {json_file}: {e}")

base_directory = "Json file path"
convert_json_to_jsonl_in_model_dirs(base_directory)
