# utils.py
# 공통 함수 정의

import openai
import json
import time
import ast
from tqdm import tqdm
from src.prompts import inst_dict_1, inst_dict_2, inst_dict_3, inst_dict_4, inst_dict_5, few_shot_mc1, few_shot_ms1, few_shot_sa1, few_shot_tf1, few_shot_sum1, few_shot_mc2, few_shot_ms2, few_shot_sa2, few_shot_tf2, few_shot_sum2

def format_multiple_select_response(response):
    if '[' in response and ']' in response:
        try:
            list_str = response[response.find('['):response.find(']') + 1]
            eval_list = ast.literal_eval(list_str)
            return eval_list
        except:
            pass

    if ',' in response:
        return [item.strip().strip("'\"") for item in response.split(',')]

    return [response.strip().strip("'\"")]

def format_multiple_choice_response(response):
    for char in response:
        if char.upper() in ["A", "B", "C", "D"]:
            return char.lower()
    return response

def query_llm(prompts, model, max_retries=3, retry_delay=2):
    print(f"Using model: {model}")
    responses = []

    for prompt in prompts:
        print(prompt)
        for attempt in range(max_retries):
            try:
                client = openai.OpenAI()
                params = {
                    "model": model,
                    "messages": [{"role": "user", "content": prompt}]
                }
                if model in ["o1-mini", "o3-mini"]:
                    params["max_completion_tokens"] = 2000
                else:
                    params["max_tokens"] = 2000
                response = client.chat.completions.create(**params)
                responses.append(response.choices[0].message.content.strip())
                break
            except Exception as e:
                if attempt < max_retries - 1:
                    print(f"오류 발생: {e}. {retry_delay}초 후 재시도합니다...")
                    time.sleep(retry_delay)
                    retry_delay *= 2
                else:
                    print(f"최대 재시도 횟수 초과. 오류: {e}")
                    responses.append("오류: API 호출 실패")

    return responses

def run_experiment(input_file_path, output_file_path, model, experiment_type):
    existing_results = []
    processed_ids = set()

    try:
        with open(output_file_path, "r", encoding="utf-8") as f:
            existing_results = json.load(f)
            processed_ids = {str(result.get('id')) for result in existing_results}
    except (FileNotFoundError, json.JSONDecodeError):
        pass

    results = existing_results if existing_results else []

    with open(input_file_path, "r", encoding="utf-8") as file:
        try:
            data = [json.loads(line) for line in file]
        except Exception as e:
            print(f"데이터 파일 파싱 오류: {e}")
            return results

    inst_dict = {
        "1": inst_dict_1,
        "2": inst_dict_2,
        "3": inst_dict_3,
        "4": inst_dict_4,
        "5": inst_dict_5
    }.get(experiment_type, inst_dict_1)

    task_types = ['multiple_choice', 'multiple_select', 'short_answer', 'true_false']
    if experiment_type in ["1", "4", "5"]:
        task_types.append('summarization')

    for item in tqdm(data, desc="Processing items"):
        if str(item.get('id')) in processed_ids:
            print(f"\n이미 처리한 ID: {item.get('id', 'unknown')}")
            continue

        item_result = {
            "id": item.get("id", ""),
            "pid": item.get("pid", ""),
            "category": item.get("category", ""),
            "results": {}
        }

        prompts = []

        print(f"\nProcessing item ID: {item.get('id', 'unknown')}")

        for task_type in task_types:
            if task_type in item:
                if task_type == 'summarization':
                    format_args = {
                        "paragraph": item.get("paragraph", ""),
                        "few_shot1": few_shot_sum1,
                        "few_shot2": few_shot_sum2
                    }
                    if experiment_type == "5":
                        format_args["category"] = item.get("category", "")
                    prompts.append(inst_dict[task_type].format(**format_args))
                elif task_type == 'short_answer':
                    format_args = {
                        "question": item[task_type]['question'],
                        "few_shot1": few_shot_sa1,
                        "few_shot2": few_shot_sa2
                    }
                    if experiment_type in ["3", "4"]:
                        format_args["paragraph"] = item.get('paragraph', '')
                    elif experiment_type == "2":
                        format_args["topic"] = item[task_type].get('topic', '')
                    elif experiment_type == "5":
                        format_args["category"] = item.get("category", "")
                    prompts.append(inst_dict[task_type].format(**format_args))
                elif task_type == 'true_false':
                    format_args = {
                        "question": item[task_type]['question'],
                        "few_shot1": few_shot_tf1,
                        "few_shot2": few_shot_tf2
                    }
                    if experiment_type in ["3", "4"]:
                        format_args["paragraph"] = item.get('paragraph', '')
                    elif experiment_type == "2":
                        format_args["topic"] = item[task_type].get('topic', '')
                    elif experiment_type == "5":
                        format_args["category"] = item.get("category", "")
                    prompts.append(inst_dict[task_type].format(**format_args))
                elif task_type == 'multiple_choice':
                    choices_text = "\n".join(item[task_type]['choices'])
                    format_args = {
                        "question": item[task_type]['question'],
                        "choices": choices_text,
                        "few_shot1": few_shot_mc1,
                        "few_shot2": few_shot_mc2
                    }
                    if experiment_type in ["3", "4"]:
                        format_args["paragraph"] = item.get('paragraph', '')
                    elif experiment_type == "2":
                        format_args["topic"] = item[task_type].get('topic', '')
                    elif experiment_type == "5":
                        format_args["category"] = item.get("category", "")
                    prompts.append(inst_dict[task_type].format(**format_args))
                elif task_type == "multiple_select":
                    choices_text = "\n".join(item[task_type]['choices'])
                    format_args = {
                        "question": item[task_type]['question'],
                        "choices": choices_text,
                        "few_shot1": few_shot_ms1,
                        "few_shot2": few_shot_ms2
                    }
                    if experiment_type in ["3", "4"]:
                        format_args["paragraph"] = item.get('paragraph', '')
                    elif experiment_type == "2":
                        format_args["topic"] = item[task_type].get('topic', '')
                    elif experiment_type == "5":
                        format_args["category"] = item.get("category", "")
                    prompts.append(inst_dict[task_type].format(**format_args))

        if prompts:
            max_attempts = 5
            for i, task_type in enumerate(task_types):
                if i < len(prompts):
                    current_prompt = prompts[i]
                    response = None
                    attempts = 0

                    while attempts < max_attempts:
                        llm_response = query_llm([current_prompt], model)[0]
                        print(f"Response: {llm_response}")

                        if task_type == 'summarization':
                            response = llm_response
                        elif task_type == 'short_answer':
                            response = llm_response
                        elif task_type == 'true_false':
                            response = llm_response
                        elif task_type == 'multiple_choice':
                            response = format_multiple_choice_response(llm_response)
                        elif task_type == 'multiple_select':
                            response = format_multiple_select_response(llm_response)

                        is_valid_response = False
                        if isinstance(response, str):
                            is_valid_response = response and response.strip()
                        elif isinstance(response, list):
                            is_valid_response = len(response) > 0
                        else:
                            is_valid_response = bool(response)

                        if is_valid_response:
                            break

                        attempts += 1
                        if attempts < max_attempts:
                            print(f"Empty response for {task_type}, retrying... (Attempt {attempts + 1}/{max_attempts})")

                    item_result["results"][task_type] = {"model_answer": response}

        results.append(item_result)

        with open(output_file_path, "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)

        print(f"Completed item ID: {item.get('id', 'unknown')}\n")

    return results