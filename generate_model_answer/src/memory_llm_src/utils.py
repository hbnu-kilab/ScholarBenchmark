import argparse
from collections import defaultdict
import json
import os
from tqdm.auto import tqdm
import torch
import json
from transformers import AutoTokenizer, AutoConfig
import importlib
from vllm import LLM, SamplingParams

def arg_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_nick", type=str, required=True, default='llama-70b')
    parser.add_argument("--exp_type", type=str, default='')
    parser.add_argument("--task_list", type=str, required=True, default='["multiple_choice"]')
    parser.add_argument("--data_path", type=str, required=True)
    parser.add_argument("--save_path", type=str, default='./')
    parser.add_argument("--lang_type", type=str, default='ko')

    return parser.parse_args()

def process_file(json_sample, task):
    if task != 'summarization':
        q = json_sample[task]["question"]
        answer = json_sample[task]["answer"]
        
    topic = ''
    paragraph = ''
    idx = json_sample['id']

    if type(json_sample['paragraph']) == list:
        for paragraph_dict in json_sample['paragraph']:
            try:
                temp = str(paragraph_dict['section'])
                if type(paragraph_dict['text']) == list:
                    for t in paragraph_dict['text']:
                        temp += t
                else:
                    temp += str(paragraph_dict['text'])
                    
                paragraph += temp
            except:
                paragraph += str(paragraph_dict)
    else:
        paragraph = str(json_sample['paragraph'])
        
    if task in ('multiple_choice', 'multiple_select'):
        choices = json_sample[task]['choices']
        topic = json_sample[task]['topic']
        paragraph = json_sample[task]['paragraph']
        model_input = f"Question: \n{q}\n\nChoices: \n" + "\n".join(choices)
    elif task == 'summarization':            
        answer = json_sample[task]['summary_text']
        topic = json_sample[task]['topic']
        model_input = f"Paragraph: \n{paragraph}"
    else:
        topic = json_sample[task]['topic']
        model_input = f"Question: \n{q}"

    
    # 최종 데이터 구성 (model_output에는 answer 문자열을 그대로 사용)
    new_data = {
        "model_input": model_input,
        "model_output": answer,
        "topic": topic,
        "paragraph": paragraph,
        "id": idx
    }
    
    return new_data

def parsed_input_data(data_path, lang_type, task_list):
    ds = []
    
    # with open(data_path, 'r', encoding='utf-8') as f:
    #     for json_file in f:
    #         ds.append(json.loads(json_file))
    
    with open(data_path, 'r', encoding='utf-8') as f:
        ds = json.load(f)
        
    ds_dict = defaultdict(list)
    
    for sample in ds:
        idx = sample['id']
        paragraph = sample['paragraph']
        category = sample['category']
        
        for task in task_list:
            if task in sample.keys():
                sample[task]['paragraph'] = paragraph
                sample[task]['id'] = idx
                
                if task in ('multiple_choice', 'multiple_select'):
                    if all(k in sample[task].keys() for k in ('answer', 'question', 'choices')):
                        ds_dict[task].append(process_file(sample, task))
                elif task == 'summarization':
                    sample[task]['topic'] = category
                    if 'summary_text' in sample[task].keys():
                        ds_dict[task].append(process_file(sample, task))
                else:
                    if all(k in sample[task].keys() for k in ('answer', 'question')):
                        if task == 'true_false':
                            false_label = '거짓' if lang_type == 'ko' else 'False'
                            sample[task]['answer'] = 0 if sample[task]['answer'] == false_label else 1
                        ds_dict[task].append(process_file(sample, task))
        
    return ds_dict

def result_clansing(result_text, task):
    result_str = str(result_text).strip()
    
    think_token_list = ['</think>', '</thought>', '<solution>']
    no_think_token_list = ['답변:', '정답:', '**답변:**', '**정답:**']

    final_think_token = ''
    for think_token in think_token_list:
        if think_token in result_str:
            final_think_token = think_token
            break
    
    if not final_think_token:
        for think_token in no_think_token_list:
            if think_token in result_str:
                final_think_token = think_token
                break
        if not final_think_token:
            final_think_token = '\n'
    
    result_str = result_str.split(final_think_token)[-1].strip()

    if not result_str:
        return ''
    
    if task == 'multiple_choice':
        if result_str.lower()[0] in ('a', 'b', 'c', 'd'):
            return result_str.lower()[0]
        
        return ''
    
    elif task == 'multiple_select':
        try:
            return str([ans.strip().lower()[0] for ans in eval(result_str.strip())])
        except:
            return str([])
    elif task in ('summarization', 'short_answer'):
        return result_str.strip()
    elif task == 'true_false':
        if '참' in result_str.lower() or 'true' in result_str.lower() or '1' in result_str.lower():
            return '1'
        elif '거짓' in result_str.lower() or 'false' in result_str.lower() or '0' in result_str.lower():
            return '0'
        else:
            return ''
    else:
        return ''
    

def inference_worker(dataset_dict, args):
    config_module = importlib.import_module('configs' if args.lang_type == 'ko' else 'configs_en')
    
    model_nicks = config_module.model_nicks
    template_dict = config_module.template_dict
    inst_dict = config_module.inst_dict
    cot_inst_dict = config_module.cot_inst_dict
    cot_path = config_module.cot_path

    task_list = json.loads(args.task_list)    
    print(f"{args.model_nick}_{args.exp_type} 추론 시작!!")

    model_nick = args.model_nick
    model_name = model_nicks[model_nick]
    
    llm = LLM(
        model=model_name,
        trust_remote_code=True, 
        tensor_parallel_size=4,
        dtype=torch.bfloat16,
        gpu_memory_utilization=0.9,
        seed=42,
    )

    generation_kwargs = SamplingParams(
        temperature=0,
        max_tokens=32000,
    )

    max_token_len = AutoConfig.from_pretrained(model_name, trust_remote_code=True).max_position_embeddings
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

    for task in task_list:
        if task not in dataset_dict.keys():
            continue
        
        data_list = dataset_dict[task]["dataset"]
        fewshot_samples = dataset_dict[task]["fewshot"]
        template = template_dict[task]
        
        if len(data_list) == 0:
            continue
        
        template = template_dict[task]
        
        if args.exp_type == 'paragraph': 
            template = '\nParagraph\n문제와 관련된 지문\n' + template_dict[task]
        elif args.exp_type == 'topic':
            template = '\nTopic\n문제에 대한 주제\n' + template_dict[task]
        elif args.exp_type == 'cot':
            template = '\nParagraph\n문제와 관련된 지문\n' + '\nCot\n문제 풀이에 관련된 cot path\n</think>\n' + template_dict[task]

        for i in range(2):
            if args.exp_type == 'paragraph':
                fewshot_samples[i]["model_input"] = f'Paragraph\n{str(fewshot_samples[i]["paragraph"])}\n\n' + fewshot_samples[i]['model_input']
            elif args.exp_type == 'topic':
                fewshot_samples[i]["model_input"] = f'Topic\n{fewshot_samples[i]["topic"]}\n\n' + fewshot_samples[i]['model_input']
            elif args.exp_type == 'cot':
                fewshot_samples[i]["model_input"] = f'Paragraph\n{str(fewshot_samples[i]["paragraph"])}\n\n' + fewshot_samples[i]['model_input']
                fewshot_samples[i]["model_output"] = f'Cot\n{cot_path[f"{task}_{i+1}"]}' + str(fewshot_samples[i]['model_output'])                

        for sample in tqdm(data_list, total=len(data_list), desc=f'{model_nick}_{task}_{args.exp_type}'):
            json_path = os.path.join(args.save_path, f"{args.exp_type}/{model_nick}/{sample['id']}.json")

            if not os.path.exists(json_path):
                result_dict = {}
                result_dict['id'] = sample['id']
                result_dict['results'] = {}
                os.makedirs(os.path.dirname(json_path), exist_ok=True)
            else:
                with open(json_path, "r", encoding='utf-8') as f:
                    result_dict = json.load(f)
     
            if args.exp_type == 'paragraph' or args.exp_type == 'cot': 
                model_input = f'Paragraph\n{str(sample["paragraph"])}\n\n' + sample['model_input']
            elif args.exp_type == 'topic':
                model_input = f'Topic\n{sample["topic"]}\n\n' + sample['model_input']
            else:
                model_input = sample['model_input']
                
            temp_inst_dict = cot_inst_dict if args.exp_type == 'cot' else inst_dict
            
            messages = [
                {"role": "user", "content": temp_inst_dict[task].format(template)+"\n\n" + fewshot_samples[0]["model_input"]},
                {"role": "assistant", "content": str(fewshot_samples[0]["model_output"])},
                {"role": "user", "content": temp_inst_dict[task].format(template)+"\n\n" + fewshot_samples[1]["model_input"]},
                {"role": "assistant", "content": str(fewshot_samples[1]["model_output"])},
                {"role": "user", "content": temp_inst_dict[task].format(template)+"\n\n" + model_input},
                ]

            input_ids = tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=True)
            input_ids = input_ids[-(max_token_len-10):]
            input_text = tokenizer.decode(input_ids, skip_special_tokens=False)

            is_generated = False
            truncation_value = 10

            while not is_generated:
                try:
                    outputs = llm.generate(
                        input_text,
                        sampling_params=generation_kwargs,
                        use_tqdm=False
                    )
                    is_generated = True
                except ValueError as e:
                    if "Prompt length" in str(e) and "is longer than the maximum model length" in str(e):
                        input_ids = input_ids[truncation_value:]
                        input_text = tokenizer.decode(input_ids, skip_special_tokens=False)

            output_text = outputs[0].outputs[0].text

            result_dict['results'][task] = {
                'model_original_answer': output_text,
                'model_answer': result_clansing(output_text, task)}

            with open(json_path, "w", encoding='utf-8') as f:
                json.dump(result_dict, f)