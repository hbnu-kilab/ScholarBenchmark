import json
import os
from typing import Dict, List, Tuple, Any


class DataLoader:
    @staticmethod
    def load_jsonl_data(file_path: str) -> Dict[int, Dict[str, Any]]:
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")
            
        data_dict = {}
        with open(file_path, 'r', encoding='utf-8') as f:
            for idx, line in enumerate(f):
                try:
                    item = json.loads(line)
                    if 'id' not in item or item['id'] == '' or not str(item['id']).isdigit():
                        print(f"Invalid ID in line {idx+1}: {item}")
                        continue
                    data_dict[int(item['id'])] = item
                except json.JSONDecodeError as e:
                    print(f"JSON decode error (line {idx+1}): {e}")
        return data_dict
    
    @staticmethod
    def get_common_ids(ground_truth_data: Dict, model_results_data: Dict) -> set:
        return set(ground_truth_data.keys()) & set(model_results_data.keys())
    
    @staticmethod
    def collect_evaluation_data(ground_truth_data: Dict, model_results_data: Dict, 
                              common_ids: set) -> Tuple[List, List, List, List, List, List]:
        summarization_data,summarization_categories, summarization_ids  = [], [], []
      
        short_answer_data, short_answer_categories, short_answer_ids = [], [], []

        multiple_choice_data, multiple_choice_categories, multiple_choice_ids  = [], [], []
        
        multiple_select_data, multiple_select_categories, multiple_select_ids = [], [], []
        
        true_false_data, true_false_categories, true_false_ids = [], [], []

        for data_id in common_ids:
            try:
                ground_truth = ground_truth_data[data_id]
                model_result = model_results_data[data_id]
                
                category = ground_truth.get('category', 'unknown')
                
                if ('short_answer' in ground_truth and 
                    'short_answer' in model_result.get('results', {})):
                    gt_answer = ground_truth['short_answer']['answer']
                    model_answer = model_result['results']['short_answer']['model_answer']
                    
                    short_answer_data.append((gt_answer, model_answer))
                    short_answer_categories.append(category)
                    short_answer_ids.append(data_id)
                
                if ('summarization' in ground_truth and 
                    'summarization' in model_result.get('results', {})):
                    gt_summary = ground_truth['summarization']['summary_text']
                    model_summary = model_result['results']['summarization']['model_answer']
                    
                    summarization_data.append((gt_summary, model_summary))
                    summarization_categories.append(category)
                    summarization_ids.append(data_id)
                    
                if ('multiple_choice' in ground_truth and 
                    'multiple_choice' in model_result.get('results', {})):
                    gt_answer = ground_truth['multiple_choice']['answer']
                    model_answer = model_result['results']['multiple_choice']['model_answer']
                    
                    multiple_choice_data.append((gt_answer, model_answer))
                    multiple_choice_categories.append(category)
                    multiple_choice_ids.append(data_id)
                
                if ('multiple_select' in ground_truth and 
                    'multiple_select' in model_result.get('results', {})):
                    gt_data = ground_truth
                    model_data = model_result
                    
                    multiple_select_data.append((gt_data, model_data))
                    multiple_select_categories.append(category)
                    multiple_select_ids.append(data_id)
                
                if ('true_false' in ground_truth and 
                    'true_false' in model_result.get('results', {})):
                    gt_raw = ground_truth['true_false']['answer']
                    
                    if gt_raw in ["참", "True"]:
                        gt_answer = '1'
                    elif gt_raw in ["거짓", "False"]:
                        gt_answer = '0'
                    else:
                        gt_answer = gt_raw
                    
                    model_answer = model_result['results']['true_false']['model_answer']
                    
                    true_false_data.append((gt_answer, model_answer))
                    true_false_categories.append(category)
                    true_false_ids.append(data_id)

            except KeyError as e:
                print(f"Error processing ID {data_id}: {e}")
        
        return (short_answer_data, short_answer_categories, short_answer_ids,
                summarization_data, summarization_categories, summarization_ids,
                multiple_choice_data, multiple_choice_categories, multiple_choice_ids,
                multiple_select_data, multiple_select_categories, multiple_select_ids,
                true_false_data, true_false_categories, true_false_ids)