"""
Metrics calculation for Scholar Benchmark evaluation
"""
import ast
import numpy as np
from typing import List, Tuple, Dict, Any
from tqdm import tqdm


class MetricsCalculator:
    def __init__(self, language: str = "ko"):
        self.language = language
        print(f"Metrics calculator initialized for language: {self.language}")
    
    def calculate_multiple_choice_accuracy(self, reference: str, hypothesis: str) -> float:
        return 1.0 if reference == hypothesis else 0.0
    
    def calculate_multiple_select_score(self, ground_truth: Dict, model_result: Dict) -> Dict[str, float]:
        def is_correct(predict, true_label, n):
            if set(true_label) == set(predict):
                return True
            
            if set(predict).issubset(set(true_label)):
                if len(set(true_label).intersection(set(predict))) >= n:
                    return True
                else:
                    return False
            else:
                return False
        
        def is_correct_per_answer_num(predict, true_label, n):
            if set(predict).issubset(set(true_label)):
                if len(set(predict).intersection(set(true_label))) == n:
                    return True
                else:
                    return False
            else:
                return False

        predict_list = model_result['results']['multiple_select']['model_answer']
        label_list = ground_truth['multiple_select']['answer']

        if isinstance(predict_list, str):
            try:
                predict_list = eval(predict_list)
            except:
                try:
                    predict_list = ast.literal_eval(predict_list)
                except:
                    predict_list = []
                    print(f"predict_list parsing failed: {model_result['results']['multiple_select']['model_answer']}")
        
        if not isinstance(predict_list, list):
            predict_list = []
            print(f"predict_list is not a list: {type(predict_list)}")

        score = {}
        
        for n in range(1, 5):
            c = 1 if is_correct(predict_list, label_list, n) else 0
            score[f'a@{n}'] = c
        
        for n in range(1, 5):
            if len(set(label_list)) == n:
                c = 1 if is_correct_per_answer_num(predict_list, label_list, n) else 0
            else:
                c = 0  
            score[f'exact_match_{n}'] = c

        return score
    
    def calculate_true_false_accuracy(self, reference: str, hypothesis: str) -> float:
        return 1.0 if reference == hypothesis else 0.0


class BatchProcessor:
    def __init__(self, metrics_calculator: MetricsCalculator):
        self.calc = metrics_calculator
    
    def process_multiple_choice(self, data_pairs: List[Tuple[str, str]]) -> List[Dict[str, Any]]:
        all_metrics = []
        
        for i, (reference, hypothesis) in enumerate(tqdm(data_pairs, desc="Processing multiple choice")):
            accuracy = self.calc.calculate_multiple_choice_accuracy(reference, hypothesis)
            
            metrics = {
                'accuracy': accuracy,
                'global_idx': i
            }
            
            all_metrics.append(metrics)
        
        return all_metrics
    
    def process_multiple_select(self, data_pairs: List[Tuple[Dict, Dict]]) -> List[Dict[str, Any]]:
        all_metrics = []
        
        for i, (ground_truth, model_result) in enumerate(tqdm(data_pairs, desc="Processing multiple select")):
            scores = self.calc.calculate_multiple_select_score(ground_truth, model_result)
            
            metrics = {
                'a@1': scores.get('a@1', 0),
                'a@2': scores.get('a@2', 0),
                'a@3': scores.get('a@3', 0),
                'a@4': scores.get('a@4', 0),
                'exact_match_1': scores.get('exact_match_1', 0),
                'exact_match_2': scores.get('exact_match_2', 0),
                'exact_match_3': scores.get('exact_match_3', 0),
                'exact_match_4': scores.get('exact_match_4', 0),
                'global_idx': i
            }
            
            all_metrics.append(metrics)
        
        return all_metrics
    
    def process_true_false(self, data_pairs: List[Tuple[str, str]]) -> List[Dict[str, Any]]:
        all_metrics = []
        
        for i, (reference, hypothesis) in enumerate(tqdm(data_pairs, desc="Processing true/false")):
            accuracy = self.calc.calculate_true_false_accuracy(reference, hypothesis)
            
            metrics = {
                'accuracy': accuracy,
                'global_idx': i
            }
            
            all_metrics.append(metrics)
        
        return all_metrics