import numpy as np
from typing import Dict, List, Any


class CategoryManager:
    @staticmethod
    def normalize_category_name(category: str) -> str:
        normalized = category.strip().lower()
        parts = [part.strip() for part in normalized.split('&')]
        unique_parts = []
        
        for part in parts:
            if part not in unique_parts:
                unique_parts.append(part)
        
        return ' & '.join(unique_parts)
    
    @staticmethod
    def create_empty_evaluation_results() -> Dict[str, Dict[str, List]]:
        return {
            'short_answer': {
                'exact_match': [], 
                'f1': [], 
                'bert_score_f1': [],
                'bluert_score': [],
                'rouge-1': [],
                'bleu-1': []
            },
            'summarization': {
                'rouge-1': [], 
                'rouge-2': [], 
                'rouge-l': [],
                'bert_score_f1': []
            }
        }
    
    @staticmethod
    def merge_category_results(category_evaluation_results: Dict) -> Dict:
        merged_results = {}
        
        for category, results in category_evaluation_results.items():
            norm_category = CategoryManager.normalize_category_name(category)
            
            if norm_category not in merged_results:
                merged_results[norm_category] = CategoryManager.create_empty_evaluation_results()
            
            for task_type, task_results in results.items():
                if isinstance(task_results, dict):
                    for metric, values in task_results.items():
                        if isinstance(values, list):
                            merged_results[norm_category][task_type][metric].extend(values)
        
        return merged_results


class ResultsProcessor:
    @staticmethod
    def calculate_mean_if_exists(values: List[float]) -> float:
        return np.mean(values) if values else None
    
    @staticmethod
    def create_summary_results(evaluation_results: Dict) -> Dict[str, Dict[str, Any]]:
        summary_results = {}
        
        for task_type, metrics in evaluation_results.items():
            summary_results[task_type] = {}
            
            for metric_name, values in metrics.items():
                summary_results[task_type][metric_name] = ResultsProcessor.calculate_mean_if_exists(values)
            
            first_metric = next(iter(metrics.values()), [])
            summary_results[task_type]['count'] = len(first_metric)
        
        return summary_results
    
    @staticmethod
    def create_category_summary(merged_category_results: Dict) -> Dict[str, Dict[str, Dict[str, Any]]]:
        category_summary = {}
        
        for category, results in merged_category_results.items():
            category_summary[category] = ResultsProcessor.create_summary_results(results)
        
        return category_summary
    
    @staticmethod
    def aggregate_metrics_to_results(metrics_list: List[Dict], 
                                   categories: List[str], 
                                   data_ids: List[int]) -> tuple:

        evaluation_results = CategoryManager.create_empty_evaluation_results()
        category_evaluation_results = {}
        
        for metrics in metrics_list:
            global_idx = metrics.get('global_idx')
            if global_idx is None or global_idx >= len(categories):
                continue
            
            category = categories[global_idx]
            
            if category not in category_evaluation_results:
                category_evaluation_results[category] = CategoryManager.create_empty_evaluation_results()
            
            task_type = None
            if 'exact_match' in metrics or 'f1' in metrics:
                task_type = 'short_answer'
            elif any(key.startswith('rouge') for key in metrics.keys()):
                task_type = 'summarization'
            
            if task_type:
                for metric_name, value in metrics.items():
                    if (metric_name in evaluation_results[task_type] and 
                        not metric_name.endswith('_idx')):
                        evaluation_results[task_type][metric_name].append(value)
                
                for metric_name, value in metrics.items():
                    if (metric_name in category_evaluation_results[category][task_type] and 
                        not metric_name.endswith('_idx')):
                        category_evaluation_results[category][task_type][metric_name].append(value)
        
        return evaluation_results, category_evaluation_results


class EvaluationLogger:
    @staticmethod
    def log_progress(current: int, total: int, step: int = 100):
        if current % step == 0:
            print(f"Processing {current+1}/{total}")
    
    @staticmethod
    def log_file_processing(filename: str, exists: bool = False):
        if exists:
            print(f"'{filename}' already exists. Skipping...")
        else:
            print(f"Processing {filename}...")
    
    @staticmethod
    def log_evaluation_summary(filename: str, summary: Dict):
        print(f"===== {filename} Evaluation Results =====")
        for task_type, metrics in summary.items():
            print(f"\n{task_type.upper()}:")
            for metric_name, value in metrics.items():
                if isinstance(value, (int, float)):
                    if isinstance(value, float):
                        print(f"  {metric_name}: {value:.4f}")
                    else:
                        print(f"  {metric_name}: {value}")