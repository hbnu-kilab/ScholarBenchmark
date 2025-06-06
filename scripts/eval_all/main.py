"""
Main evaluation class for NLP model performance evaluation
"""
import os
import json
from typing import Dict, List, Tuple, Any, Optional

from config import Config
from data_loader import DataLoader
from metrics_calculator import MetricsCalculator, BatchProcessor
from evaluation_utils import CategoryManager, ResultsProcessor, EvaluationLogger


class ScholarBenchmarkEvaluator:
    def __init__(self, config: Optional[Config] = None, language: str = "en"):
        self.config = config or Config()
        self.language = language
        
        self.data_loader = DataLoader()
        self.metrics_calculator = MetricsCalculator(
            device=self.config.DEVICE,
            bert_model=self.config.bert_model,
            bleurt_checkpoint=self.config.BLEURT_CHECKPOINT_PATH,
            language=language
        )
        self.batch_processor = BatchProcessor(self.metrics_calculator)
        self.category_manager = CategoryManager()
        self.results_processor = ResultsProcessor()
        self.logger = EvaluationLogger()
    
    def evaluate_single_model(self, ground_truth_file: str, 
                            model_results_file: str, 
                            batch_size: int = None) -> Tuple[Dict, Dict, Dict, Dict]:

        batch_size = batch_size or self.config.DEFAULT_BATCH_SIZE
        
        ground_truth_data = self.data_loader.load_jsonl_data(ground_truth_file)
        model_results_data = self.data_loader.load_jsonl_data(model_results_file)
        
        common_ids = self.data_loader.get_common_ids(ground_truth_data, model_results_data)
        print(f"Common IDs count: {len(common_ids)}")
        
        (short_answer_data, short_answer_categories, short_answer_ids,
         summarization_data, summarization_categories, summarization_ids,
         multiple_choice_data, multiple_choice_categories, multiple_choice_ids,
         multiple_select_data, multiple_select_categories, multiple_select_ids,
         true_false_data, true_false_categories, true_false_ids) = \
            self.data_loader.collect_evaluation_data(ground_truth_data, model_results_data, common_ids)
        
        evaluation_results = self.category_manager.create_empty_evaluation_results()
        category_evaluation_results = {}
        
        if short_answer_data:
            print("Evaluating short answers...")
            sa_metrics = self.batch_processor.process_short_answers(short_answer_data, batch_size)
            
            sa_eval_results, sa_category_results = self.results_processor.aggregate_metrics_to_results(
                sa_metrics, short_answer_categories, short_answer_ids
            )
            
            for metric_name, values in sa_eval_results['short_answer'].items():
                evaluation_results['short_answer'][metric_name].extend(values)
            
            for category, results in sa_category_results.items():
                if category not in category_evaluation_results:
                    category_evaluation_results[category] = self.category_manager.create_empty_evaluation_results()
                
                for metric_name, values in results['short_answer'].items():
                    category_evaluation_results[category]['short_answer'][metric_name].extend(values)
        
        if summarization_data:
            print("Evaluating summarization...")
            sum_metrics = self.batch_processor.process_summarization(summarization_data, batch_size)
            
            sum_eval_results, sum_category_results = self.results_processor.aggregate_metrics_to_results(
                sum_metrics, summarization_categories, summarization_ids
            )
            
            for metric_name, values in sum_eval_results['summarization'].items():
                evaluation_results['summarization'][metric_name].extend(values)
            
            for category, results in sum_category_results.items():
                if category not in category_evaluation_results:
                    category_evaluation_results[category] = self.category_manager.create_empty_evaluation_results()
                
                for metric_name, values in results['summarization'].items():
                    category_evaluation_results[category]['summarization'][metric_name].extend(values)
        
        if multiple_choice_data:
            print("Evaluating multiple choice...")
            mc_metrics = self.batch_processor.process_multiple_choice(multiple_choice_data)
            
            mc_eval_results, mc_category_results = self.results_processor.aggregate_metrics_to_results(
                mc_metrics, multiple_choice_categories, multiple_choice_ids, 'multiple_choice'
            )
            
            for metric_name, values in mc_eval_results['multiple_choice'].items():
                evaluation_results['multiple_choice'][metric_name].extend(values)
            
            for category, results in mc_category_results.items():
                if category not in category_evaluation_results:
                    category_evaluation_results[category] = self.category_manager.create_empty_evaluation_results()
                
                for metric_name, values in results['multiple_choice'].items():
                    category_evaluation_results[category]['multiple_choice'][metric_name].extend(values)
        
        if multiple_select_data:
            print("Evaluating multiple select...")
            ms_metrics = self.batch_processor.process_multiple_select(multiple_select_data)
            
            ms_eval_results, ms_category_results = self.results_processor.aggregate_metrics_to_results(
                ms_metrics, multiple_select_categories, multiple_select_ids, 'multiple_select'
            )
            
            for metric_name, values in ms_eval_results['multiple_select'].items():
                evaluation_results['multiple_select'][metric_name].extend(values)
            
            for category, results in ms_category_results.items():
                if category not in category_evaluation_results:
                    category_evaluation_results[category] = self.category_manager.create_empty_evaluation_results()
                
                for metric_name, values in results['multiple_select'].items():
                    category_evaluation_results[category]['multiple_select'][metric_name].extend(values)
        
        if true_false_data:
            print("Evaluating true/false...")
            tf_metrics = self.batch_processor.process_true_false(true_false_data)
            
            tf_eval_results, tf_category_results = self.results_processor.aggregate_metrics_to_results(
                tf_metrics, true_false_categories, true_false_ids, 'true_false'
            )
            
            for metric_name, values in tf_eval_results['true_false'].items():
                evaluation_results['true_false'][metric_name].extend(values)
            
            for category, results in tf_category_results.items():
                if category not in category_evaluation_results:
                    category_evaluation_results[category] = self.category_manager.create_empty_evaluation_results()
                
                for metric_name, values in results['true_false'].items():
                    category_evaluation_results[category]['true_false'][metric_name].extend(values)

        merged_category_results = self.category_manager.merge_category_results(category_evaluation_results)
        summary_results = self.results_processor.create_summary_results(evaluation_results)
        category_summary = self.results_processor.create_category_summary(merged_category_results)
        
        return summary_results, evaluation_results, category_summary, merged_category_results
    
    def evaluate_multiple_models(self, ground_truth_file: str, 
                                results_dir: str, 
                                output_dir: str,
                                batch_size: int = None,
                                save_detailed_results: bool = True) -> Dict[str, Dict]:
        
        batch_size = batch_size or self.config.DEFAULT_BATCH_SIZE
        os.makedirs(output_dir, exist_ok=True)
        
        all_results = {}
        
        for filename in os.listdir(results_dir):
            if not filename.endswith(".jsonl"):
                continue
            
            model_results_file = os.path.join(results_dir, filename)
            output_file = os.path.join(output_dir, f"evaluation_{filename.replace('.jsonl', '.json')}")
            
            if os.path.exists(output_file):
                self.logger.log_file_processing(output_file, exists=True)
                continue
            
            self.logger.log_file_processing(filename)
            
            try:
                summary, evaluation_results, category_summary, merged_category_results = \
                    self.evaluate_single_model(ground_truth_file, model_results_file, batch_size)
                
                self.logger.log_evaluation_summary(filename, summary)
                
                results_to_save = {'summary': summary}
                if save_detailed_results:
                    results_to_save.update({
                        'category_summary': category_summary,
                        # 'merged_category_results': merged_category_results, # if you check each score delect '#'
                        # 'evaluation_results': evaluation_results
                    })
                
                with open(output_file, 'w', encoding='utf-8') as f:
                    json.dump(results_to_save, f, indent=2, ensure_ascii=False)
                
                all_results[filename] = summary
                
            except Exception as e:
                print(f"Error evaluating {filename}: {e}")
                continue
        
        comprehensive_results_file = os.path.join(output_dir, "comprehensive_results.json")
        with open(comprehensive_results_file, 'w', encoding='utf-8') as f:
            json.dump(all_results, f, indent=2, ensure_ascii=False)
        
        print(f"Evaluation completed. Results saved to {output_dir}")
        return all_results
    
    def compare_models(self, evaluation_results: Dict[str, Dict]) -> Dict[str, Any]:
        comparison = {
            'model_count': len(evaluation_results),
            'rankings': {},
            'best_models': {},
            'metrics_comparison': {}
        }
        
        all_metrics = set()
        for model_results in evaluation_results.values():
            for task_type, metrics in model_results.items():
                for metric_name in metrics.keys():
                    if metric_name != 'count':
                        all_metrics.add(f"{task_type}_{metric_name}")
        
        for metric in all_metrics:
            task_type, metric_name = metric.split('_', 1)
            
            model_scores = []
            for model_name, model_results in evaluation_results.items():
                if (task_type in model_results and 
                    metric_name in model_results[task_type] and
                    model_results[task_type][metric_name] is not None):
                    score = model_results[task_type][metric_name]
                    model_scores.append((model_name, score))
            
            if model_scores:
                model_scores.sort(key=lambda x: x[1], reverse=True)
                
                comparison['rankings'][metric] = [
                    {'model': name, 'score': score} 
                    for name, score in model_scores
                ]
                comparison['best_models'][metric] = model_scores[0][0]
        
        for metric in all_metrics:
            task_type, metric_name = metric.split('_', 1)
            comparison['metrics_comparison'][metric] = {}
            
            for model_name, model_results in evaluation_results.items():
                if (task_type in model_results and 
                    metric_name in model_results[task_type]):
                    score = model_results[task_type][metric_name]
                    comparison['metrics_comparison'][metric][model_name] = score
        
        return comparison
    
    def generate_evaluation_report(self, ground_truth_file: str, 
                                 results_dir: str, 
                                 output_dir: str,
                                 batch_size: int = None) -> str:
        
        print("Starting comprehensive evaluation...")
        
        all_results = self.evaluate_multiple_models(
            ground_truth_file, results_dir, output_dir, 
            batch_size, save_detailed_results=True
        )
        
        comparison = self.compare_models(all_results)
        
        comparison_file = os.path.join(output_dir, "model_comparison.json")
        with open(comparison_file, 'w', encoding='utf-8') as f:
            json.dump(comparison, f, indent=2, ensure_ascii=False)
        
        report_file = os.path.join(output_dir, "evaluation_report.txt")
        with open(report_file, 'w', encoding='utf-8') as f:
            f.write("=" * 60 + "\n")
            f.write("MODEL EVALUATION REPORT\n")
            f.write("=" * 60 + "\n\n")
            
            f.write(f"Number of models evaluated: {comparison['model_count']}\n\n")
            
            f.write("BEST PERFORMING MODELS BY METRIC:\n")
            f.write("-" * 40 + "\n")
            for metric, model in comparison['best_models'].items():
                f.write(f"{metric}: {model}\n")
            
            f.write("\n")
            
            f.write("DETAILED RANKINGS:\n")
            f.write("-" * 40 + "\n")
            for metric, rankings in comparison['rankings'].items():
                f.write(f"\n{metric}:\n")
                for i, entry in enumerate(rankings, 1):
                    f.write(f"  {i}. {entry['model']}: {entry['score']:.4f}\n")
        
        print(f"Comprehensive evaluation report generated: {report_file}")
        return report_file


def main():
    config = Config()
    config.create_dirs()
    
    evaluator = ScholarBenchmarkEvaluator(config, language="ko") 
    
    report_path = evaluator.generate_evaluation_report(
        ground_truth_file=config.GROUND_TRUTH_FILE,
        results_dir=config.RESULTS_DIR,
        output_dir=config.OUTPUT_DIR,
        batch_size=config.DEFAULT_BATCH_SIZE
    )
    
    print(f"Evaluation completed. Report saved to: {report_path}")

if __name__ == "__main__":
    main()