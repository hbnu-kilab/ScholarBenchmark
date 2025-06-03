import argparse
import sys
import os
import json
from pathlib import Path

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from config import Config
from main import ScholarBenchmarkEvaluator


def setup_parser():
    """Set up command line argument parser"""
    parser = argparse.ArgumentParser(
        description="Scholar Benchmark Evaluation Tool",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
  # single model eval
  python cli.py single --ground-truth data/ground_truth.jsonl --model-result data/model_result.jsonl --output results/

  # multi model eval
  python cli.py batch --ground-truth data/ground_truth.jsonl --results-dir data/models/ --output results/

  # total report
  python cli.py report --ground-truth data/ground_truth.jsonl --results-dir data/models/ --output results/
        """
    )
    
    parser.add_argument(
        "--language", 
        choices=["ko", "en"], 
        default="ko",
        help="Language for evaluation (default: ko)"
    )
    
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable verbose output"
    )
    
    subparsers = parser.add_subparsers(dest="command", help="Available commands")
    
    single_parser = subparsers.add_parser(
        "single", 
        help="Evaluate a single model"
    )
    single_parser.add_argument(
        "--ground-truth", "-g",
        required=True,
        type=str,
        help="Path to ground truth JSONL file"
    )
    single_parser.add_argument(
        "--model-result", "-m",
        required=True,
        type=str,
        help="Path to model result JSONL file"
    )
    single_parser.add_argument(
        "--output", "-o",
        required=True,
        type=str,
        help="Output directory for results"
    )
    single_parser.add_argument(
        "--detailed",
        action="store_true",
        help="Save detailed results including category breakdowns"
    )
    

    batch_parser = subparsers.add_parser(
        "batch", 
        help="Evaluate multiple models"
    )
    batch_parser.add_argument(
        "--ground-truth", "-g",
        required=True,
        type=str,
        help="Path to ground truth JSONL file"
    )
    batch_parser.add_argument(
        "--results-dir", "-r",
        required=True,
        type=str,
        help="Directory containing model result JSONL files"
    )
    batch_parser.add_argument(
        "--output", "-o",
        required=True,
        type=str,
        help="Output directory for results"
    )
    batch_parser.add_argument(
        "--detailed",
        action="store_true",
        help="Save detailed results including category breakdowns"
    )
    

    report_parser = subparsers.add_parser(
        "report", 
        help="Generate comprehensive evaluation report"
    )
    report_parser.add_argument(
        "--ground-truth", "-g",
        required=True,
        type=str,
        help="Path to ground truth JSONL file"
    )
    report_parser.add_argument(
        "--results-dir", "-r",
        required=True,
        type=str,
        help="Directory containing model result JSONL files"
    )
    report_parser.add_argument(
        "--output", "-o",
        required=True,
        type=str,
        help="Output directory for results"
    )
    

    config_parser = subparsers.add_parser(
        "config",
        help="Show current configuration"
    )
    
    return parser


def validate_paths(args):
    if hasattr(args, 'ground_truth'):
        if not os.path.exists(args.ground_truth):
            print(f"Ground truth file not found: {args.ground_truth}")
            return False
            
    if hasattr(args, 'model_result'):
        if not os.path.exists(args.model_result):
            print(f"Model result file not found: {args.model_result}")
            return False
            
    if hasattr(args, 'results_dir'):
        if not os.path.exists(args.results_dir):
            print(f"Results directory not found: {args.results_dir}")
            return False
        
        jsonl_files = list(Path(args.results_dir).glob("*.jsonl"))
        if not jsonl_files:
            print(f"No .jsonl files found in results directory: {args.results_dir}")
            return False
        
        if args.verbose:
            print(f"Found {len(jsonl_files)} .jsonl files in results directory")
    
    if hasattr(args, 'output'):
        os.makedirs(args.output, exist_ok=True)
        if args.verbose:
            print(f"Output directory ready: {args.output}")
    
    return True


def handle_single_evaluation(args):
    print("Starting single model evaluation...")
    
    if not validate_paths(args):
        return False
    
    evaluator = ScholarBenchmarkEvaluator(language=args.language)
    
    try:
        summary, evaluation_results, category_summary, merged_category_results = \
            evaluator.evaluate_single_model(args.ground_truth, args.model_result)
        
        if summary is None:
            print("Evaluation failed")
            return False
        
        model_name = Path(args.model_result).stem
        output_file = Path(args.output) / f"evaluation_{model_name}.json"
        
        results_to_save = {'summary': summary}
        if args.detailed:
            results_to_save.update({
                'category_summary': category_summary,
                'evaluation_results': evaluation_results,
                'merged_category_results': merged_category_results
            })
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(results_to_save, f, indent=2, ensure_ascii=False)
        
        print(f"Evaluation completed successfully!")
        print(f"Results saved to: {output_file}")
        
        print("\nEVALUATION SUMMARY:")
        print("-" * 40)
        for task_type, metrics in summary.items():
            print(f"\n{task_type.upper()}:")
            for metric_name, value in metrics.items():
                if isinstance(value, (int, float)):
                    if isinstance(value, float):
                        print(f"  {metric_name}: {value:.4f}")
                    else:
                        print(f"  {metric_name}: {value}")
        
        return True
        
    except Exception as e:
        print(f"Error during evaluation: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        return False


def handle_batch_evaluation(args):
    print("Starting batch model evaluation...")
    
    if not validate_paths(args):
        return False
    
    evaluator = ScholarBenchmarkEvaluator(language=args.language)
    
    try:
        all_results = evaluator.evaluate_multiple_models(
            args.ground_truth,
            args.results_dir,
            args.output,
            save_detailed_results=args.detailed
        )
        
        if not all_results:
            print("No models were successfully evaluated")
            return False
        
        print(f"Batch evaluation completed successfully!")
        print(f"Evaluated {len(all_results)} models")
        print(f"Results saved to: {args.output}")
        
        return True
        
    except Exception as e:
        print(f"Error during batch evaluation: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        return False


def handle_report_generation(args):
    print("Starting comprehensive report generation...")
    
    if not validate_paths(args):
        return False
    
    evaluator = ScholarBenchmarkEvaluator(language=args.language)
    
    try:
        report_path = evaluator.generate_evaluation_report(
            args.ground_truth,
            args.results_dir,
            args.output
        )
        
        print(f"Comprehensive report generated successfully!")
        print(f"Report saved to: {report_path}")
        
        return True
        
    except Exception as e:
        print(f"Error during report generation: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        return False


def handle_config_display():
    config = Config()
    
    print("CURRENT CONFIGURATION:")
    print("=" * 50)
    print(f"Ground Truth File: {config.GROUND_TRUTH_FILE}")
    print(f"Results Directory: {config.RESULTS_DIR}")
    print(f"Output Directory: {config.OUTPUT_DIR}")
    print(f"Device: {config.DEVICE}")
    print(f"Default Batch Size: {config.DEFAULT_BATCH_SIZE}")
    print(f"Supported Languages: {', '.join(config.SUPPORTED_LANGUAGES)}")
    print("\nMetrics:")
    print(f"  Multiple Choice: {', '.join(config.MULTIPLE_CHOICE_METRICS)}")
    print(f"  Multiple Select: {', '.join(config.MULTIPLE_SELECT_METRICS)}")
    print(f"  True/False: {', '.join(config.TRUE_FALSE_METRICS)}")
    print("=" * 50)


def main():
    parser = setup_parser()
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return
    
    print("Scholar Benchmark Evaluation Tool")
    print("=" * 50)
    
    success = False
    
    if args.command == "single":
        success = handle_single_evaluation(args)
    elif args.command == "batch":
        success = handle_batch_evaluation(args)
    elif args.command == "report":
        success = handle_report_generation(args)
    elif args.command == "config":
        handle_config_display()
        success = True
    else:
        print(f"Unknown command: {args.command}")
        parser.print_help()
    
    if success:
        print("success")
    else:
        print("error")
        sys.exit(1)


if __name__ == "__main__":
    main()
