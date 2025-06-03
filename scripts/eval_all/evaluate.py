import argparse
import sys
import os
import json
from pathlib import Path
from typing import Dict, List, Tuple, Any

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from config import Config
from main import ScholarBenchmarkEvaluator

def setup_parser():
    parser = argparse.ArgumentParser(
        description="Scholar Benchmark Evaluation Tool",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
  # single model evaluation
  python cli.py single --ground-truth data/ground_truth.jsonl --model-result data/model_result.jsonl --output results/

  # batch model evaluation
  python cli.py batch --ground-truth data/ground_truth.jsonl --results-dir data/models/ --output results/

  # generate comprehensive report
  python cli.py report --ground-truth data/ground_truth.jsonl --results-dir data/models/ --output results/

    """
    )
    
    parser.add_argument(
        "--language", 
        choices=["ko", "en"], 
        default="en",
        help="(default: en)"
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
        type=str,
        help="Path to ground truth JSONL file (overrides config/env variable)"
    )
    single_parser.add_argument(
        "--model-result", "-m",
        required=True,
        type=str,
        help="Path to model result JSONL file"
    )
    single_parser.add_argument(
        "--output", "-o",
        type=str,
        help="Output directory for results (overrides config/env variable)"
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
        type=str,
        help="Path to ground truth JSONL file (overrides config/env variable)"
    )
    batch_parser.add_argument(
        "--results-dir", "-r",
        type=str,
        help="Directory containing model result JSONL files (overrides config/env variable)"
    )
    batch_parser.add_argument(
        "--output", "-o",
        type=str,
        help="Output directory for results (overrides config/env variable)"
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
        type=str,
        help="Path to ground truth JSONL file (overrides config/env variable)"
    )
    report_parser.add_argument(
        "--results-dir", "-r",
        type=str,
        help="Directory containing model result JSONL files (overrides config/env variable)"
    )
    report_parser.add_argument(
        "--output", "-o",
        type=str,
        help="Output directory for results (overrides config/env variable)"
    )
    
    config_parser = subparsers.add_parser(
        "config",
        help="Show current configuration"
    )
    
    validate_parser = subparsers.add_parser(
        "validate",
        help="Validate configuration and data files"
    )
    validate_parser.add_argument(
        "--ground-truth", "-g",
        type=str,
        help="Path to ground truth JSONL file to validate"
    )
    validate_parser.add_argument(
        "--results-dir", "-r",
        type=str,
        help="Directory containing model result JSONL files to validate"
    )
    
    return parser


def get_effective_paths(args):
    config = Config()
    
    ground_truth = getattr(args, 'ground_truth', None) or config.GROUND_TRUTH_FILE
    results_dir = getattr(args, 'results_dir', None) or config.RESULTS_DIR
    output_dir = getattr(args, 'output', None) or config.OUTPUT_DIR
    
    return ground_truth, results_dir, output_dir


def validate_paths(ground_truth_file, results_dir=None, output_dir=None, model_result_file=None, verbose=False):
    if ground_truth_file and not os.path.exists(ground_truth_file):
        print(f"Ground truth file not found: {ground_truth_file}")
        return False
            
    if model_result_file and not os.path.exists(model_result_file):
        print(f"Model result file not found: {model_result_file}")
        return False
            
    if results_dir:
        if not os.path.exists(results_dir):
            print(f"Results directory not found: {results_dir}")
            return False
        
        jsonl_files = list(Path(results_dir).glob("*.jsonl"))
        if not jsonl_files:
            print(f"No .jsonl files found in results directory: {results_dir}")
            return False
        
        if verbose:
            print(f"Found {len(jsonl_files)} .jsonl files in results directory")
    
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        if verbose:
            print(f"Output directory ready: {output_dir}")
    
    return True


def handle_single_evaluation(args):
    print("Starting single model evaluation...")
    
    ground_truth_file, _, output_dir = get_effective_paths(args)
    
    if not ground_truth_file:
        print("Ground truth file must be specified via --ground-truth argument or configuration")
        return False
    
    if not output_dir:
        print("Output directory must be specified via --output argument or configuration")
        return False
    
    if not validate_paths(ground_truth_file, output_dir=output_dir, 
                         model_result_file=args.model_result, verbose=args.verbose):
        return False
    
    evaluator = ScholarBenchmarkEvaluator(language=args.language)
    
    try:
        summary, evaluation_results, category_summary, merged_category_results = \
            evaluator.evaluate_single_model(ground_truth_file, args.model_result)
        
        if summary is None:
            print("Evaluation failed")
            return False
        
        model_name = Path(args.model_result).stem
        output_file = Path(output_dir) / f"evaluation_{model_name}.json"
        
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
    
    ground_truth_file, results_dir, output_dir = get_effective_paths(args)
    
    if not ground_truth_file:
        print("Ground truth file must be specified via --ground-truth argument or configuration")
        return False
    
    if not results_dir:
        print("Results directory must be specified via --results-dir argument or configuration")
        return False
    
    if not output_dir:
        print("Output directory must be specified via --output argument or configuration") 
        return False
    
    if not validate_paths(ground_truth_file, results_dir=results_dir, 
                         output_dir=output_dir, verbose=args.verbose):
        return False
    
    evaluator = ScholarBenchmarkEvaluator(language=args.language)
    
    try:
        all_results = evaluator.evaluate_multiple_models(
            ground_truth_file,
            results_dir,
            output_dir,
            save_detailed_results=args.detailed
        )
        
        if not all_results:
            print("No models were successfully evaluated")
            return False
        
        print(f"Batch evaluation completed successfully!")
        print(f"Evaluated {len(all_results)} models")
        print(f"Results saved to: {output_dir}")
        
        return True
        
    except Exception as e:
        print(f"Error during batch evaluation: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        return False

def handle_report_generation(args):
    print("Starting comprehensive report generation...")
    
    ground_truth_file, results_dir, output_dir = get_effective_paths(args)
    
    if not ground_truth_file:
        print("Ground truth file must be specified via --ground-truth argument or configuration")
        return False
    
    if not results_dir:
        print("Results directory must be specified via --results-dir argument or configuration")
        return False
    
    if not output_dir:
        print("Output directory must be specified via --output argument or configuration")
        return False
    
    if not validate_paths(ground_truth_file, results_dir=results_dir, 
                         output_dir=output_dir, verbose=args.verbose):
        return False
    
    evaluator = ScholarBenchmarkEvaluator(language=args.language)
    
    try:
        report_path = evaluator.generate_evaluation_report(
            ground_truth_file,
            results_dir,
            output_dir
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

def handle_config_display(args=None):
    config = Config()
    
    print("CURRENT CONFIGURATION:")
    print("=" * 50)
    
    try:
        if hasattr(config, 'get_all_settings'):
            settings = config.get_all_settings()
            for key, value in settings.items():
                if isinstance(value, list):
                    print(f"{key}: {', '.join(map(str, value))}")
                else:
                    print(f"{key}: {value}")
        else:
            print(f"Ground Truth File: {getattr(config, 'GROUND_TRUTH_FILE', 'Not set')}")
            print(f"Results Directory: {getattr(config, 'RESULTS_DIR', 'Not set')}")
            print(f"Output Directory: {getattr(config, 'OUTPUT_DIR', 'Not set')}")
            print(f"Device: {getattr(config, 'DEVICE', 'Not set')}")
            print(f"Default Batch Size: {getattr(config, 'DEFAULT_BATCH_SIZE', 'Not set')}")
            print(f"Supported Languages: {', '.join(getattr(config, 'SUPPORTED_LANGUAGES', []))}")
            
            print("\nMetrics:")
            print(f"  Multiple Choice: {', '.join(getattr(config, 'MULTIPLE_CHOICE_METRICS', []))}")
            print(f"  Multiple Select: {', '.join(getattr(config, 'MULTIPLE_SELECT_METRICS', []))}")
            print(f"  True/False: {', '.join(getattr(config, 'TRUE_FALSE_METRICS', []))}")
    except Exception as e:
        print(f"Error loading configuration: {e}")
    
    print("\nConfiguration Validation:")
    try:
        if hasattr(config, 'validate_config'):
            issues = config.validate_config()
            if issues:
                print("Issues found:")
                for issue in issues:
                    print(f"  - {issue}")
            else:
                print("Configuration is valid")
        else:
            print("Configuration validation not available")
    except Exception as e:
        print(f"Error validating configuration: {e}")
    
    print("=" * 50)


def handle_validation(args):
    print("Validating configuration and data files...")
    
    ground_truth_file, results_dir, _ = get_effective_paths(args)
    
    config = Config()
    try:
        if hasattr(config, 'validate_config'):
            config_issues = config.validate_config()
            if config_issues:
                print("Configuration issues:")
                for issue in config_issues:
                    print(f"  - {issue}")
            else:
                print("Configuration is valid")
        else:
            print("Configuration validation method not available")
    except Exception as e:
        print(f"Configuration validation error: {e}")
    
    data_issues = []
    
    if ground_truth_file:
        try:
            try:
                from data_loader import DataLoader
                gt_data = DataLoader.load_jsonl_data(ground_truth_file)
                print(f"Ground truth file loaded: {len(gt_data)} entries")
            except ImportError:
                import json
                with open(ground_truth_file, 'r', encoding='utf-8') as f:
                    gt_data = [json.loads(line) for line in f if line.strip()]
                print(f"Ground truth file loaded: {len(gt_data)} entries")
        except Exception as e:
            data_issues.append(f"Ground truth file error: {e}")
    
    if results_dir and os.path.exists(results_dir):
        jsonl_files = list(Path(results_dir).glob("*.jsonl"))
        valid_files = 0
        for file_path in jsonl_files:
            try:
                try:
                    from data_loader import DataLoader
                    model_data = DataLoader.load_jsonl_data(str(file_path))
                except ImportError:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        model_data = [json.loads(line) for line in f if line.strip()]
                valid_files += 1
                if args.verbose:
                    print(f"{file_path.name}: {len(model_data)} entries")
            except Exception as e:
                data_issues.append(f"Model file {file_path.name} error: {e}")
        
        print(f"Valid model result files: {valid_files}/{len(jsonl_files)}")
    
    if data_issues:
        print("Data validation issues:")
        for issue in data_issues:
            print(f"  - {issue}")
        return False
    else:
        print("All data files are valid")
        return True


def main():
    parser = setup_parser()
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return
    
    print("Scholar Benchmark Evaluation Tool")
    print("=" * 50)
    
    success = False
    
    try:
        if args.command == "single":
            success = handle_single_evaluation(args)
        elif args.command == "batch":
            success = handle_batch_evaluation(args)
        elif args.command == "report":
            success = handle_report_generation(args)
        elif args.command == "config":
            handle_config_display(args)
            success = True
        elif args.command == "validate":
            success = handle_validation(args)
        else:
            print(f"Unknown command: {args.command}")
            parser.print_help()
    
    except KeyboardInterrupt:
        sys.exit(1)
    except Exception as e:
        print(f"\n{e}")
        if hasattr(args, 'verbose') and args.verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)
    
    if success:
        print("")
    else:
        print("\nerror occurred during the process")
        sys.exit(1)


if __name__ == "__main__":
    main()