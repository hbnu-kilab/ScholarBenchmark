import os
from pathlib import Path


class Config:
    DEVICE = "cuda" 
    
    DEFAULT_BATCH_SIZE = 16
    
    GROUND_TRUTH_FILE = os.getenv(
        'SCHOLAR_GROUND_TRUTH_FILE',
        str(Path(__file__).parent / "data" / "ko_eval_dataset.jsonl")
    )
    
    RESULTS_DIR = os.getenv(
        'SCHOLAR_RESULTS_DIR', 
        str(Path(__file__).parent / "model_results")
    )
    
    OUTPUT_DIR = os.getenv(
        'SCHOLAR_OUTPUT_DIR',
        str(Path(__file__).parent / "evaluation_results")
    )
    
    SUPPORTED_LANGUAGES = ["en", "ko"]
    
    MULTIPLE_CHOICE_METRICS = ['accuracy']
    MULTIPLE_SELECT_METRICS = [
        'a@1', 'a@2', 'a@3', 'a@4', 
        'exact_match_1', 'exact_match_2', 'exact_match_3', 'exact_match_4'
    ]
    TRUE_FALSE_METRICS = ['accuracy']
    
    LOG_LEVEL = os.getenv('SCHOLAR_LOG_LEVEL', 'INFO')
    VERBOSE = os.getenv('SCHOLAR_VERBOSE', 'False').lower() == 'true'
    
    @classmethod
    def create_dirs(cls):
        dirs_to_create = [
            cls.OUTPUT_DIR,
            os.path.dirname(cls.GROUND_TRUTH_FILE),
            cls.RESULTS_DIR
        ]
        
        for dir_path in dirs_to_create:
            if dir_path: 
                os.makedirs(dir_path, exist_ok=True)
    
    @classmethod
    def validate_config(cls):
        issues = []
        
        if not os.path.exists(cls.GROUND_TRUTH_FILE):
            issues.append(f"Ground truth file not found: {cls.GROUND_TRUTH_FILE}")
        
        if not os.path.exists(cls.RESULTS_DIR):
            issues.append(f"Results directory not found: {cls.RESULTS_DIR}")
        else:
            jsonl_files = list(Path(cls.RESULTS_DIR).glob("*.jsonl"))
            if not jsonl_files:
                issues.append(f"No .jsonl files found in results directory: {cls.RESULTS_DIR}")
        
        if cls.LOG_LEVEL not in ['DEBUG', 'INFO', 'WARNING', 'ERROR']:
            issues.append(f"Invalid log level: {cls.LOG_LEVEL}")
        
        return issues
    
    
    @classmethod
    def get_all_settings(cls):
        return {
            'DEVICE': cls.DEVICE,
            'DEFAULT_BATCH_SIZE': cls.DEFAULT_BATCH_SIZE,
            'GROUND_TRUTH_FILE': cls.GROUND_TRUTH_FILE,
            'RESULTS_DIR': cls.RESULTS_DIR,
            'OUTPUT_DIR': cls.OUTPUT_DIR,
            'SUPPORTED_LANGUAGES': cls.SUPPORTED_LANGUAGES,
            'MULTIPLE_CHOICE_METRICS': cls.MULTIPLE_CHOICE_METRICS,
            'MULTIPLE_SELECT_METRICS': cls.MULTIPLE_SELECT_METRICS,
            'TRUE_FALSE_METRICS': cls.TRUE_FALSE_METRICS,
            'LOG_LEVEL': cls.LOG_LEVEL,
            'VERBOSE': cls.VERBOSE
        }
    
    @classmethod
    def set_paths(cls, ground_truth_file=None, results_dir=None, output_dir=None):
        if ground_truth_file:
            cls.GROUND_TRUTH_FILE = ground_truth_file
        if results_dir:
            cls.RESULTS_DIR = results_dir
        if output_dir:
            cls.OUTPUT_DIR = output_dir
