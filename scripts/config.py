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

    SHORT_ANSWER_METRICS = [
        'exact_match', 'f1', 'bert_score_f1', 
        'bluert_score', 'rouge-1', 'bleu-1'
    ]

    SUMMARIZATION_METRICS = [
        'rouge-1', 'rouge-2', 'rouge-l', 'bert_score_f1'
    ]

    MULTIPLE_CHOICE_METRICS = ['accuracy']

    MULTIPLE_SELECT_METRICS = [
        'a@1', 'a@2', 'a@3', 'a@4', 
        'exact_match_1', 'exact_match_2', 'exact_match_3', 'exact_match_4'
    ]
    
    TRUE_FALSE_METRICS = ['accuracy']

    bert_model = "xlm-roberta-base"  
    BLEURT_CHECKPOINT_PATH = None  

    @classmethod
    def create_dirs(cls):
        os.makedirs(cls.OUTPUT_DIR, exist_ok=True)
        os.makedirs(os.path.dirname(cls.GROUND_TRUTH_FILE), exist_ok=True)