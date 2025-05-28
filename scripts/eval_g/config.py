import os

class Config:
    DEVICE = "cuda"
    DEFAULT_BATCH_SIZE = 16
    
    GROUND_TRUTH_FILE = "/home/kilab_kdh/bench_backup/scholarBenchmark/eval_scholarBench/original_data/ko_eval_dataset.jsonl"
    RESULTS_DIR = "/home/kilab_kdh/bench_backup/scholarBenchmark/eval_scholarBench/data_ko/jsonl"
    OUTPUT_DIR = "/home/kilab_kdh/bench_backup/scholarBenchmark/eval_scholarBench/code/test_for_github/result"

    SUPPORTED_LANGUAGES = ["en", "ko"]

    SHORT_ANSWER_METRICS = [
        'exact_match', 'f1', 'bert_score_f1', 
        'bluert_score', 'rouge-1', 'bleu-1'
    ]

    SUMMARIZATION_METRICS = [
        'rouge-1', 'rouge-2', 'rouge-l', 'bert_score_f1'
    ]

    bert_model = "xlm-roberta-base"  
    BLEURT_CHECKPOINT_PATH = None  # Set to None or specify a valid path to BLEURT checkpoint

    @classmethod
    def create_dirs(cls):
        os.makedirs(cls.OUTPUT_DIR, exist_ok=True)
        os.makedirs(os.path.dirname(cls.GROUND_TRUTH_FILE), exist_ok=True)