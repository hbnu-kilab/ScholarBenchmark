import os

class Config:
    DEVICE = "cuda"
    DEFAULT_BATCH_SIZE = 16
    
    SUPPORTED_LANGUAGES = ["en", "ko"]

    SHORT_ANSWER_METRICS = [
        'exact_match', 'f1', 'bert_score_f1', 
        'bluert_score', 'rouge-1', 'bleu-1'
    ]

    SUMMARIZATION_METRICS = [
        'rouge-1', 'rouge-2', 'rouge-l', 'bert_score_f1'
    ]

    bert_model = "xlm-roberta-base"  
    BLEURT_CHECKPOINT_PATH = None

    @classmethod
    def create_dirs(cls):
        os.makedirs(cls.OUTPUT_DIR, exist_ok=True)
        os.makedirs(os.path.dirname(cls.GROUND_TRUTH_FILE), exist_ok=True)
