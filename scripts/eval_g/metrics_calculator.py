import torch
import numpy as np
from typing import List, Tuple, Dict, Any, Optional
from rouge import Rouge
from bert_score import BERTScorer
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from nltk.tokenize import word_tokenize
from evaluate import load
from tqdm import tqdm

try:
    from bleurt import score as bleurt_score
    BLEURT_AVAILABLE = True
except ImportError:
    BLEURT_AVAILABLE = False
    print("Warning: BLEURT not available. Install bleurt-pytorch for BLEURT scores.")

try:
    from konlpy.tag import Komoran
    KONLPY_AVAILABLE = True
    komoran = Komoran()
except ImportError:
    KONLPY_AVAILABLE = False
    print("Warning: KoNLPy not available. Korean tokenization will use basic tokenization.")

class MetricsCalculator:
    def __init__(self, device: str = "cuda", bert_model: str = "xlm-roberta-base",
                 bleurt_checkpoint: Optional[str] = None, language: str = "en"):
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.language = language
        
        self.bert_scorer = BERTScorer(
            rescale_with_baseline=False, 
            model_type=bert_model, 
            device=self.device
        )
        
        self.squad_metric = load("squad")
        
        if BLEURT_AVAILABLE and bleurt_checkpoint:
            try:
                self.bleurt_scorer = bleurt_score.BleurtScorer(bleurt_checkpoint)
            except Exception as e:
                print(f"Warning: Could not load BLEURT scorer: {e}")
                self.bleurt_scorer = None
        else:
            self.bleurt_scorer = None
        
        self.smoothing_function = SmoothingFunction().method1
        
        print(f"Metrics calculator initialized on device: {self.device}")
    
    def tokenize_text(self, text: str) -> List[str]:
        """Tokenize text based on language"""
        if self.language == "ko" and KONLPY_AVAILABLE:
            return komoran.morphs(text)
        else:
            return word_tokenize(text)
    
    def calculate_exact_match_f1(self, references: List[str], 
                                hypotheses: List[str]) -> Tuple[List[float], List[float]]:
        """Calculate exact match and F1 scores"""
        exact_matches = []
        f1_scores = []

        for i, (ref, hyp) in enumerate(zip(references, hypotheses)):
            ref = ref.lower()
            hyp = hyp.lower()
            
            pred = [{"id": str(i), "prediction_text": hyp}]
            ref_formatted = [{
                "id": str(i),
                "answers": {
                    "text": [ref],
                    "answer_start": [0],
                }
            }]
            
            try:
                result = self.squad_metric.compute(predictions=pred, references=ref_formatted)
                exact_matches.append(result["exact_match"])
                f1_scores.append(result["f1"])
            except Exception as e:
                print(f"Error calculating exact match/F1 for item {i}: {e}")
                exact_matches.append(0.0)
                f1_scores.append(0.0)

        return exact_matches, f1_scores
    
    def calculate_bert_score(self, references: List[str], 
                           hypotheses: List[str]) -> List[float]:
        """Calculate BERTScore F1"""
        try:
            with torch.no_grad():
                P, R, F1 = self.bert_scorer.score(hypotheses, references)
                return F1.cpu().numpy().tolist()
        except Exception as e:
            print(f"Error calculating BERTScore: {e}")
            return [0.0] * len(references)
    
    def calculate_bleurt_score(self, references: List[str], 
                              hypotheses: List[str]) -> List[float]:
        """Calculate BLEURT scores"""
        if not self.bleurt_scorer:
            print("BLEURT scorer not available")
            return [0.0] * len(references)
        
        try:
            return self.bleurt_scorer.score(references=references, candidates=hypotheses)
        except Exception as e:
            print(f"Error calculating BLEURT: {e}")
            return [0.0] * len(references)
    
    def calculate_rouge_score(self, reference: str, hypothesis: str) -> Dict[str, float]:
        """Calculate ROUGE scores for a single pair"""
        if not reference or not hypothesis:
            return {}
        
        try:
            if self.language == "ko" and KONLPY_AVAILABLE:
                reference = ' '.join(self.tokenize_text(reference))
                hypothesis = ' '.join(self.tokenize_text(hypothesis))
            else:
                reference = ' '.join(word_tokenize(reference))
                hypothesis = ' '.join(word_tokenize(hypothesis))
            
            rouge = Rouge()
            scores = rouge.get_scores(hypothesis, reference)[0]
            return {
                'rouge-1': scores['rouge-1']['f'],
                'rouge-2': scores['rouge-2']['f'],
                'rouge-l': scores['rouge-l']['f']
            }
        except Exception as e:
            print(f"ROUGE calculation error: {e}")
            return {}
    
    def calculate_bleu1_score(self, reference: str, hypothesis: str) -> float:
        """Calculate BLEU-1 score for a single pair"""
        try:
            ref_tokens = self.tokenize_text(reference)
            hyp_tokens = self.tokenize_text(hypothesis)
            
            return sentence_bleu(
                [ref_tokens], hyp_tokens, 
                weights=(1, 0, 0, 0), 
                smoothing_function=self.smoothing_function
            )
        except Exception as e:
            print(f"BLEU-1 calculation error: {e}")
            return 0.0


class BatchProcessor:
    def __init__(self, metrics_calculator: MetricsCalculator):
        self.calc = metrics_calculator
    
    def process_short_answers(self, data_pairs: List[Tuple[str, str]], 
                            batch_size: int = 16) -> List[Dict[str, Any]]:
        """Process short answer evaluation in batches"""
        all_metrics = []
        num_batches = (len(data_pairs) + batch_size - 1) // batch_size
        
        for batch_idx in tqdm(range(num_batches), desc="Processing short answers"):
            start_idx = batch_idx * batch_size
            end_idx = min((batch_idx + 1) * batch_size, len(data_pairs))
            
            batch_data = data_pairs[start_idx:end_idx]
            references = [pair[0] for pair in batch_data]
            hypotheses = [pair[1] for pair in batch_data]
            
            exact_matches, f1_scores = self.calc.calculate_exact_match_f1(references, hypotheses)
            bert_scores = self.calc.calculate_bert_score(references, hypotheses)
            bleurt_scores = self.calc.calculate_bleurt_score(references, hypotheses)
            
            for i, (ref, hyp) in enumerate(batch_data):
                rouge_scores = self.calc.calculate_rouge_score(ref, hyp)
                bleu_score = self.calc.calculate_bleu1_score(ref, hyp)
                
                metrics = {
                    'exact_match': exact_matches[i] if i < len(exact_matches) else 0.0,
                    'f1': f1_scores[i] if i < len(f1_scores) else 0.0,
                    'bert_score_f1': bert_scores[i] if i < len(bert_scores) else 0.0,
                    'bluert_score': bleurt_scores[i] if i < len(bleurt_scores) else 0.0,
                    'rouge-1': rouge_scores.get('rouge-1', 0.0),
                    'bleu-1': bleu_score,
                    'global_idx': start_idx + i,
                    'batch_idx': batch_idx,
                    'item_idx': i
                }
                
                all_metrics.append(metrics)
        
        return all_metrics
    
    def process_summarization(self, data_pairs: List[Tuple[str, str]], 
                            batch_size: int = 16) -> List[Dict[str, Any]]:
        """Process summarization evaluation in batches"""
        all_metrics = []
        num_batches = (len(data_pairs) + batch_size - 1) // batch_size
        
        for batch_idx in tqdm(range(num_batches), desc="Processing summarization"):
            start_idx = batch_idx * batch_size
            end_idx = min((batch_idx + 1) * batch_size, len(data_pairs))
            
            batch_data = data_pairs[start_idx:end_idx]
            references = [pair[0] for pair in batch_data]
            hypotheses = [pair[1] for pair in batch_data]
            
            bert_scores = self.calc.calculate_bert_score(references, hypotheses)
            
            for i, (ref, hyp) in enumerate(batch_data):
                rouge_scores = self.calc.calculate_rouge_score(ref, hyp)
                
                metrics = {
                    'rouge-1': rouge_scores.get('rouge-1', 0.0),
                    'rouge-2': rouge_scores.get('rouge-2', 0.0),
                    'rouge-l': rouge_scores.get('rouge-l', 0.0),
                    'bert_score_f1': bert_scores[i] if i < len(bert_scores) else 0.0,
                    'global_idx': start_idx + i,
                    'batch_idx': batch_idx,
                    'item_idx': i
                }
                
                all_metrics.append(metrics)
        
        return all_metrics