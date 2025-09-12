"""
Evaluation metrics for Russian-Tatar translation model.
"""

import numpy as np
from typing import List, Dict, Tuple
import logging
from evaluate import load
from sacrebleu import corpus_bleu, sentence_bleu
from rouge_score import rouge_scorer
import torch
from transformers import AutoTokenizer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TranslationEvaluator:
    """
    Comprehensive evaluation of translation quality using multiple metrics.
    """
    
    def __init__(self):
        """Initialize the evaluator with various metrics."""
        self.metrics = {}
        self._load_metrics()
    
    def _load_metrics(self):
        """Load evaluation metrics."""
        try:
            # BLEU metric
            self.metrics['bleu'] = load('bleu')
            logger.info("BLEU metric loaded")
        except Exception as e:
            logger.warning(f"Could not load BLEU metric: {e}")
        
        try:
            # ROUGE metric
            self.metrics['rouge'] = load('rouge')
            logger.info("ROUGE metric loaded")
        except Exception as e:
            logger.warning(f"Could not load ROUGE metric: {e}")
        
        # ROUGE scorer as backup
        self.rouge_scorer = rouge_scorer.RougeScorer(
            ['rouge1', 'rouge2', 'rougeL'], 
            use_stemmer=True
        )
    
    def compute_bleu(
        self, 
        predictions: List[str], 
        references: List[str],
        use_sacrebleu: bool = True
    ) -> Dict[str, float]:
        """
        Compute BLEU scores.
        
        Args:
            predictions: List of predicted translations
            references: List of reference translations
            use_sacrebleu: Whether to use SacreBLEU implementation
            
        Returns:
            Dictionary with BLEU scores
        """
        
        if use_sacrebleu:
            # Using SacreBLEU
            # Convert references to list of lists for corpus_bleu
            refs_formatted = [[ref] for ref in references]
            
            bleu_score = corpus_bleu(predictions, list(zip(*refs_formatted)))
            
            # Also compute sentence-level BLEU
            sentence_bleus = []
            for pred, ref in zip(predictions, references):
                sent_bleu = sentence_bleu(pred, [ref])
                sentence_bleus.append(sent_bleu.score)
            
            return {
                'bleu_corpus': bleu_score.score,
                'bleu_sentence_avg': np.mean(sentence_bleus),
                'bleu_sentence_std': np.std(sentence_bleus)
            }
        else:
            # Using HuggingFace evaluate
            if 'bleu' in self.metrics:
                result = self.metrics['bleu'].compute(
                    predictions=predictions,
                    references=[[ref] for ref in references]
                )
                return {
                    'bleu': result['bleu'],
                    'precisions': result['precisions']
                }
            else:
                logger.warning("BLEU metric not available")
                return {}
    
    def compute_rouge(
        self, 
        predictions: List[str], 
        references: List[str]
    ) -> Dict[str, float]:
        """
        Compute ROUGE scores.
        
        Args:
            predictions: List of predicted translations
            references: List of reference translations
            
        Returns:
            Dictionary with ROUGE scores
        """
        
        if 'rouge' in self.metrics:
            # Using HuggingFace evaluate
            result = self.metrics['rouge'].compute(
                predictions=predictions,
                references=references
            )
            return result
        else:
            # Using rouge_score as backup
            rouge_scores = {
                'rouge1_f': [],
                'rouge1_p': [],
                'rouge1_r': [],
                'rouge2_f': [],
                'rouge2_p': [],
                'rouge2_r': [],
                'rougeL_f': [],
                'rougeL_p': [],
                'rougeL_r': []
            }
            
            for pred, ref in zip(predictions, references):
                scores = self.rouge_scorer.score(ref, pred)
                
                for rouge_type in ['rouge1', 'rouge2', 'rougeL']:
                    rouge_scores[f'{rouge_type}_f'].append(scores[rouge_type].fmeasure)
                    rouge_scores[f'{rouge_type}_p'].append(scores[rouge_type].precision)
                    rouge_scores[f'{rouge_type}_r'].append(scores[rouge_type].recall)
            
            # Average the scores
            averaged_scores = {}
            for key, values in rouge_scores.items():
                averaged_scores[key] = np.mean(values)
            
            return averaged_scores
    
    def compute_length_ratio(
        self, 
        predictions: List[str], 
        references: List[str]
    ) -> Dict[str, float]:
        """
        Compute length-based metrics.
        
        Args:
            predictions: List of predicted translations
            references: List of reference translations
            
        Returns:
            Dictionary with length metrics
        """
        
        pred_lengths = [len(pred.split()) for pred in predictions]
        ref_lengths = [len(ref.split()) for ref in references]
        
        length_ratios = [p_len / r_len if r_len > 0 else 0 
                        for p_len, r_len in zip(pred_lengths, ref_lengths)]
        
        return {
            'avg_pred_length': np.mean(pred_lengths),
            'avg_ref_length': np.mean(ref_lengths),
            'length_ratio_avg': np.mean(length_ratios),
            'length_ratio_std': np.std(length_ratios),
            'length_ratio_median': np.median(length_ratios)
        }
    
    def compute_exact_match(
        self, 
        predictions: List[str], 
        references: List[str]
    ) -> Dict[str, float]:
        """
        Compute exact match metrics.
        
        Args:
            predictions: List of predicted translations
            references: List of reference translations
            
        Returns:
            Dictionary with exact match metrics
        """
        
        exact_matches = [pred.strip() == ref.strip() 
                        for pred, ref in zip(predictions, references)]
        
        return {
            'exact_match': np.mean(exact_matches),
            'exact_match_count': sum(exact_matches),
            'total_samples': len(predictions)
        }
    
    def compute_character_level_metrics(
        self, 
        predictions: List[str], 
        references: List[str]
    ) -> Dict[str, float]:
        """
        Compute character-level metrics.
        
        Args:
            predictions: List of predicted translations
            references: List of reference translations
            
        Returns:
            Dictionary with character-level metrics
        """
        
        char_precisions = []
        char_recalls = []
        char_f1s = []
        
        for pred, ref in zip(predictions, references):
            pred_chars = set(pred.lower())
            ref_chars = set(ref.lower())
            
            if len(pred_chars) == 0 and len(ref_chars) == 0:
                precision = recall = f1 = 1.0
            elif len(pred_chars) == 0:
                precision = recall = f1 = 0.0
            else:
                intersection = pred_chars & ref_chars
                precision = len(intersection) / len(pred_chars)
                recall = len(intersection) / len(ref_chars) if len(ref_chars) > 0 else 0
                f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
            
            char_precisions.append(precision)
            char_recalls.append(recall)
            char_f1s.append(f1)
        
        return {
            'char_precision': np.mean(char_precisions),
            'char_recall': np.mean(char_recalls),
            'char_f1': np.mean(char_f1s)
        }
    
    def evaluate_comprehensive(
        self, 
        predictions: List[str], 
        references: List[str],
        compute_all: bool = True
    ) -> Dict[str, float]:
        """
        Compute comprehensive evaluation metrics.
        
        Args:
            predictions: List of predicted translations
            references: List of reference translations
            compute_all: Whether to compute all available metrics
            
        Returns:
            Dictionary with all evaluation metrics
        """
        
        logger.info(f"Evaluating {len(predictions)} translations...")
        
        results = {}
        
        # BLEU scores
        try:
            bleu_results = self.compute_bleu(predictions, references)
            results.update(bleu_results)
            logger.info(f"BLEU: {bleu_results.get('bleu_corpus', bleu_results.get('bleu', 0)):.4f}")
        except Exception as e:
            logger.warning(f"Could not compute BLEU: {e}")
        
        # ROUGE scores
        if compute_all:
            try:
                rouge_results = self.compute_rouge(predictions, references)
                results.update(rouge_results)
                logger.info(f"ROUGE-L: {rouge_results.get('rougeL', rouge_results.get('rougeL_f', 0)):.4f}")
            except Exception as e:
                logger.warning(f"Could not compute ROUGE: {e}")
        
        # Length metrics
        try:
            length_results = self.compute_length_ratio(predictions, references)
            results.update(length_results)
        except Exception as e:
            logger.warning(f"Could not compute length metrics: {e}")
        
        # Exact match
        try:
            em_results = self.compute_exact_match(predictions, references)
            results.update(em_results)
            logger.info(f"Exact Match: {em_results['exact_match']:.4f}")
        except Exception as e:
            logger.warning(f"Could not compute exact match: {e}")
        
        # Character-level metrics
        if compute_all:
            try:
                char_results = self.compute_character_level_metrics(predictions, references)
                results.update(char_results)
            except Exception as e:
                logger.warning(f"Could not compute character metrics: {e}")
        
        return results
    
    def evaluate_model_on_dataset(
        self, 
        model, 
        dataset, 
        max_samples: int = None,
        batch_size: int = 8
    ) -> Dict[str, float]:
        """
        Evaluate a model on a dataset.
        
        Args:
            model: Translation model with translate() method
            dataset: Dataset with 'russian' and 'tatar' columns
            max_samples: Maximum number of samples to evaluate
            batch_size: Batch size for translation
            
        Returns:
            Dictionary with evaluation metrics
        """
        
        # Get test data
        if max_samples:
            indices = np.random.choice(len(dataset), min(max_samples, len(dataset)), replace=False)
            russian_texts = [dataset[i]['russian'] for i in indices]
            tatar_refs = [dataset[i]['tatar'] for i in indices]
        else:
            russian_texts = dataset['russian']
            tatar_refs = dataset['tatar']
        
        # Generate translations in batches
        predictions = []
        for i in range(0, len(russian_texts), batch_size):
            batch = russian_texts[i:i + batch_size]
            batch_predictions = model.translate(batch)
            predictions.extend(batch_predictions)
        
        # Evaluate
        results = self.evaluate_comprehensive(predictions, tatar_refs)
        
        return results


def compare_models(
    model_results: Dict[str, Dict[str, float]],
    metrics: List[str] = None
) -> Dict[str, str]:
    """
    Compare multiple models based on evaluation metrics.
    
    Args:
        model_results: Dictionary mapping model names to their results
        metrics: List of metrics to compare (if None, uses BLEU and ROUGE-L)
        
    Returns:
        Dictionary with comparison results
    """
    
    if metrics is None:
        metrics = ['bleu_corpus', 'bleu', 'rougeL', 'rougeL_f', 'exact_match']
    
    comparison = {}
    
    for metric in metrics:
        metric_scores = {}
        for model_name, results in model_results.items():
            if metric in results:
                metric_scores[model_name] = results[metric]
        
        if metric_scores:
            best_model = max(metric_scores.keys(), key=lambda k: metric_scores[k])
            comparison[f'best_{metric}'] = best_model
            comparison[f'{metric}_scores'] = metric_scores
    
    return comparison


if __name__ == "__main__":
    # Example usage
    evaluator = TranslationEvaluator()
    
    # Sample predictions and references
    predictions = [
        "Сәлам",
        "Ничек эшләр?",
        "Рәхмәт"
    ]
    
    references = [
        "Сәлам",
        "Ничек эшләр?",
        "Рәхмәт"
    ]
    
    # Evaluate
    results = evaluator.evaluate_comprehensive(predictions, references)
    
    print("Evaluation Results:")
    for metric, score in results.items():
        print(f"{metric}: {score:.4f}")