"""
Evaluation script for Russian-Tatar translation model.
"""

import argparse
import sys
import os
from pathlib import Path

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.data_preprocessing import load_tatar_dataset, create_sample_dataset
from src.model import TatarTranslationModel
from src.evaluation import TranslationEvaluator, compare_models
from src.utils import load_config, save_results, set_seed
import logging
import time

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Evaluate Russian-Tatar translation model"
    )
    
    parser.add_argument(
        '--model_path', 
        type=str, 
        required=True,
        help='Path to trained model directory'
    )
    
    parser.add_argument(
        '--config_path', 
        type=str, 
        default=None,
        help='Path to configuration file'
    )
    
    parser.add_argument(
        '--output_dir', 
        type=str, 
        default='./evaluation_results',
        help='Output directory for evaluation results'
    )
    
    parser.add_argument(
        '--dataset_name', 
        type=str, 
        default='IPSAN/tatar_translation_dataset',
        help='Dataset name or path'
    )
    
    parser.add_argument(
        '--max_samples', 
        type=int, 
        default=None,
        help='Maximum number of samples to evaluate'
    )
    
    parser.add_argument(
        '--batch_size', 
        type=int, 
        default=8,
        help='Batch size for evaluation'
    )
    
    parser.add_argument(
        '--sample_dataset', 
        action='store_true',
        help='Use sample dataset instead of loading from HuggingFace'
    )
    
    parser.add_argument(
        '--split', 
        type=str, 
        default='test',
        choices=['train', 'validation', 'test'],
        help='Dataset split to evaluate on'
    )
    
    parser.add_argument(
        '--seed', 
        type=int, 
        default=42,
        help='Random seed'
    )
    
    return parser.parse_args()


def main():
    """Main evaluation function."""
    
    # Parse arguments
    args = parse_args()
    
    logger.info("Starting Russian-Tatar translation model evaluation")
    logger.info(f"Arguments: {vars(args)}")
    
    # Set random seed
    set_seed(args.seed)
    
    # Create output directory
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    
    start_time = time.time()
    
    try:
        # Load configuration if provided
        config = None
        if args.config_path and Path(args.config_path).exists():
            config = load_config(args.config_path)
            logger.info(f"Loaded configuration from {args.config_path}")
        
        # Load dataset
        logger.info("Loading dataset...")
        if args.sample_dataset:
            dataset = create_sample_dataset(num_samples=1000)
        else:
            dataset = load_tatar_dataset(args.dataset_name)
        
        logger.info(f"Dataset loaded with {len(dataset[args.split])} samples in {args.split} split")
        
        # Load model
        logger.info(f"Loading model from {args.model_path}...")
        
        # Determine model architecture from config or try to infer
        architecture = 't5'  # Default
        model_name = 't5-small'  # Default
        
        if config:
            architecture = config.get('model', {}).get('architecture', 't5')
            model_name = config.get('model', {}).get('model_name', 't5-small')
        
        model = TatarTranslationModel(
            model_name=model_name,
            architecture=architecture
        )
        
        # Load the trained weights
        model.load_model(args.model_path)
        
        # Initialize evaluator
        evaluator = TranslationEvaluator()
        
        # Evaluate model
        logger.info(f"Evaluating model on {args.split} split...")
        eval_start_time = time.time()
        
        results = evaluator.evaluate_model_on_dataset(
            model=model,
            dataset=dataset[args.split],
            max_samples=args.max_samples,
            batch_size=args.batch_size
        )
        
        eval_time = time.time() - eval_start_time
        logger.info(f"Evaluation completed in {eval_time:.2f} seconds")
        
        # Print results
        logger.info("Evaluation Results:")
        for metric, score in results.items():
            if isinstance(score, (int, float)):
                logger.info(f"  {metric}: {score:.4f}")
            else:
                logger.info(f"  {metric}: {score}")
        
        # Generate some sample translations
        logger.info("Sample translations:")
        sample_size = min(10, len(dataset[args.split]))
        sample_indices = list(range(sample_size))
        
        sample_russian = [dataset[args.split][i]['russian'] for i in sample_indices]
        sample_tatar_ref = [dataset[args.split][i]['tatar'] for i in sample_indices]
        sample_tatar_pred = model.translate(sample_russian)
        
        for i, (ru, tat_ref, tat_pred) in enumerate(zip(sample_russian, sample_tatar_ref, sample_tatar_pred)):
            logger.info(f"  Sample {i+1}:")
            logger.info(f"    RU:  {ru}")
            logger.info(f"    REF: {tat_ref}")
            logger.info(f"    PRED: {tat_pred}")
        
        # Save detailed results
        detailed_results = {
            'model_path': args.model_path,
            'dataset_split': args.split,
            'num_samples': len(dataset[args.split]),
            'max_samples_evaluated': args.max_samples,
            'evaluation_time_seconds': eval_time,
            'metrics': results,
            'sample_translations': [
                {
                    'russian': ru,
                    'reference': ref,
                    'prediction': pred
                }
                for ru, ref, pred in zip(sample_russian, sample_tatar_ref, sample_tatar_pred)
            ]
        }
        
        if config:
            detailed_results['config'] = config
        
        # Save results
        results_path = os.path.join(args.output_dir, f'evaluation_results_{args.split}.json')
        save_results(detailed_results, results_path)
        
        total_time = time.time() - start_time
        logger.info(f"Total evaluation time: {total_time:.2f} seconds")
        logger.info(f"Evaluation completed successfully!")
        logger.info(f"Results saved to: {results_path}")
        
        return results_path
        
    except Exception as e:
        logger.error(f"Evaluation failed: {e}")
        import traceback
        traceback.print_exc()
        raise e


if __name__ == "__main__":
    main()