"""
Main training script for Russian-Tatar translation model.
"""

import argparse
import sys
import os
from pathlib import Path

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.data_preprocessing import load_tatar_dataset, get_dataset_statistics
from src.model import TatarTranslationModel, get_recommended_model_config
from src.evaluation import TranslationEvaluator
from src.utils import (
    set_seed, save_config, load_config, save_results, 
    get_device, create_experiment_dir, validate_config,
    create_default_config, MetricsLogger, log_gpu_memory
)
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
        description="Train Russian-Tatar translation model"
    )
    
    parser.add_argument(
        '--config', 
        type=str, 
        default=None,
        help='Path to configuration file'
    )
    
    parser.add_argument(
        '--output_dir', 
        type=str, 
        default='./results',
        help='Output directory for model and results'
    )
    
    parser.add_argument(
        '--model_name', 
        type=str, 
        default='t5-small',
        help='Pretrained model name'
    )
    
    parser.add_argument(
        '--architecture', 
        type=str, 
        default='t5',
        choices=['t5', 'marian', 'auto'],
        help='Model architecture'
    )
    
    parser.add_argument(
        '--num_epochs', 
        type=int, 
        default=3,
        help='Number of training epochs'
    )
    
    parser.add_argument(
        '--batch_size', 
        type=int, 
        default=8,
        help='Training batch size'
    )
    
    parser.add_argument(
        '--learning_rate', 
        type=float, 
        default=5e-5,
        help='Learning rate'
    )
    
    parser.add_argument(
        '--max_length', 
        type=int, 
        default=128,
        help='Maximum sequence length'
    )
    
    parser.add_argument(
        '--seed', 
        type=int, 
        default=42,
        help='Random seed'
    )
    
    parser.add_argument(
        '--no_eval', 
        action='store_true',
        help='Skip evaluation after training'
    )
    
    parser.add_argument(
        '--sample_dataset', 
        action='store_true',
        help='Use sample dataset instead of trying to load from HuggingFace'
    )
    
    return parser.parse_args()


def setup_experiment(args):
    """Set up experiment configuration and directories."""
    
    # Set random seed
    set_seed(args.seed)
    
    # Create experiment directory
    if args.config:
        # Use existing output directory if config is provided
        exp_dir = args.output_dir
        Path(exp_dir).mkdir(parents=True, exist_ok=True)
    else:
        # Create new experiment directory
        exp_dir = create_experiment_dir(args.output_dir)
    
    # Load or create configuration
    if args.config and Path(args.config).exists():
        config = load_config(args.config)
        logger.info(f"Loaded configuration from {args.config}")
    else:
        # Create configuration from arguments
        config = create_default_config()
        
        # Update config with command line arguments
        config['model']['model_name'] = args.model_name
        config['model']['architecture'] = args.architecture
        config['model']['max_length'] = args.max_length
        
        config['training']['num_epochs'] = args.num_epochs
        config['training']['batch_size'] = args.batch_size
        config['training']['learning_rate'] = args.learning_rate
        
        config['paths']['output_dir'] = exp_dir
        config['paths']['model_dir'] = os.path.join(exp_dir, 'model')
        config['paths']['logs_dir'] = os.path.join(exp_dir, 'logs')
        
        if args.sample_dataset:
            config['data']['use_sample'] = True
    
    # Validate configuration
    if not validate_config(config):
        raise ValueError("Invalid configuration")
    
    # Save configuration
    config_path = os.path.join(exp_dir, 'config.yaml')
    save_config(config, config_path)
    
    return config, exp_dir


def main():
    """Main training function."""
    
    # Parse arguments
    args = parse_args()
    
    logger.info("Starting Russian-Tatar translation model training")
    logger.info(f"Arguments: {vars(args)}")
    
    # Setup experiment
    config, exp_dir = setup_experiment(args)
    
    # Log system information
    device = get_device()
    log_gpu_memory()
    
    # Create metrics logger
    metrics_logger = MetricsLogger(
        log_file=os.path.join(config['paths']['logs_dir'], 'metrics.json')
    )
    
    start_time = time.time()
    
    try:
        # Load dataset
        logger.info("Loading dataset...")
        if config['data'].get('use_sample', False) or args.sample_dataset:
            from src.data_preprocessing import create_sample_dataset
            dataset = create_sample_dataset(num_samples=1000)
        else:
            dataset = load_tatar_dataset(config['data']['dataset_name'])
        
        # Get dataset statistics
        logger.info("Analyzing dataset...")
        stats = get_dataset_statistics(dataset)
        
        logger.info("Dataset statistics:")
        for split_name, split_stats in stats.items():
            logger.info(f"{split_name.upper()}:")
            for key, value in split_stats.items():
                if isinstance(value, float):
                    logger.info(f"  {key}: {value:.2f}")
                else:
                    logger.info(f"  {key}: {value}")
        
        # Get recommended model configuration based on dataset size
        dataset_size = len(dataset['train'])
        recommended_config = get_recommended_model_config(dataset_size)
        
        logger.info(f"Recommended configuration for {dataset_size} samples:")
        for key, value in recommended_config.items():
            logger.info(f"  {key}: {value}")
        
        # Initialize model
        logger.info("Initializing model...")
        model = TatarTranslationModel(
            model_name=config['model']['model_name'],
            architecture=config['model']['architecture'],
            max_length=config['model']['max_length'],
            device=device
        )
        
        # Train model
        logger.info("Starting training...")
        train_start_time = time.time()
        
        training_args = {
            'dataset': dataset,
            'output_dir': config['paths']['model_dir'],
            'num_train_epochs': config['training']['num_epochs'],
            'per_device_train_batch_size': config['training']['batch_size'],
            'learning_rate': config['training']['learning_rate'],
            'warmup_steps': config['training']['warmup_steps'],
            'logging_steps': config['training']['logging_steps'],
            'save_steps': config['training']['save_steps'],
            'eval_steps': config['training']['eval_steps'],
            'patience': config['training']['patience']
        }
        
        train_result = model.train(**training_args)
        
        train_time = time.time() - train_start_time
        logger.info(f"Training completed in {train_time:.2f} seconds")
        
        # Log training metrics
        metrics_logger.log(
            step=0,
            metrics={
                'training_loss': train_result.training_loss,
                'train_time_seconds': train_time
            }
        )
        
        # Save model
        logger.info("Saving model...")
        final_model_path = os.path.join(exp_dir, 'final_model')
        model.save_model(final_model_path)
        
        # Evaluate model
        if not args.no_eval:
            logger.info("Evaluating model...")
            eval_start_time = time.time()
            
            evaluator = TranslationEvaluator()
            
            # Evaluate on validation set
            val_results = evaluator.evaluate_model_on_dataset(
                model=model,
                dataset=dataset['validation'],
                max_samples=config['evaluation'].get('max_samples', None)
            )
            
            # Evaluate on test set
            test_results = evaluator.evaluate_model_on_dataset(
                model=model,
                dataset=dataset['test'],
                max_samples=config['evaluation'].get('max_samples', None)
            )
            
            eval_time = time.time() - eval_start_time
            logger.info(f"Evaluation completed in {eval_time:.2f} seconds")
            
            # Log evaluation results
            logger.info("Validation Results:")
            for metric, score in val_results.items():
                if isinstance(score, (int, float)):
                    logger.info(f"  {metric}: {score:.4f}")
            
            logger.info("Test Results:")
            for metric, score in test_results.items():
                if isinstance(score, (int, float)):
                    logger.info(f"  {metric}: {score:.4f}")
            
            # Save evaluation results
            all_results = {
                'training': {
                    'loss': train_result.training_loss,
                    'time_seconds': train_time
                },
                'validation': val_results,
                'test': test_results,
                'dataset_stats': stats,
                'config': config
            }
            
            results_path = os.path.join(exp_dir, 'results.json')
            save_results(all_results, results_path)
            
            # Test some sample translations
            logger.info("Sample translations:")
            sample_texts = [
                "Привет",
                "Как дела?",
                "Спасибо",
                "До свидания"
            ]
            
            translations = model.translate(sample_texts)
            for russian, tatar in zip(sample_texts, translations):
                logger.info(f"  RU: {russian} -> TAT: {tatar}")
        
        total_time = time.time() - start_time
        logger.info(f"Total experiment time: {total_time:.2f} seconds")
        logger.info(f"Experiment completed successfully!")
        logger.info(f"Results saved to: {exp_dir}")
        
        return exp_dir
        
    except Exception as e:
        logger.error(f"Training failed: {e}")
        import traceback
        traceback.print_exc()
        raise e


if __name__ == "__main__":
    main()