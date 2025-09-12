"""
Demo script showing the translation system functionality without requiring model downloads.
This demonstrates the system architecture and data processing capabilities.
"""

import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.data_preprocessing import create_sample_dataset, get_dataset_statistics
from src.evaluation import TranslationEvaluator
from src.utils import set_seed, create_default_config, validate_config
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def demo_data_processing():
    """Demonstrate data preprocessing capabilities."""
    logger.info("=== Data Processing Demo ===")
    
    # Create sample dataset
    logger.info("Creating sample dataset...")
    dataset = create_sample_dataset(num_samples=100)
    
    logger.info(f"Dataset created with splits:")
    for split_name, split_data in dataset.items():
        logger.info(f"  {split_name}: {len(split_data)} samples")
    
    # Show sample data
    logger.info("\nSample translations:")
    for i in range(min(5, len(dataset['train']))):
        sample = dataset['train'][i]
        logger.info(f"  RU: {sample['russian']}")
        logger.info(f"  TAT: {sample['tatar']}")
        logger.info("")
    
    # Get statistics
    logger.info("Computing dataset statistics...")
    stats = get_dataset_statistics(dataset)
    
    for split_name, split_stats in stats.items():
        logger.info(f"{split_name.upper()} statistics:")
        for key, value in split_stats.items():
            if isinstance(value, float):
                logger.info(f"  {key}: {value:.2f}")
            else:
                logger.info(f"  {key}: {value}")
        logger.info("")
    
    return dataset


def demo_evaluation():
    """Demonstrate evaluation capabilities."""
    logger.info("=== Evaluation Demo ===")
    
    # Create sample predictions and references
    predictions = [
        "Сәлам",
        "Ничек эшләр?", 
        "Рәхмәт",
        "Сау бул",
        "Исәнмесез"
    ]
    
    references = [
        "Сәлам",
        "Ничек эшләр?",
        "Рәхмәт", 
        "Сау бул",
        "Исәнмесез"
    ]
    
    logger.info("Sample predictions and references:")
    for pred, ref in zip(predictions, references):
        logger.info(f"  PRED: {pred}")
        logger.info(f"  REF:  {ref}")
        logger.info("")
    
    # Evaluate
    evaluator = TranslationEvaluator()
    results = evaluator.evaluate_comprehensive(predictions, references)
    
    logger.info("Evaluation results:")
    for metric, score in results.items():
        if isinstance(score, (int, float)):
            logger.info(f"  {metric}: {score:.4f}")
        else:
            logger.info(f"  {metric}: {score}")


def demo_configuration():
    """Demonstrate configuration system."""
    logger.info("=== Configuration Demo ===")
    
    # Create default configuration
    config = create_default_config()
    
    logger.info("Default configuration:")
    logger.info(f"  Model: {config['model']['model_name']} ({config['model']['architecture']})")
    logger.info(f"  Training epochs: {config['training']['num_epochs']}")
    logger.info(f"  Batch size: {config['training']['batch_size']}")
    logger.info(f"  Learning rate: {config['training']['learning_rate']}")
    logger.info(f"  Dataset: {config['data']['dataset_name']}")
    
    # Validate configuration
    is_valid = validate_config(config)
    logger.info(f"Configuration is valid: {is_valid}")


def demo_system_overview():
    """Provide an overview of the complete system."""
    logger.info("=== Russian-Tatar Translation System Overview ===")
    logger.info("")
    logger.info("This system provides:")
    logger.info("1. Data preprocessing for Russian-Tatar translation pairs")
    logger.info("2. Multiple model architectures (T5, MarianMT, BERT-based)")
    logger.info("3. Comprehensive evaluation with BLEU, ROUGE, and custom metrics")
    logger.info("4. Training pipeline with early stopping and checkpointing")
    logger.info("5. Inference capabilities for interactive and batch translation")
    logger.info("6. Configuration management with YAML files")
    logger.info("")
    logger.info("Key features:")
    logger.info("- Supports datasets from Hugging Face Hub")
    logger.info("- Automatic model configuration based on dataset size")
    logger.info("- GPU acceleration when available")
    logger.info("- Extensible architecture for new models and metrics")
    logger.info("- Production-ready inference scripts")
    logger.info("")
    logger.info("Files included:")
    logger.info("- src/data_preprocessing.py: Dataset loading and preprocessing")
    logger.info("- src/model.py: Translation model implementations")
    logger.info("- src/evaluation.py: Comprehensive evaluation metrics")
    logger.info("- src/utils.py: Utility functions and configuration")
    logger.info("- scripts/train.py: Training script")
    logger.info("- scripts/eval_model.py: Evaluation script")
    logger.info("- scripts/inference.py: Inference script")
    logger.info("- configs/config.yaml: Configuration template")
    logger.info("- docs/model_documentation.md: Detailed documentation")
    logger.info("")


def main():
    """Run the complete demo."""
    set_seed(42)
    
    demo_system_overview()
    dataset = demo_data_processing()
    demo_evaluation()
    demo_configuration()
    
    logger.info("=== Demo Complete ===")
    logger.info("")
    logger.info("To use the system:")
    logger.info("1. Ensure you have internet access for model downloads")
    logger.info("2. Run: python scripts/train.py --sample_dataset")
    logger.info("3. Evaluate: python scripts/eval_model.py --model_path ./results/model --sample_dataset")
    logger.info("4. Translate: python scripts/inference.py --model_path ./results/model --interactive")
    logger.info("")
    logger.info("For real datasets, replace --sample_dataset with the actual dataset configuration.")


if __name__ == "__main__":
    main()