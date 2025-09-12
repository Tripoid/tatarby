"""
Utility functions for the Russian-Tatar translation system.
"""

import os
import json
import yaml
import torch
import numpy as np
import random
from pathlib import Path
from typing import Dict, Any, List, Optional
import logging
from datetime import datetime

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def set_seed(seed: int = 42):
    """
    Set random seeds for reproducibility.
    
    Args:
        seed: Random seed value
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    
    # For deterministic behavior (may impact performance)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    logger.info(f"Random seed set to {seed}")


def save_config(config: Dict[str, Any], config_path: str):
    """
    Save configuration to YAML file.
    
    Args:
        config: Configuration dictionary
        config_path: Path to save the configuration
    """
    config_path = Path(config_path)
    config_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(config_path, 'w', encoding='utf-8') as f:
        yaml.dump(config, f, default_flow_style=False, allow_unicode=True)
    
    logger.info(f"Configuration saved to {config_path}")


def load_config(config_path: str) -> Dict[str, Any]:
    """
    Load configuration from YAML file.
    
    Args:
        config_path: Path to the configuration file
        
    Returns:
        Configuration dictionary
    """
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    logger.info(f"Configuration loaded from {config_path}")
    return config


def save_results(results: Dict[str, Any], results_path: str):
    """
    Save results to JSON file.
    
    Args:
        results: Results dictionary
        results_path: Path to save the results
    """
    results_path = Path(results_path)
    results_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Add timestamp
    results['timestamp'] = datetime.now().isoformat()
    
    with open(results_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    logger.info(f"Results saved to {results_path}")


def load_results(results_path: str) -> Dict[str, Any]:
    """
    Load results from JSON file.
    
    Args:
        results_path: Path to the results file
        
    Returns:
        Results dictionary
    """
    with open(results_path, 'r', encoding='utf-8') as f:
        results = json.load(f)
    
    logger.info(f"Results loaded from {results_path}")
    return results


def get_device() -> str:
    """
    Get the best available device for computation.
    
    Returns:
        Device string ('cuda', 'mps', or 'cpu')
    """
    if torch.cuda.is_available():
        device = "cuda"
        logger.info(f"Using CUDA: {torch.cuda.get_device_name()}")
    elif torch.backends.mps.is_available():
        device = "mps"
        logger.info("Using Apple Metal Performance Shaders (MPS)")
    else:
        device = "cpu"
        logger.info("Using CPU")
    
    return device


def count_parameters(model) -> int:
    """
    Count the number of trainable parameters in a model.
    
    Args:
        model: PyTorch model
        
    Returns:
        Number of trainable parameters
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def format_time(seconds: float) -> str:
    """
    Format time in seconds to a human-readable string.
    
    Args:
        seconds: Time in seconds
        
    Returns:
        Formatted time string
    """
    if seconds < 60:
        return f"{seconds:.1f}s"
    elif seconds < 3600:
        minutes = seconds / 60
        return f"{minutes:.1f}m"
    else:
        hours = seconds / 3600
        return f"{hours:.1f}h"


def create_experiment_dir(base_dir: str = "experiments") -> str:
    """
    Create a unique experiment directory with timestamp.
    
    Args:
        base_dir: Base directory for experiments
        
    Returns:
        Path to the created experiment directory
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    exp_dir = Path(base_dir) / f"exp_{timestamp}"
    exp_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"Created experiment directory: {exp_dir}")
    return str(exp_dir)


def log_gpu_memory():
    """Log GPU memory usage if CUDA is available."""
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            allocated = torch.cuda.memory_allocated(i) / 1024**3  # GB
            cached = torch.cuda.memory_reserved(i) / 1024**3  # GB
            logger.info(f"GPU {i}: {allocated:.2f}GB allocated, {cached:.2f}GB cached")


class MetricsLogger:
    """Simple metrics logger for tracking training progress."""
    
    def __init__(self, log_file: Optional[str] = None):
        """
        Initialize the metrics logger.
        
        Args:
            log_file: Optional file to save metrics
        """
        self.metrics = []
        self.log_file = log_file
        
        if self.log_file:
            Path(self.log_file).parent.mkdir(parents=True, exist_ok=True)
    
    def log(self, step: int, metrics: Dict[str, float]):
        """
        Log metrics for a given step.
        
        Args:
            step: Training step
            metrics: Dictionary of metrics
        """
        entry = {"step": step, "timestamp": datetime.now().isoformat()}
        entry.update(metrics)
        self.metrics.append(entry)
        
        # Print to console
        metrics_str = ", ".join([f"{k}: {v:.4f}" for k, v in metrics.items()])
        logger.info(f"Step {step}: {metrics_str}")
        
        # Save to file if specified
        if self.log_file:
            with open(self.log_file, 'w', encoding='utf-8') as f:
                json.dump(self.metrics, f, indent=2, ensure_ascii=False)
    
    def get_metrics(self) -> List[Dict[str, Any]]:
        """Get all logged metrics."""
        return self.metrics
    
    def get_best_metric(self, metric_name: str, mode: str = "max") -> Dict[str, Any]:
        """
        Get the best value for a specific metric.
        
        Args:
            metric_name: Name of the metric
            mode: 'max' for highest value, 'min' for lowest value
            
        Returns:
            Dictionary with the best metric entry
        """
        if not self.metrics:
            return {}
        
        valid_entries = [entry for entry in self.metrics if metric_name in entry]
        if not valid_entries:
            return {}
        
        if mode == "max":
            best_entry = max(valid_entries, key=lambda x: x[metric_name])
        else:
            best_entry = min(valid_entries, key=lambda x: x[metric_name])
        
        return best_entry


def validate_config(config: Dict[str, Any]) -> bool:
    """
    Validate configuration dictionary.
    
    Args:
        config: Configuration dictionary
        
    Returns:
        True if configuration is valid
    """
    required_keys = ['model', 'training', 'data']
    
    for key in required_keys:
        if key not in config:
            logger.error(f"Missing required configuration key: {key}")
            return False
    
    # Validate model config
    model_config = config['model']
    if 'architecture' not in model_config:
        logger.error("Missing 'architecture' in model configuration")
        return False
    
    if model_config['architecture'] not in ['t5', 'marian', 'auto']:
        logger.error(f"Unsupported architecture: {model_config['architecture']}")
        return False
    
    # Validate training config
    training_config = config['training']
    required_training_keys = ['num_epochs', 'batch_size', 'learning_rate']
    
    for key in required_training_keys:
        if key not in training_config:
            logger.error(f"Missing required training configuration key: {key}")
            return False
    
    logger.info("Configuration validation passed")
    return True


def create_default_config() -> Dict[str, Any]:
    """
    Create a default configuration dictionary.
    
    Returns:
        Default configuration
    """
    return {
        'model': {
            'architecture': 't5',
            'model_name': 't5-small',
            'max_length': 128
        },
        'training': {
            'num_epochs': 3,
            'batch_size': 8,
            'learning_rate': 5e-5,
            'warmup_steps': 500,
            'save_steps': 500,
            'eval_steps': 500,
            'logging_steps': 100,
            'patience': 3
        },
        'data': {
            'dataset_name': 'IPSAN/tatar_translation_dataset',
            'source_lang': 'russian',
            'target_lang': 'tatar',
            'test_size': 0.1,
            'val_size': 0.1
        },
        'evaluation': {
            'metrics': ['bleu', 'rouge', 'exact_match'],
            'max_samples': 1000
        },
        'paths': {
            'output_dir': './results',
            'model_dir': './models',
            'logs_dir': './logs'
        }
    }


if __name__ == "__main__":
    # Example usage
    print("Setting up utilities...")
    
    # Set seed for reproducibility
    set_seed(42)
    
    # Create default config
    config = create_default_config()
    print("Default configuration:")
    print(yaml.dump(config, default_flow_style=False))
    
    # Validate config
    is_valid = validate_config(config)
    print(f"Configuration valid: {is_valid}")
    
    # Check device
    device = get_device()
    print(f"Using device: {device}")
    
    # Create experiment directory
    exp_dir = create_experiment_dir()
    print(f"Experiment directory: {exp_dir}")
    
    # Example metrics logging
    logger_instance = MetricsLogger()
    logger_instance.log(100, {"loss": 0.5, "bleu": 0.3})
    logger_instance.log(200, {"loss": 0.4, "bleu": 0.35})
    
    best_bleu = logger_instance.get_best_metric("bleu", "max")
    print(f"Best BLEU: {best_bleu}")