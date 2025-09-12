"""
Data preprocessing utilities for Russian-Tatar translation.
"""

import pandas as pd
import numpy as np
from datasets import Dataset, DatasetDict
from transformers import AutoTokenizer
from sklearn.model_selection import train_test_split
import logging
from typing import Dict, List, Tuple, Optional

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def create_sample_dataset(num_samples: int = 1000) -> DatasetDict:
    """
    Create a sample Russian-Tatar translation dataset for demonstration.
    This would be replaced with actual dataset loading once available.
    
    Args:
        num_samples: Number of sample translation pairs to generate
        
    Returns:
        DatasetDict with train/validation/test splits
    """
    
    # Sample Russian-Tatar translation pairs for demonstration
    sample_pairs = [
        ("Привет", "Сәлам"),
        ("Как дела?", "Ничек эшләр?"),
        ("Спасибо", "Рәхмәт"),
        ("До свидания", "Сау бул"),
        ("Добро пожаловать", "Рәхим итегез"),
        ("Здравствуйте", "Исәнмесез"),
        ("Пожалуйста", "Зинһар"),
        ("Извините", "Гафу итегез"),
        ("Я изучаю татарский язык", "Мин татар телен өйрәнәм"),
        ("Сегодня хорошая погода", "Бүген яхшы һава"),
        ("Мне нравится этот город", "Миңа бу шәһәр ошый"),
        ("Где находится библиотека?", "Китапханә кайда?"),
        ("Сколько это стоит?", "Бу ничә тора?"),
        ("Я не понимаю", "Мин аңламыйм"),
        ("Повторите, пожалуйста", "Кабатлагыз әле"),
        ("Меня зовут", "Минем исемем"),
        ("Очень приятно познакомиться", "Танышуга бик шат"),
        ("Какое время?", "Ничә сәгать?"),
        ("Я учусь в университете", "Мин университетта укыйм"),
        ("Это мой друг", "Бу минем дустым"),
    ]
    
    # Expand the dataset by creating variations
    russian_texts = []
    tatar_texts = []
    
    # Add base samples multiple times with slight variations
    for i in range(num_samples):
        base_idx = i % len(sample_pairs)
        russian, tatar = sample_pairs[base_idx]
        
        # Add some variations for larger dataset
        if i > len(sample_pairs):
            russian = f"{russian} (вариант {i // len(sample_pairs)})"
            tatar = f"{tatar} ({i // len(sample_pairs)} вариант)"
        
        russian_texts.append(russian)
        tatar_texts.append(tatar)
    
    # Create dataset
    data = {
        'russian': russian_texts,
        'tatar': tatar_texts
    }
    
    dataset = Dataset.from_dict(data)
    
    # Split into train/validation/test
    train_data, temp_data = train_test_split(
        list(zip(russian_texts, tatar_texts)), 
        test_size=0.3, 
        random_state=42
    )
    
    val_data, test_data = train_test_split(
        temp_data, 
        test_size=0.5, 
        random_state=42
    )
    
    # Create DatasetDict
    dataset_dict = DatasetDict({
        'train': Dataset.from_dict({
            'russian': [item[0] for item in train_data],
            'tatar': [item[1] for item in train_data]
        }),
        'validation': Dataset.from_dict({
            'russian': [item[0] for item in val_data],
            'tatar': [item[1] for item in val_data]
        }),
        'test': Dataset.from_dict({
            'russian': [item[0] for item in test_data],
            'tatar': [item[1] for item in test_data]
        })
    })
    
    logger.info(f"Created sample dataset with {len(dataset_dict['train'])} training samples")
    logger.info(f"Validation samples: {len(dataset_dict['validation'])}")
    logger.info(f"Test samples: {len(dataset_dict['test'])}")
    
    return dataset_dict


def load_tatar_dataset(dataset_name: str = "IPSAN/tatar_translation_dataset") -> DatasetDict:
    """
    Load the Tatar translation dataset from Hugging Face Hub.
    Falls back to sample dataset if the main dataset is not available.
    
    Args:
        dataset_name: Name of the dataset on Hugging Face Hub
        
    Returns:
        DatasetDict with train/validation/test splits
    """
    try:
        from datasets import load_dataset
        logger.info(f"Attempting to load {dataset_name}...")
        dataset = load_dataset(dataset_name)
        logger.info(f"Successfully loaded {dataset_name}")
        return dataset
    except Exception as e:
        logger.warning(f"Could not load {dataset_name}: {e}")
        logger.info("Falling back to sample dataset for demonstration")
        return create_sample_dataset()


def preprocess_dataset(
    dataset: DatasetDict,
    tokenizer_name: str = "Helsinki-NLP/opus-mt-ru-en",
    max_length: int = 128,
    source_lang: str = "russian",
    target_lang: str = "tatar"
) -> DatasetDict:
    """
    Preprocess the dataset for translation training.
    
    Args:
        dataset: Raw dataset
        tokenizer_name: Name of the tokenizer to use
        max_length: Maximum sequence length
        source_lang: Source language column name
        target_lang: Target language column name
        
    Returns:
        Preprocessed dataset ready for training
    """
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    
    # Ensure tokenizer has a pad token
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    def preprocess_function(examples):
        # Get source and target texts
        inputs = examples[source_lang]
        targets = examples[target_lang]
        
        # Tokenize inputs
        model_inputs = tokenizer(
            inputs, 
            max_length=max_length, 
            truncation=True, 
            padding=True,
            return_tensors=None
        )
        
        # Tokenize targets
        with tokenizer.as_target_tokenizer():
            labels = tokenizer(
                targets, 
                max_length=max_length, 
                truncation=True, 
                padding=True,
                return_tensors=None
            )
        
        model_inputs["labels"] = labels["input_ids"]
        return model_inputs
    
    # Apply preprocessing to all splits
    processed_dataset = dataset.map(
        preprocess_function,
        batched=True,
        remove_columns=dataset["train"].column_names
    )
    
    logger.info("Dataset preprocessing completed")
    for split_name, split_data in processed_dataset.items():
        logger.info(f"{split_name}: {len(split_data)} samples")
    
    return processed_dataset


def get_dataset_statistics(dataset: DatasetDict, source_lang: str = "russian", target_lang: str = "tatar") -> Dict:
    """
    Get statistics about the dataset.
    
    Args:
        dataset: Dataset to analyze
        source_lang: Source language column name
        target_lang: Target language column name
        
    Returns:
        Dictionary with dataset statistics
    """
    
    stats = {}
    
    for split_name, split_data in dataset.items():
        split_stats = {
            'num_samples': len(split_data),
            'avg_source_length': np.mean([len(text.split()) for text in split_data[source_lang]]),
            'avg_target_length': np.mean([len(text.split()) for text in split_data[target_lang]]),
            'max_source_length': max([len(text.split()) for text in split_data[source_lang]]),
            'max_target_length': max([len(text.split()) for text in split_data[target_lang]]),
        }
        stats[split_name] = split_stats
    
    return stats


if __name__ == "__main__":
    # Example usage
    print("Loading Tatar translation dataset...")
    dataset = load_tatar_dataset()
    
    print("\nDataset structure:")
    print(dataset)
    
    print("\nDataset statistics:")
    stats = get_dataset_statistics(dataset)
    for split_name, split_stats in stats.items():
        print(f"\n{split_name.upper()}:")
        for key, value in split_stats.items():
            print(f"  {key}: {value:.2f}" if isinstance(value, float) else f"  {key}: {value}")
    
    print("\nPreprocessing dataset...")
    processed_dataset = preprocess_dataset(dataset)
    print("Preprocessing completed!")