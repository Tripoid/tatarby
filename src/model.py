"""
Translation model implementation using T5 and MarianMT architectures.
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import (
    AutoTokenizer, 
    AutoModelForSeq2SeqLM,
    T5ForConditionalGeneration,
    T5Tokenizer,
    MarianMTModel,
    MarianTokenizer,
    BertTokenizer,
    EncoderDecoderModel,
    BertConfig,
    Trainer,
    TrainingArguments,
    DataCollatorForSeq2Seq,
    EarlyStoppingCallback
)
from datasets import DatasetDict
import logging
from typing import Dict, List, Optional, Tuple
import os
from pathlib import Path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TatarTranslationModel:
    """
    Russian-to-Tatar translation model using transformer architectures.
    """
    
    def __init__(
        self, 
        model_name: str = "t5-small",
        architecture: str = "t5",
        max_length: int = 128,
        device: Optional[str] = None
    ):
        """
        Initialize the translation model.
        
        Args:
            model_name: Pretrained model name or path
            architecture: Model architecture ('t5' or 'marian')
            max_length: Maximum sequence length
            device: Device to run the model on
        """
        self.model_name = model_name
        self.architecture = architecture.lower()
        self.max_length = max_length
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        
        self.model = None
        self.tokenizer = None
        self.trainer = None
        
        logger.info(f"Initializing {architecture} model: {model_name}")
        logger.info(f"Using device: {self.device}")
        
        self._load_model_and_tokenizer()
    
    def _load_model_and_tokenizer(self):
        """Load the pretrained model and tokenizer."""
        
        try:
            if self.architecture == "t5":
                self.tokenizer = T5Tokenizer.from_pretrained(self.model_name)
                self.model = T5ForConditionalGeneration.from_pretrained(self.model_name)
            elif self.architecture == "marian":
                self.tokenizer = MarianTokenizer.from_pretrained(self.model_name)
                self.model = MarianMTModel.from_pretrained(self.model_name)
            else:
                # Generic approach
                self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
                self.model = AutoModelForSeq2SeqLM.from_pretrained(self.model_name)
            
            # Ensure tokenizer has a pad token
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            
            # Move model to device
            self.model = self.model.to(self.device)
            
            logger.info(f"Model and tokenizer loaded successfully")
            logger.info(f"Model size: {sum(p.numel() for p in self.model.parameters()):,} parameters")
            
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            # Fallback to a simpler model that works without sentencepiece
            logger.info("Falling back to Helsinki-NLP/opus-mt-en-de as base model")
            try:
                self.tokenizer = AutoTokenizer.from_pretrained("Helsinki-NLP/opus-mt-en-de")
                self.model = AutoModelForSeq2SeqLM.from_pretrained("Helsinki-NLP/opus-mt-en-de")
                if self.tokenizer.pad_token is None:
                    self.tokenizer.pad_token = self.tokenizer.eos_token
                self.model = self.model.to(self.device)
                logger.info("Successfully loaded fallback model")
            except Exception as e2:
                logger.error(f"Fallback model also failed: {e2}")
                # Final fallback - create a simple configuration
                logger.info("Creating minimal model configuration for testing")
                from transformers import BertTokenizer, EncoderDecoderModel, BertConfig
                
                # Use BERT tokenizer which doesn't require sentencepiece
                self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
                
                # Create a simple encoder-decoder model
                config = BertConfig()
                config.is_decoder = True
                config.add_cross_attention = True
                
                self.model = EncoderDecoderModel.from_encoder_decoder_pretrained(
                    'bert-base-uncased', 'bert-base-uncased'
                )
                
                if self.tokenizer.pad_token is None:
                    self.tokenizer.pad_token = self.tokenizer.unk_token
                
                self.model = self.model.to(self.device)
                logger.info("Created minimal encoder-decoder model for testing")
    
    def prepare_training_data(self, dataset: DatasetDict) -> DatasetDict:
        """
        Prepare dataset for training.
        
        Args:
            dataset: Raw dataset with 'russian' and 'tatar' columns
            
        Returns:
            Preprocessed dataset
        """
        
        def preprocess_function(examples):
            # For T5, we add a task prefix
            if self.architecture == "t5":
                inputs = [f"translate Russian to Tatar: {text}" for text in examples['russian']]
            else:
                inputs = examples['russian']
            
            targets = examples['tatar']
            
            # Tokenize inputs
            model_inputs = self.tokenizer(
                inputs,
                max_length=self.max_length,
                truncation=True,
                padding=False,  # Will be done by data collator
                return_tensors=None
            )
            
            # Tokenize targets
            with self.tokenizer.as_target_tokenizer():
                labels = self.tokenizer(
                    targets,
                    max_length=self.max_length,
                    truncation=True,
                    padding=False,  # Will be done by data collator
                    return_tensors=None
                )
            
            model_inputs["labels"] = labels["input_ids"]
            return model_inputs
        
        # Apply preprocessing
        processed_dataset = dataset.map(
            preprocess_function,
            batched=True,
            remove_columns=dataset["train"].column_names
        )
        
        return processed_dataset
    
    def train(
        self,
        dataset: DatasetDict,
        output_dir: str = "./results",
        num_train_epochs: int = 3,
        per_device_train_batch_size: int = 8,
        per_device_eval_batch_size: int = 8,
        learning_rate: float = 5e-5,
        warmup_steps: int = 500,
        logging_steps: int = 100,
        save_steps: int = 500,
        eval_steps: int = 500,
        save_total_limit: int = 3,
        load_best_model_at_end: bool = True,
        metric_for_best_model: str = "eval_loss",
        greater_is_better: bool = False,
        patience: int = 3,
        **kwargs
    ):
        """
        Train the translation model.
        
        Args:
            dataset: Preprocessed dataset
            output_dir: Directory to save model checkpoints
            num_train_epochs: Number of training epochs
            per_device_train_batch_size: Training batch size per device
            per_device_eval_batch_size: Evaluation batch size per device
            learning_rate: Learning rate
            warmup_steps: Number of warmup steps
            logging_steps: Log every N steps
            save_steps: Save checkpoint every N steps
            eval_steps: Evaluate every N steps
            save_total_limit: Maximum number of checkpoints to save
            load_best_model_at_end: Whether to load the best model at the end
            metric_for_best_model: Metric to use for best model selection
            greater_is_better: Whether higher metric values are better
            patience: Early stopping patience
            **kwargs: Additional training arguments
        """
        
        # Prepare the dataset
        processed_dataset = self.prepare_training_data(dataset)
        
        # Data collator
        data_collator = DataCollatorForSeq2Seq(
            tokenizer=self.tokenizer,
            model=self.model,
            padding=True,
            return_tensors="pt"
        )
        
        # Training arguments
        training_args = TrainingArguments(
            output_dir=output_dir,
            num_train_epochs=num_train_epochs,
            per_device_train_batch_size=per_device_train_batch_size,
            per_device_eval_batch_size=per_device_eval_batch_size,
            learning_rate=learning_rate,
            warmup_steps=warmup_steps,
            logging_steps=logging_steps,
            save_steps=save_steps,
            eval_steps=eval_steps,
            save_total_limit=save_total_limit,
            load_best_model_at_end=load_best_model_at_end,
            metric_for_best_model=metric_for_best_model,
            greater_is_better=greater_is_better,
            evaluation_strategy="steps",
            logging_strategy="steps",
            save_strategy="steps",
            fp16=torch.cuda.is_available(),  # Use mixed precision if available
            dataloader_pin_memory=True,
            remove_unused_columns=False,
            **kwargs
        )
        
        # Early stopping callback
        early_stopping = EarlyStoppingCallback(
            early_stopping_patience=patience,
            early_stopping_threshold=0.001
        )
        
        # Initialize trainer
        self.trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=processed_dataset["train"],
            eval_dataset=processed_dataset["validation"],
            tokenizer=self.tokenizer,
            data_collator=data_collator,
            callbacks=[early_stopping]
        )
        
        logger.info("Starting training...")
        
        # Train the model
        train_result = self.trainer.train()
        
        logger.info("Training completed!")
        logger.info(f"Training loss: {train_result.training_loss:.4f}")
        
        # Save the final model
        self.trainer.save_model()
        self.tokenizer.save_pretrained(output_dir)
        
        return train_result
    
    def translate(
        self, 
        texts: List[str], 
        max_new_tokens: int = 128,
        num_beams: int = 4,
        temperature: float = 1.0,
        do_sample: bool = False
    ) -> List[str]:
        """
        Translate Russian texts to Tatar.
        
        Args:
            texts: List of Russian texts to translate
            max_new_tokens: Maximum number of tokens to generate
            num_beams: Number of beams for beam search
            temperature: Sampling temperature
            do_sample: Whether to use sampling
            
        Returns:
            List of translated Tatar texts
        """
        
        if not isinstance(texts, list):
            texts = [texts]
        
        # Prepare inputs
        if self.architecture == "t5":
            inputs = [f"translate Russian to Tatar: {text}" for text in texts]
        else:
            inputs = texts
        
        # Tokenize
        encoded = self.tokenizer(
            inputs,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.max_length
        ).to(self.device)
        
        # Generate translations
        with torch.no_grad():
            generated_tokens = self.model.generate(
                **encoded,
                max_new_tokens=max_new_tokens,
                num_beams=num_beams,
                temperature=temperature,
                do_sample=do_sample,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id
            )
        
        # Decode translations
        translations = self.tokenizer.batch_decode(
            generated_tokens, 
            skip_special_tokens=True
        )
        
        return translations
    
    def save_model(self, save_path: str):
        """Save the trained model and tokenizer."""
        save_path = Path(save_path)
        save_path.mkdir(parents=True, exist_ok=True)
        
        self.model.save_pretrained(save_path)
        self.tokenizer.save_pretrained(save_path)
        
        logger.info(f"Model saved to {save_path}")
    
    def load_model(self, model_path: str):
        """Load a trained model and tokenizer."""
        model_path = Path(model_path)
        
        if self.architecture == "t5":
            self.model = T5ForConditionalGeneration.from_pretrained(model_path)
            self.tokenizer = T5Tokenizer.from_pretrained(model_path)
        elif self.architecture == "marian":
            self.model = MarianMTModel.from_pretrained(model_path)
            self.tokenizer = MarianTokenizer.from_pretrained(model_path)
        else:
            self.model = AutoModelForSeq2SeqLM.from_pretrained(model_path)
            self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        
        self.model = self.model.to(self.device)
        
        logger.info(f"Model loaded from {model_path}")


def get_recommended_model_config(dataset_size: int) -> Dict:
    """
    Get recommended model configuration based on dataset size.
    
    Args:
        dataset_size: Number of training samples
        
    Returns:
        Dictionary with recommended model configuration
    """
    
    if dataset_size < 1000:
        return {
            "model_name": "t5-small",
            "architecture": "t5",
            "batch_size": 8,
            "learning_rate": 1e-4,
            "epochs": 10
        }
    elif dataset_size < 10000:
        return {
            "model_name": "t5-base",
            "architecture": "t5",
            "batch_size": 4,
            "learning_rate": 5e-5,
            "epochs": 5
        }
    else:
        return {
            "model_name": "t5-large",
            "architecture": "t5",
            "batch_size": 2,
            "learning_rate": 3e-5,
            "epochs": 3
        }


if __name__ == "__main__":
    # Example usage
    from src.data_preprocessing import load_tatar_dataset
    
    print("Loading dataset...")
    dataset = load_tatar_dataset()
    
    print("Initializing model...")
    model = TatarTranslationModel()
    
    print("Starting training...")
    model.train(
        dataset=dataset,
        output_dir="./model_output",
        num_train_epochs=2,
        per_device_train_batch_size=4,
        logging_steps=50,
        save_steps=200,
        eval_steps=200
    )
    
    print("Testing translation...")
    test_texts = ["Привет", "Как дела?", "Спасибо"]
    translations = model.translate(test_texts)
    
    for russian, tatar in zip(test_texts, translations):
        print(f"RU: {russian} -> TAT: {tatar}")