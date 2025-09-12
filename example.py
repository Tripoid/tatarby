"""
Example usage of the Tatarby Russian-Tatar translation system.
"""

import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from src.data_preprocessing import create_sample_dataset
from src.evaluation import TranslationEvaluator


def basic_example():
    """Basic example of using the translation system."""
    print("=== Tatarby Translation System Example ===\n")
    
    # 1. Create or load dataset
    print("1. Creating sample dataset...")
    dataset = create_sample_dataset(num_samples=20)
    print(f"   Created dataset with {len(dataset['train'])} training samples\n")
    
    # 2. Show some examples
    print("2. Sample Russian-Tatar pairs:")
    for i in range(5):
        sample = dataset['train'][i]
        print(f"   RU:  {sample['russian']}")
        print(f"   TAT: {sample['tatar']}")
        print()
    
    # 3. Demonstrate evaluation
    print("3. Evaluation example:")
    
    # Sample translations (perfect matches for demo)
    predictions = ["Сәлам", "Ничек эшләр?", "Рәхмәт"]
    references = ["Сәлам", "Ничек эшләр?", "Рәхмәт"]
    
    evaluator = TranslationEvaluator()
    results = evaluator.evaluate_comprehensive(predictions, references)
    
    print("   Evaluation metrics:")
    key_metrics = ['exact_match', 'bleu_sentence_avg', 'char_f1']
    for metric in key_metrics:
        if metric in results:
            print(f"   - {metric}: {results[metric]:.3f}")
    
    print("\n4. Next steps:")
    print("   - Run training: python scripts/train.py --sample_dataset")
    print("   - Run evaluation: python scripts/eval_model.py --model_path ./results/model")
    print("   - Interactive translation: python scripts/inference.py --model_path ./results/model --interactive")
    print("   - See full demo: python scripts/demo.py")


if __name__ == "__main__":
    basic_example()