"""
Inference script for Russian-Tatar translation model.
"""

import argparse
import sys
import os
from pathlib import Path

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.model import TatarTranslationModel
from src.utils import load_config, set_seed
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Perform inference with Russian-Tatar translation model"
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
        '--text', 
        type=str, 
        default=None,
        help='Russian text to translate'
    )
    
    parser.add_argument(
        '--input_file', 
        type=str, 
        default=None,
        help='File containing Russian texts to translate (one per line)'
    )
    
    parser.add_argument(
        '--output_file', 
        type=str, 
        default=None,
        help='File to save translations'
    )
    
    parser.add_argument(
        '--interactive', 
        action='store_true',
        help='Run in interactive mode'
    )
    
    parser.add_argument(
        '--num_beams', 
        type=int, 
        default=4,
        help='Number of beams for beam search'
    )
    
    parser.add_argument(
        '--max_new_tokens', 
        type=int, 
        default=128,
        help='Maximum number of tokens to generate'
    )
    
    parser.add_argument(
        '--temperature', 
        type=float, 
        default=1.0,
        help='Sampling temperature'
    )
    
    parser.add_argument(
        '--do_sample', 
        action='store_true',
        help='Use sampling instead of greedy decoding'
    )
    
    parser.add_argument(
        '--seed', 
        type=int, 
        default=42,
        help='Random seed'
    )
    
    return parser.parse_args()


def load_model_with_config(model_path: str, config_path: str = None) -> TatarTranslationModel:
    """Load model with configuration."""
    
    # Load configuration if provided
    config = None
    if config_path and Path(config_path).exists():
        config = load_config(config_path)
        logger.info(f"Loaded configuration from {config_path}")
    
    # Determine model architecture from config or try to infer
    architecture = 't5'  # Default
    model_name = 't5-small'  # Default
    
    if config:
        architecture = config.get('model', {}).get('architecture', 't5')
        model_name = config.get('model', {}).get('model_name', 't5-small')
    
    # Initialize model
    model = TatarTranslationModel(
        model_name=model_name,
        architecture=architecture
    )
    
    # Load the trained weights
    model.load_model(model_path)
    
    return model


def translate_text(model: TatarTranslationModel, text: str, **generation_kwargs) -> str:
    """Translate a single text."""
    translations = model.translate([text], **generation_kwargs)
    return translations[0]


def translate_file(model: TatarTranslationModel, input_file: str, output_file: str = None, **generation_kwargs):
    """Translate texts from a file."""
    
    # Read input texts
    with open(input_file, 'r', encoding='utf-8') as f:
        texts = [line.strip() for line in f.readlines() if line.strip()]
    
    logger.info(f"Translating {len(texts)} texts from {input_file}")
    
    # Translate
    translations = model.translate(texts, **generation_kwargs)
    
    # Save or print results
    if output_file:
        with open(output_file, 'w', encoding='utf-8') as f:
            for translation in translations:
                f.write(translation + '\n')
        logger.info(f"Translations saved to {output_file}")
    else:
        for i, (text, translation) in enumerate(zip(texts, translations)):
            print(f"--- Translation {i+1} ---")
            print(f"RU:  {text}")
            print(f"TAT: {translation}")
            print()


def interactive_mode(model: TatarTranslationModel, **generation_kwargs):
    """Run in interactive mode."""
    
    print("Russian-Tatar Translation Interactive Mode")
    print("Enter Russian text to translate (type 'quit' to exit):")
    print()
    
    while True:
        try:
            # Get input
            text = input("RU: ").strip()
            
            if text.lower() in ['quit', 'exit', 'q']:
                break
            
            if not text:
                continue
            
            # Translate
            translation = translate_text(model, text, **generation_kwargs)
            print(f"TAT: {translation}")
            print()
            
        except KeyboardInterrupt:
            print("\nExiting...")
            break
        except Exception as e:
            print(f"Error: {e}")
            continue


def main():
    """Main inference function."""
    
    # Parse arguments
    args = parse_args()
    
    logger.info("Starting Russian-Tatar translation inference")
    logger.info(f"Model path: {args.model_path}")
    
    # Set random seed
    set_seed(args.seed)
    
    try:
        # Load model
        logger.info("Loading model...")
        model = load_model_with_config(args.model_path, args.config_path)
        logger.info("Model loaded successfully")
        
        # Prepare generation kwargs
        generation_kwargs = {
            'num_beams': args.num_beams,
            'max_new_tokens': args.max_new_tokens,
            'temperature': args.temperature,
            'do_sample': args.do_sample
        }
        
        # Determine mode
        if args.interactive:
            # Interactive mode
            interactive_mode(model, **generation_kwargs)
            
        elif args.input_file:
            # File mode
            if not Path(args.input_file).exists():
                raise FileNotFoundError(f"Input file not found: {args.input_file}")
            
            translate_file(
                model, 
                args.input_file, 
                args.output_file, 
                **generation_kwargs
            )
            
        elif args.text:
            # Single text mode
            translation = translate_text(model, args.text, **generation_kwargs)
            print(f"Russian: {args.text}")
            print(f"Tatar:   {translation}")
            
        else:
            # Default interactive mode
            logger.info("No input specified, running in interactive mode")
            interactive_mode(model, **generation_kwargs)
        
        logger.info("Inference completed successfully!")
        
    except Exception as e:
        logger.error(f"Inference failed: {e}")
        import traceback
        traceback.print_exc()
        raise e


if __name__ == "__main__":
    main()