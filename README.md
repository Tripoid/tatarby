# Tatarby - Russian to Tatar Translation System

![License](https://img.shields.io/badge/license-MIT-blue.svg)
![Python](https://img.shields.io/badge/python-3.8%2B-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-orange.svg)
![Transformers](https://img.shields.io/badge/🤗-Transformers-yellow.svg)

A high-quality neural machine translation system for Russian-to-Tatar language translation, built with state-of-the-art transformer architectures including T5 and MarianMT.

## 🚀 Features

- **Multiple Architectures**: Support for T5, MarianMT, and other transformer models
- **Comprehensive Evaluation**: BLEU, ROUGE, and custom metrics
- **Easy Training**: Simple scripts for training and fine-tuning
- **Production Ready**: Inference scripts for batch and interactive translation
- **Configurable**: YAML-based configuration system
- **Well Documented**: Comprehensive documentation and examples

## 📋 Table of Contents

- [Installation](#-installation)
- [Quick Start](#-quick-start)
- [Dataset](#-dataset)
- [Training](#-training)
- [Evaluation](#-evaluation)
- [Inference](#-inference)
- [Configuration](#-configuration)
- [Project Structure](#-project-structure)
- [Documentation](#-documentation)
- [Contributing](#-contributing)
- [License](#-license)

## 🛠 Installation

### Prerequisites

- Python 3.8 or higher
- PyTorch 2.0+
- CUDA (optional, for GPU acceleration)

### Install Dependencies

```bash
git clone https://github.com/Tripoid/tatarby.git
cd tatarby
pip install -r requirements.txt
```

### Quick Test

```bash
python -c "import torch; print('PyTorch version:', torch.__version__)"
python -c "import transformers; print('Transformers version:', transformers.__version__)"
```

## 🚀 Quick Start

### 1. Training a Model

```bash
# Train with sample dataset (for testing)
python scripts/train.py --sample_dataset --num_epochs 2

# Train with custom configuration
python scripts/train.py --config configs/config.yaml
```

### 2. Evaluating the Model

```bash
python scripts/evaluate.py --model_path ./results/model --sample_dataset
```

### 3. Using for Translation

```bash
# Interactive mode
python scripts/inference.py --model_path ./results/model --interactive

# Single translation
python scripts/inference.py --model_path ./results/model --text "Привет, как дела?"
```

## 📊 Dataset

The system is designed to work with the `IPSAN/tatar_translation_dataset` from Hugging Face. However, due to potential connectivity issues, a sample dataset is included for demonstration purposes.

### Expected Dataset Format

```python
{
    "train": {
        "russian": ["Привет", "Как дела?", ...],
        "tatar": ["Сәлам", "Ничек эшләр?", ...]
    },
    "validation": {...},
    "test": {...}
}
```

### Using Your Own Dataset

To use a custom dataset, modify the `load_tatar_dataset` function in `src/data_preprocessing.py` or format your data according to the expected structure.

## 🎯 Training

### Basic Training

```bash
python scripts/train.py \
    --model_name t5-small \
    --num_epochs 5 \
    --batch_size 8 \
    --learning_rate 5e-5 \
    --sample_dataset
```

### Advanced Training

```bash
python scripts/train.py \
    --config configs/config.yaml \
    --output_dir ./my_experiment
```

### Training Parameters

- `--model_name`: Pretrained model (t5-small, t5-base, t5-large)
- `--architecture`: Model type (t5, marian, auto)
- `--num_epochs`: Number of training epochs
- `--batch_size`: Training batch size
- `--learning_rate`: Learning rate
- `--sample_dataset`: Use built-in sample dataset

## 📈 Evaluation

### Comprehensive Evaluation

```bash
python scripts/evaluate.py \
    --model_path ./results/model \
    --split test \
    --max_samples 1000
```

### Metrics Included

- **BLEU Score**: Industry standard translation metric
- **ROUGE Score**: Recall-oriented evaluation
- **Exact Match**: Percentage of perfect translations
- **Character-level Metrics**: Precision, recall, F1
- **Length Analysis**: Translation length comparison

## 🔮 Inference

### Interactive Translation

```bash
python scripts/inference.py \
    --model_path ./results/model \
    --interactive
```

### Batch Translation

```bash
# Create input file with Russian texts (one per line)
echo -e "Привет\nКак дела?\nСпасибо" > input.txt

# Translate
python scripts/inference.py \
    --model_path ./results/model \
    --input_file input.txt \
    --output_file output.txt
```

### API-style Usage

```python
from src.model import TatarTranslationModel

# Load model
model = TatarTranslationModel()
model.load_model('./results/model')

# Translate
translations = model.translate([
    "Привет",
    "Как дела?",
    "Спасибо"
])

print(translations)
```

## ⚙️ Configuration

The system uses YAML configuration files for easy customization:

```yaml
# configs/config.yaml
model:
  architecture: "t5"
  model_name: "t5-small"
  max_length: 128

training:
  num_epochs: 5
  batch_size: 8
  learning_rate: 5e-5

data:
  dataset_name: "IPSAN/tatar_translation_dataset"
  source_lang: "russian"
  target_lang: "tatar"
```

See `configs/config.yaml` for complete configuration options.

## 📁 Project Structure

```
tatarby/
├── src/                          # Core source code
│   ├── __init__.py
│   ├── data_preprocessing.py     # Dataset loading and preprocessing
│   ├── model.py                  # Translation model implementation
│   ├── evaluation.py             # Evaluation metrics and utilities
│   └── utils.py                  # General utilities
├── scripts/                      # Executable scripts
│   ├── train.py                  # Training script
│   ├── evaluate.py               # Evaluation script
│   └── inference.py              # Inference script
├── configs/                      # Configuration files
│   └── config.yaml               # Default configuration
├── docs/                         # Documentation
│   └── model_documentation.md    # Detailed model documentation
├── requirements.txt              # Python dependencies
└── README.md                     # This file
```

## 📚 Documentation

- [Model Documentation](docs/model_documentation.md): Detailed technical documentation
- [Configuration Guide](configs/config.yaml): Configuration options
- [API Reference](src/): Source code documentation

## 🎯 Performance

### Expected Results

For demonstration with sample data:
- **BLEU Score**: 85-95 (very high due to sample nature)
- **ROUGE-L**: 0.90-0.95
- **Exact Match**: 80-90%

For real-world datasets:
- **BLEU Score**: 25-35 (good quality)
- **ROUGE-L**: 0.45-0.55
- **Exact Match**: 10-20%

### Hardware Requirements

- **Minimum**: 8GB RAM, CPU training
- **Recommended**: 16GB+ RAM, NVIDIA GPU with 8GB+ VRAM
- **Optimal**: 32GB+ RAM, NVIDIA GPU with 16GB+ VRAM

## 🤝 Contributing

We welcome contributions! Please see our [Contributing Guidelines](CONTRIBUTING.md) for details.

### Areas for Contribution

- Model architecture improvements
- Dataset expansion and quality improvement
- Evaluation metric enhancements
- Documentation and examples
- Performance optimizations

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- Hugging Face Transformers library
- Google T5 model
- Helsinki-NLP MarianMT
- The Tatar language community

## 📞 Support

For questions, issues, or support:

1. Check the [documentation](docs/model_documentation.md)
2. Search existing [issues](https://github.com/Tripoid/tatarby/issues)
3. Create a new issue if needed

---

**Note**: This system is designed for research and educational purposes. For production use, please ensure adequate testing and validation with real-world data.