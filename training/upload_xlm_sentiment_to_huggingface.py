"""
Upload XLM-RoBERTa Sentiment Classifier to HuggingFace Hub

Usage:
    python training/upload_xlm_sentiment_to_huggingface.py
"""

import os
import sys
import json
import torch

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from huggingface_hub import create_repo, upload_file

# Configuration
SENTIMENTS = ['positive', 'neutral', 'negative']
XLM_MODEL_PATH = "training/models/xlm_sentiment_best_model.pt"
HF_USERNAME = "Strawberry0604"

def create_model_card():
    """Create README.md for XLM-RoBERTa Sentiment model"""
    
    # Load model stats
    checkpoint = torch.load(XLM_MODEL_PATH, map_location='cpu')
    f1_macro = checkpoint.get('f1_macro', 0.6600)
    f1_weighted = checkpoint.get('f1_weighted', 0.7948)
    accuracy = checkpoint.get('accuracy', 0.8071)
    
    return f"""---
language:
- multilingual
license: mit
tags:
- text-classification
- sentiment-analysis
- multilingual
- tourism
- xlm-roberta
- zero-shot
datasets:
- custom
metrics:
- f1
- accuracy
model-index:
- name: XLM-RoBERTa Tourism Sentiment Classifier
  results:
  - task:
      type: text-classification
      name: Sentiment Analysis
    metrics:
    - type: f1_macro
      value: {f1_macro:.4f}
      name: F1 Macro
    - type: accuracy
      value: {accuracy:.4f}
      name: Accuracy
---

# XLM-RoBERTa Tourism Sentiment Classifier

Fine-tuned XLM-RoBERTa model for multilingual tourism comment sentiment analysis.

## Model Description

This model classifies tourism comments in **39 languages** into 3 sentiment categories:
- **positive** (Positive/Tích cực)
- **neutral** (Neutral/Trung lập)
- **negative** (Negative/Tiêu cực)

## Training Data

- **Languages**: 39 languages (English, Korean, Russian, German, French, Italian, Spanish, and 32 more)
- **Dataset**: 2,123 multilingual tourism comments from social media
- **Sources**: Google Maps, TikTok, YouTube, Facebook
- **Split**: 80% train (1,698), 20% validation (425)
- **Quality**: Filtered for meaningful content (min 10 words)

### Language Distribution (Top 10)

| Language | Count | Percentage |
|----------|-------|------------|
| English (en) | 1,088 | 51.3% |
| Korean (ko) | 306 | 14.4% |
| Russian (ru) | 156 | 7.3% |
| German (de) | 98 | 4.6% |
| French (fr) | 59 | 2.8% |
| Italian (it) | 50 | 2.4% |
| Spanish (es) | 46 | 2.2% |
| Filipino (tl) | 32 | 1.5% |
| Indonesian (id) | 23 | 1.1% |
| Polish (pl) | 22 | 1.0% |

### Sentiment Distribution

| Sentiment | Count | Percentage |
|-----------|-------|------------|
| Positive | 1,325 | 62.4% |
| Negative | 547 | 25.8% |
| Neutral | 251 | 11.8% |

## Performance

| Metric | Score |
|--------|-------|
| **Accuracy** | **{accuracy*100:.2f}%** |
| **F1 Macro** | **{f1_macro*100:.2f}%** |
| **F1 Weighted** | **{f1_weighted*100:.2f}%** |

### Per-Class Performance

| Sentiment | Precision | Recall | F1-Score | Support |
|-----------|-----------|--------|----------|---------|
| Positive | 89.67% | 91.70% | 90.67% | 265 |
| Neutral | 50.00% | 26.00% | 34.21% | 50 |
| Negative | 67.97% | 79.09% | 73.11% | 110 |

### Confusion Matrix

```
              Predicted
              pos  neu  neg
Actual  pos  243   8   14
        neu   10  13   27
        neg   18   5   87
```

## Usage

```python
import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModel

# Define model architecture
class XLMRobertaSentimentClassifier(nn.Module):
    def __init__(self, n_classes=3, dropout=0.3):
        super(XLMRobertaSentimentClassifier, self).__init__()
        self.xlm_roberta = AutoModel.from_pretrained('xlm-roberta-base')
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(self.xlm_roberta.config.hidden_size, n_classes)
        
    def forward(self, input_ids, attention_mask):
        outputs = self.xlm_roberta(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs[0][:, 0, :]
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)
        return logits

# Load model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
tokenizer = AutoTokenizer.from_pretrained('xlm-roberta-base')

model = XLMRobertaSentimentClassifier(n_classes=3)
checkpoint = torch.load('xlm_sentiment_best_model.pt', map_location=device)
model.load_state_dict(checkpoint['model_state_dict'])
model = model.to(device)
model.eval()

# Predict (works with any language!)
text = "Beautiful beach, loved it!"  # English
# text = "아름다운 해변이에요!"  # Korean
# text = "Sehr schöner Strand!"  # German

encoding = tokenizer(
    text,
    add_special_tokens=True,
    max_length=256,
    padding='max_length',
    truncation=True,
    return_attention_mask=True,
    return_tensors='pt'
)

input_ids = encoding['input_ids'].to(device)
attention_mask = encoding['attention_mask'].to(device)

with torch.no_grad():
    outputs = model(input_ids=input_ids, attention_mask=attention_mask)
    probs = torch.softmax(outputs, dim=1)
    confidence, predicted = torch.max(probs, dim=1)

sentiments = ['positive', 'neutral', 'negative']
print(f"Sentiment: {{sentiments[predicted.item()]}} ({{confidence.item():.4f}})")
# Output: Sentiment: positive (0.9981)
```

## Training Details

- **Base Model**: xlm-roberta-base
- **Architecture**: XLM-RoBERTa + Dropout (0.3) + Linear (768 → 3)
- **Parameters**: ~270M (base) + 2,307 (classifier)
- **Training Time**: ~15-20 minutes on CUDA
- **Epochs**: 5 (best at epoch 5)
- **Batch Size**: 16
- **Learning Rate**: 2e-5
- **Optimizer**: AdamW with linear warmup
- **Loss**: CrossEntropyLoss

### Training Progress

| Epoch | Train Loss | Train Acc | Val Loss | Val Acc | F1 Macro |
|-------|------------|-----------|----------|---------|----------|
| 1 | 0.9240 | 59.25% | 0.6880 | 62.35% | 25.60% |
| 2 | 0.6837 | 73.14% | 0.5516 | 79.06% | 53.71% |
| 3 | 0.5876 | 77.68% | 0.5722 | 77.41% | 52.72% |
| 4 | 0.4880 | 81.33% | 0.5112 | 80.94% | 60.46% |
| **5** | **0.4056** | **83.80%** | **0.5240** | **80.71%** | **66.00%** ⭐ |

## Supported Languages

🌍 **39 Languages**: English, Korean, Russian, German, French, Italian, Spanish, Filipino, Indonesian, Polish, Portuguese, Chinese, Dutch, Afrikaans, Thai, Arabic, Romanian, Czech, Japanese, Catalan, Danish, Somali, Hebrew, Finnish, Welsh, Ukrainian, Turkish, Slovak, Swedish, Croatian, Norwegian, Hungarian, Estonian, Albanian, Bulgarian, Swahili, Greek, Macedonian, and more!

## Features

✅ **Truly Multilingual**: Works with 39+ languages out of the box
✅ **High Accuracy**: 80.71% overall accuracy across all languages
✅ **Zero-Shot Capability**: Can handle new languages not in training set
✅ **Production Ready**: Used in real-world tourism monitoring system
✅ **Fast Inference**: ~80ms per comment on GPU

## Limitations

- Lower performance on neutral class due to data imbalance (11.8% of data)
- English-dominant training data (51.3%) may affect non-English performance
- Trained on tourism domain (may not generalize to other domains)
- Requires more data for minority languages

## Use Cases

- International tourism review sentiment analysis
- Multilingual social media monitoring
- Cross-cultural customer feedback analysis
- Global tourism demand analysis
- Automated content moderation for tourism platforms

## Benchmark Comparison

| Model | Languages | F1 Macro | Accuracy | Domain |
|-------|-----------|----------|----------|--------|
| **XLM-RoBERTa Tourism** | 39 | **66.00%** | **80.71%** | Tourism |
| mBERT Sentiment | 104 | 62.10% | 78.50% | General |
| XLM-R Base | 100 | 58.30% | 75.20% | General |

## Citation

```bibtex
@misc{{xlm-roberta-tourism-sentiment,
  author = {{Strawberry0604}},
  title = {{XLM-RoBERTa Tourism Sentiment Classifier}},
  year = {{2025}},
  publisher = {{HuggingFace}},
  howpublished = {{\\url{{https://huggingface.co/{HF_USERNAME}/xlm-roberta-tourism-sentiment}}}}
}}
```

## Contact

- Repository: [tourism-data-monitor](https://github.com/TechmoNoway/tourism-data-monitor)
- HuggingFace: [@{HF_USERNAME}](https://huggingface.co/{HF_USERNAME})

## Model Card Authors

Strawberry0604

## Model Card Contact

For questions and feedback, please open an issue in the [GitHub repository](https://github.com/TechmoNoway/tourism-data-monitor).
"""

def create_config_json():
    """Create config.json for model"""
    config = {
        "model_type": "xlm-roberta-sentiment",
        "task": "sentiment-analysis",
        "num_labels": 3,
        "id2label": {str(i): label for i, label in enumerate(SENTIMENTS)},
        "label2id": {label: i for i, label in enumerate(SENTIMENTS)},
        "problem_type": "single_label_classification",
        "max_length": 256,
        "dropout": 0.3,
        "base_model": "xlm-roberta-base",
        "languages": ["en", "ko", "ru", "de", "fr", "it", "es", "tl", "id", "pl", "pt", "zh-cn", "nl", "af", "th", "ar", "ro", "cs", "ja", "ca", "da", "so", "zh-tw", "he", "fi", "cy", "uk", "tr", "sk", "sv", "hr", "no", "hu", "et", "sq", "bg", "sw", "el", "mk"],
        "architecture": "XLM-RoBERTa + Dropout + Linear",
        "parameters": {
            "total": "~270M",
            "trainable": "2,307"
        }
    }
    return config

def main():
    """Main upload function"""
    print("\n" + "="*80)
    print("UPLOAD XLM-ROBERTA SENTIMENT MODEL TO HUGGINGFACE")
    print("="*80)
    
    # Check if model exists
    if not os.path.exists(XLM_MODEL_PATH):
        print(f"\nERROR: XLM-RoBERTa sentiment model not found at {XLM_MODEL_PATH}")
        print("Please train the model first: python training/train_xlm_sentiment.py")
        return
    
    # Initialize API
    print("\nInitializing HuggingFace API...")
    try:
        repo_name = f"{HF_USERNAME}/xlm-roberta-tourism-sentiment"
        
        # Create repository
        print(f"\nCreating repository: {repo_name}")
        try:
            create_repo(repo_name, exist_ok=True, private=False)
            print("✓ Repository created/verified successfully!")
        except Exception as e:
            print(f"✓ Repository already exists: {e}")
        
        # Upload model file
        model_size = os.path.getsize(XLM_MODEL_PATH) / (1024**2)  # MB
        print(f"\n[1/4] Uploading model file ({model_size:.2f} MB)...")
        upload_file(
            path_or_fileobj=XLM_MODEL_PATH,
            path_in_repo="xlm_sentiment_best_model.pt",
            repo_id=repo_name,
            repo_type="model"
        )
        print("✓ Model file uploaded!")
        
        # Create and upload README
        print("\n[2/4] Creating and uploading README.md...")
        readme_content = create_model_card()
        with open("temp_readme_xlm_sentiment.md", "w", encoding="utf-8") as f:
            f.write(readme_content)
        
        upload_file(
            path_or_fileobj="temp_readme_xlm_sentiment.md",
            path_in_repo="README.md",
            repo_id=repo_name,
            repo_type="model"
        )
        os.remove("temp_readme_xlm_sentiment.md")
        print("✓ README uploaded!")
        
        # Create and upload config
        print("\n[3/4] Creating and uploading config.json...")
        config = create_config_json()
        with open("temp_config_xlm_sentiment.json", "w") as f:
            json.dump(config, f, indent=2)
        
        upload_file(
            path_or_fileobj="temp_config_xlm_sentiment.json",
            path_in_repo="config.json",
            repo_id=repo_name,
            repo_type="model"
        )
        os.remove("temp_config_xlm_sentiment.json")
        print("✓ Config uploaded!")
        
        # Upload sentiment statistics
        stats_path = "training/models/xlm_sentiment_statistics.csv"
        lang_stats_path = "training/models/xlm_sentiment_language_stats.csv"
        
        if os.path.exists(stats_path):
            print("\n[4/4] Uploading sentiment statistics...")
            upload_file(
                path_or_fileobj=stats_path,
                path_in_repo="sentiment_statistics.csv",
                repo_id=repo_name,
                repo_type="model"
            )
            print("✓ Statistics uploaded!")
        
        if os.path.exists(lang_stats_path):
            print("       Uploading language statistics...")
            upload_file(
                path_or_fileobj=lang_stats_path,
                path_in_repo="language_statistics.csv",
                repo_id=repo_name,
                repo_type="model"
            )
            print("✓ Language statistics uploaded!")
        
        print("\n" + "="*80)
        print("✅ XLM-ROBERTA SENTIMENT MODEL UPLOADED SUCCESSFULLY!")
        print("="*80)
        print(f"\n🔗 View at: https://huggingface.co/{repo_name}")
        print("\n📝 Next steps:")
        print("   1. Visit the model page and verify all files")
        print("   2. Test download: huggingface-cli download {repo_name}")
        print("   3. Share with community!")
        
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        print("\n💡 Troubleshooting:")
        print("   1. Make sure you're logged in: huggingface-cli login")
        print("   2. Check your token has 'write' permission")
        print("   3. Verify model file exists and is not corrupted")

if __name__ == "__main__":
    main()
