"""
Upload PhoBERT Sentiment Classifier to HuggingFace Hub

Usage:
    python training/upload_phobert_sentiment_to_huggingface.py
"""

import os
import sys
import json
import torch

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from huggingface_hub import create_repo, upload_file

# Configuration
SENTIMENTS = ['positive', 'neutral', 'negative']
PHOBERT_MODEL_PATH = "training/models/phobert_sentiment_best_model.pt"
HF_USERNAME = "Strawberry0604"

def create_model_card():
    """Create README.md for PhoBERT Sentiment model"""
    
    # Load model stats
    checkpoint = torch.load(PHOBERT_MODEL_PATH, map_location='cpu')
    f1_macro = checkpoint.get('f1_macro', 0.7959)
    f1_weighted = checkpoint.get('f1_weighted', 0.8726)
    accuracy = checkpoint.get('accuracy', 0.8725)
    
    return f"""---
language:
- vi
license: mit
tags:
- text-classification
- sentiment-analysis
- vietnamese
- tourism
- phobert
datasets:
- custom
metrics:
- f1
- accuracy
model-index:
- name: PhoBERT Tourism Sentiment Classifier
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

# PhoBERT Tourism Sentiment Classifier

Fine-tuned PhoBERT model for Vietnamese tourism comment sentiment analysis.

## Model Description

This model classifies Vietnamese tourism comments into 3 sentiment categories:
- **positive** (Tích cực)
- **neutral** (Trung lập)
- **negative** (Tiêu cực)

## Training Data

- **Language**: Vietnamese only
- **Dataset**: 11,371 Vietnamese tourism comments from social media
- **Sources**: Google Maps, TikTok, YouTube, Facebook
- **Split**: 80% train (9,096), 20% validation (2,275)
- **Quality**: Filtered for meaningful content (min 10 words)

### Sentiment Distribution

| Sentiment | Count | Percentage |
|-----------|-------|------------|
| Positive | 7,669 | 67.4% |
| Neutral | 1,894 | 16.7% |
| Negative | 1,808 | 15.9% |

## Performance

| Metric | Score |
|--------|-------|
| **Accuracy** | **{accuracy*100:.2f}%** |
| **F1 Macro** | **{f1_macro*100:.2f}%** |
| **F1 Weighted** | **{f1_weighted*100:.2f}%** |

### Per-Class Performance

| Sentiment | Precision | Recall | F1-Score | Support |
|-----------|-----------|--------|----------|---------|
| Positive | 94.36% | 94.85% | 94.60% | 1,534 |
| Neutral | 64.24% | 79.16% | 70.92% | 379 |
| Negative | 86.47% | 63.54% | 73.25% | 362 |

### Confusion Matrix

```
              Predicted
              pos  neu  neg
Actual  pos  1455  63   16
        neu    59  300  20
        neg    28  104  230
```

## Usage

```python
import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModel

# Define model architecture
class PhoBERTSentimentClassifier(nn.Module):
    def __init__(self, n_classes=3, dropout=0.3):
        super(PhoBERTSentimentClassifier, self).__init__()
        self.phobert = AutoModel.from_pretrained('vinai/phobert-base')
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(self.phobert.config.hidden_size, n_classes)
        
    def forward(self, input_ids, attention_mask):
        outputs = self.phobert(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs[1]
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)
        return logits

# Load model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
tokenizer = AutoTokenizer.from_pretrained('vinai/phobert-base')

model = PhoBERTSentimentClassifier(n_classes=3)
checkpoint = torch.load('phobert_sentiment_best_model.pt', map_location=device)
model.load_state_dict(checkpoint['model_state_dict'])
model = model.to(device)
model.eval()

# Predict
text = "Bãi biển đẹp quá, tôi rất thích!"
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
# Output: Sentiment: positive (0.9965)
```

## Training Details

- **Base Model**: vinai/phobert-base
- **Architecture**: PhoBERT + Dropout (0.3) + Linear (768 → 3)
- **Parameters**: ~135M (base) + 2,307 (classifier)
- **Training Time**: ~20-25 minutes on CUDA
- **Epochs**: 5 (best at epoch 3)
- **Batch Size**: 16
- **Learning Rate**: 2e-5
- **Optimizer**: AdamW with linear warmup
- **Loss**: CrossEntropyLoss

### Training Progress

| Epoch | Train Loss | Train Acc | Val Loss | Val Acc | F1 Macro |
|-------|------------|-----------|----------|---------|----------|
| 1 | 0.6369 | 73.76% | 0.4033 | 84.44% | 75.33% |
| 2 | 0.3697 | 86.77% | 0.3681 | 86.59% | 78.88% |
| **3** | **0.2565** | **91.60%** | **0.3863** | **87.25%** | **79.59%** ⭐ |
| 4 | 0.1857 | 94.49% | 0.5379 | 87.12% | 79.47% |
| 5 | 0.1369 | 96.19% | 0.5483 | 86.73% | 79.00% |

## Features

✅ **High Accuracy**: 87.25% overall accuracy
✅ **Excellent for Positive**: 94.6% F1 for positive sentiment
✅ **Balanced Performance**: Good handling of all sentiment classes
✅ **Fast Inference**: ~50ms per comment on GPU
✅ **Production Ready**: Used in real-world tourism monitoring system

## Limitations

- Vietnamese text only (not multilingual)
- Trained on tourism domain (may not generalize to other domains)
- Slightly lower performance on neutral/negative classes due to data imbalance
- Requires GPU for optimal inference speed

## Use Cases

- Tourism review sentiment analysis
- Social media monitoring for tourism destinations
- Customer feedback analysis for hotels/attractions
- Tourism demand analysis
- Automated content moderation

## Citation

```bibtex
@misc{{phobert-tourism-sentiment,
  author = {{Strawberry0604}},
  title = {{PhoBERT Tourism Sentiment Classifier}},
  year = {{2025}},
  publisher = {{HuggingFace}},
  howpublished = {{\\url{{https://huggingface.co/{HF_USERNAME}/phobert-tourism-sentiment}}}}
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
        "model_type": "phobert-sentiment",
        "task": "sentiment-analysis",
        "num_labels": 3,
        "id2label": {str(i): label for i, label in enumerate(SENTIMENTS)},
        "label2id": {label: i for i, label in enumerate(SENTIMENTS)},
        "problem_type": "single_label_classification",
        "max_length": 256,
        "dropout": 0.3,
        "base_model": "vinai/phobert-base",
        "language": "vi",
        "architecture": "PhoBERT + Dropout + Linear",
        "parameters": {
            "total": "~135M",
            "trainable": "2,307"
        }
    }
    return config

def main():
    """Main upload function"""
    print("\n" + "="*80)
    print("UPLOAD PHOBERT SENTIMENT MODEL TO HUGGINGFACE")
    print("="*80)
    
    # Check if model exists
    if not os.path.exists(PHOBERT_MODEL_PATH):
        print(f"\nERROR: PhoBERT sentiment model not found at {PHOBERT_MODEL_PATH}")
        print("Please train the model first: python training/train_phobert_sentiment.py")
        return
    
    # Initialize API
    print("\nInitializing HuggingFace API...")
    try:
        repo_name = f"{HF_USERNAME}/phobert-tourism-sentiment"
        
        # Create repository
        print(f"\nCreating repository: {repo_name}")
        try:
            create_repo(repo_name, exist_ok=True, private=False)
            print("✓ Repository created/verified successfully!")
        except Exception as e:
            print(f"✓ Repository already exists: {e}")
        
        # Upload model file
        model_size = os.path.getsize(PHOBERT_MODEL_PATH) / (1024**2)  # MB
        print(f"\n[1/4] Uploading model file ({model_size:.2f} MB)...")
        upload_file(
            path_or_fileobj=PHOBERT_MODEL_PATH,
            path_in_repo="phobert_sentiment_best_model.pt",
            repo_id=repo_name,
            repo_type="model"
        )
        print("✓ Model file uploaded!")
        
        # Create and upload README
        print("\n[2/4] Creating and uploading README.md...")
        readme_content = create_model_card()
        with open("temp_readme_phobert_sentiment.md", "w", encoding="utf-8") as f:
            f.write(readme_content)
        
        upload_file(
            path_or_fileobj="temp_readme_phobert_sentiment.md",
            path_in_repo="README.md",
            repo_id=repo_name,
            repo_type="model"
        )
        os.remove("temp_readme_phobert_sentiment.md")
        print("✓ README uploaded!")
        
        # Create and upload config
        print("\n[3/4] Creating and uploading config.json...")
        config = create_config_json()
        with open("temp_config_phobert_sentiment.json", "w") as f:
            json.dump(config, f, indent=2)
        
        upload_file(
            path_or_fileobj="temp_config_phobert_sentiment.json",
            path_in_repo="config.json",
            repo_id=repo_name,
            repo_type="model"
        )
        os.remove("temp_config_phobert_sentiment.json")
        print("✓ Config uploaded!")
        
        # Upload sentiment statistics
        stats_path = "training/models/phobert_sentiment_statistics.csv"
        if os.path.exists(stats_path):
            print("\n[4/4] Uploading sentiment statistics...")
            upload_file(
                path_or_fileobj=stats_path,
                path_in_repo="sentiment_statistics.csv",
                repo_id=repo_name,
                repo_type="model"
            )
            print("✓ Statistics uploaded!")
        
        print("\n" + "="*80)
        print("✅ PHOBERT SENTIMENT MODEL UPLOADED SUCCESSFULLY!")
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
