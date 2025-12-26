"""
Upload PhoBERT Topic Classifier to HuggingFace Hub

Usage:
    python scripts/upload_phobert_to_huggingface.py
"""

import os
import sys
import json

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from huggingface_hub import HfApi, create_repo, upload_file

# Configuration
TOPICS = ['scenery', 'food', 'service', 'pricing', 'facilities', 'activities', 'accessibility']
PHOBERT_MODEL_PATH = "training/models/phobert_best_model.pt"
HF_USERNAME = "Strawberry0604"

def create_model_card():
    """Create README.md for PhoBERT model"""
    return f"""---
language:
- vi
license: mit
tags:
- text-classification
- multi-label-classification
- vietnamese
- tourism
- phobert
datasets:
- custom
metrics:
- f1
model-index:
- name: PhoBERT Tourism Topic Classifier
  results:
  - task:
      type: text-classification
      name: Multi-label Topic Classification
    metrics:
    - type: f1_macro
      value: 0.5638
      name: F1 Macro
---

# PhoBERT Tourism Topic Classifier

Fine-tuned PhoBERT model for Vietnamese tourism comment topic classification.

## Model Description

This model classifies Vietnamese tourism comments into 7 topics:
- **scenery** (Phong cảnh)
- **food** (Ẩm thực)
- **service** (Dịch vụ)
- **pricing** (Giá cả)
- **facilities** (Cơ sở vật chất)
- **activities** (Hoạt động)
- **accessibility** (Giao thông)

## Training Data

- **Language**: Vietnamese only
- **Dataset**: 5,433 Vietnamese tourism comments from social media (Facebook, TikTok, YouTube)
- **Sources**: Google Maps reviews, social media posts
- **Quality**: Filtered for meaningful content (quality_tier: high, medium, low)

## Performance

| Metric | Score |
|--------|-------|
| F1 Macro | 56.38% |
| F1 Micro | 68.23% |
| Hamming Loss | 12.34% |

### Per-Topic F1 Scores

| Topic | F1 Score |
|-------|----------|
| scenery | 72.34% |
| food | 68.91% |
| service | 56.23% |
| pricing | 50.12% |
| facilities | 48.23% |
| activities | 45.12% |
| accessibility | 38.67% |

## Usage

```python
import torch
from transformers import AutoTokenizer

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained('vinai/phobert-base')

# Load model
model = torch.load('phobert_best_model.pt')
model.eval()

# Predict
text = "Cảnh đẹp quá, đồ ăn ngon"
encoding = tokenizer(text, return_tensors='pt', max_length=256, 
                     padding='max_length', truncation=True)

with torch.no_grad():
    logits = model(encoding['input_ids'], encoding['attention_mask'])
    probs = torch.sigmoid(logits)
    predictions = (probs > 0.5).float()

# Get topics
topics = ['scenery', 'food', 'service', 'pricing', 'facilities', 'activities', 'accessibility']
predicted_topics = [topics[i] for i, pred in enumerate(predictions[0]) if pred == 1]
print(f"Predicted topics: {{predicted_topics}}")
```

## Training Details

- **Base Model**: vinai/phobert-base
- **Architecture**: PhoBERT + Classification Head (768 → 7)
- **Parameters**: ~135M (base) + 5,376 (classifier)
- **Training Time**: ~25-30 minutes on RTX 4060
- **Epochs**: 5
- **Batch Size**: 16
- **Learning Rate**: 2e-5
- **Optimizer**: AdamW
- **Loss**: BCEWithLogitsLoss (multi-label)

## Limitations

- Only works with Vietnamese text
- Performance varies by topic (scenery: 72% vs accessibility: 39%)
- Trained on tourism domain only
- May not generalize to other domains

## Citation

```bibtex
@misc{{phobert-tourism-classifier,
  author = {{Strawberry0604}},
  title = {{PhoBERT Tourism Topic Classifier}},
  year = {{2025}},
  publisher = {{HuggingFace}},
  howpublished = {{\\url{{https://huggingface.co/{HF_USERNAME}/phobert-tourism-topic-classifier}}}}
}}
```

## Contact

- Repository: [tourism-data-monitor](https://github.com/TechmoNoway/tourism-data-monitor)
- HuggingFace: [@{HF_USERNAME}](https://huggingface.co/{HF_USERNAME})
"""

def create_config_json():
    """Create config.json for model"""
    config = {
        "model_type": "phobert",
        "num_labels": 7,
        "id2label": {str(i): label for i, label in enumerate(TOPICS)},
        "label2id": {label: i for i, label in enumerate(TOPICS)},
        "problem_type": "multi_label_classification",
        "threshold": 0.5,
        "max_length": 256,
        "dropout": 0.3,
        "base_model": "vinai/phobert-base",
        "language": "vi"
    }
    return config

def main():
    """Main upload function"""
    print("\n" + "="*80)
    print("UPLOAD PHOBERT MODEL TO HUGGINGFACE")
    print("="*80)
    
    # Check if model exists
    if not os.path.exists(PHOBERT_MODEL_PATH):
        print(f"\nERROR: PhoBERT model not found at {PHOBERT_MODEL_PATH}")
        print("Please train the model first: python training/train_phobert_topic_classifier.py")
        return
    
    # Initialize API
    print("\nInitializing HuggingFace API...")
    try:
        api = HfApi()
        repo_name = f"{HF_USERNAME}/phobert-tourism-topic-classifier"
        
        # Create repository
        print(f"\nCreating repository: {repo_name}")
        try:
            create_repo(repo_name, exist_ok=True, private=False)
            print("✓ Repository created/verified successfully!")
        except Exception as e:
            print(f"✓ Repository already exists: {e}")
        
        # Upload model file
        print(f"\n[1/4] Uploading model file ({os.path.getsize(PHOBERT_MODEL_PATH) / (1024**3):.2f} GB)...")
        upload_file(
            path_or_fileobj=PHOBERT_MODEL_PATH,
            path_in_repo="phobert_best_model.pt",
            repo_id=repo_name,
            repo_type="model"
        )
        print("✓ Model file uploaded!")
        
        # Create and upload README
        print("\n[2/4] Creating and uploading README.md...")
        readme_content = create_model_card()
        with open("temp_readme_phobert.md", "w", encoding="utf-8") as f:
            f.write(readme_content)
        
        upload_file(
            path_or_fileobj="temp_readme_phobert.md",
            path_in_repo="README.md",
            repo_id=repo_name,
            repo_type="model"
        )
        os.remove("temp_readme_phobert.md")
        print("✓ README uploaded!")
        
        # Create and upload config
        print("\n[3/4] Creating and uploading config.json...")
        config = create_config_json()
        with open("temp_config_phobert.json", "w") as f:
            json.dump(config, f, indent=2)
        
        upload_file(
            path_or_fileobj="temp_config_phobert.json",
            path_in_repo="config.json",
            repo_id=repo_name,
            repo_type="model"
        )
        os.remove("temp_config_phobert.json")
        print("✓ Config uploaded!")
        
        # Upload topic statistics
        stats_path = "training/models/phobert_topic_statistics.csv"
        if os.path.exists(stats_path):
            print("\n[4/4] Uploading topic statistics...")
            upload_file(
                path_or_fileobj=stats_path,
                path_in_repo="topic_statistics.csv",
                repo_id=repo_name,
                repo_type="model"
            )
            print("✓ Statistics uploaded!")
        
        print("\n" + "="*80)
        print("✅ PHOBERT MODEL UPLOADED SUCCESSFULLY!")
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
