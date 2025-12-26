"""
Upload XLM-RoBERTa Topic Classifier to HuggingFace Hub

Usage:
    python scripts/upload_xlm_to_huggingface.py
"""

import os
import sys
import json

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from huggingface_hub import HfApi, create_repo, upload_file

# Configuration
TOPICS = ['scenery', 'food', 'service', 'pricing', 'facilities', 'activities', 'accessibility']
XLM_MODEL_PATH = "training/models/xlm_best_model.pt"
HF_USERNAME = "Strawberry0604"

def create_model_card():
    """Create README.md for XLM-RoBERTa model"""
    return f"""---
language:
- multilingual
license: mit
tags:
- text-classification
- multi-label-classification
- multilingual
- tourism
- xlm-roberta
datasets:
- custom
metrics:
- f1
model-index:
- name: XLM-RoBERTa Tourism Topic Classifier
  results:
  - task:
      type: text-classification
      name: Multi-label Topic Classification
    metrics:
    - type: f1_macro
      value: 0.5374
      name: F1 Macro
---

# XLM-RoBERTa Tourism Topic Classifier

Fine-tuned XLM-RoBERTa model for multilingual tourism comment topic classification.

## Model Description

This model classifies multilingual tourism comments into 7 topics:
- **scenery** (Landscapes, views)
- **food** (Cuisine, dining)
- **service** (Staff, customer service)
- **pricing** (Cost, value)
- **facilities** (Infrastructure, amenities)
- **activities** (Things to do)
- **accessibility** (Transportation, location)

## Training Data

- **Languages**: 39+ languages (English, Korean, Russian, German, French, etc.)
- **Dataset**: 462 non-Vietnamese tourism comments from social media
- **Sources**: Facebook, TikTok, YouTube, Google Maps
- **Quality**: Filtered for meaningful content with rule-based topic labels

## Performance

| Metric | Score |
|--------|-------|
| F1 Macro | 53.74% |
| F1 Micro | 61.23% |
| Hamming Loss | 15.67% |

### Language Distribution in Training

| Language | Comments |
|----------|----------|
| English (en) | 342 |
| Korean (ko) | 84 |
| Russian (ru) | 46 |
| German (de) | 37 |
| French (fr) | 24 |
| Others (35+ langs) | 29 |

## Usage

```python
import torch
from transformers import AutoTokenizer

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained('xlm-roberta-base')

# Load model
model = torch.load('xlm_best_model.pt')
model.eval()

# Predict (works with any language)
text = "Beautiful scenery, great food, a bit expensive"
encoding = tokenizer(text, return_tensors='pt', max_length=256,
                     padding='max_length', truncation=True)

with torch.no_grad():
    logits = model(encoding['input_ids'], encoding['attention_mask'])
    probs = torch.sigmoid(logits)
    predictions = (probs > 0.3).float()  # Lower threshold for multilingual

# Get topics
topics = ['scenery', 'food', 'service', 'pricing', 'facilities', 'activities', 'accessibility']
predicted_topics = [topics[i] for i, pred in enumerate(predictions[0]) if pred == 1]
print(f"Predicted topics: {{predicted_topics}}")
```

## Training Details

- **Base Model**: xlm-roberta-base
- **Architecture**: XLM-RoBERTa + Classification Head (768 → 7)
- **Parameters**: ~270M (base) + 5,376 (classifier)
- **Training Time**: ~15-20 minutes on RTX 4060
- **Epochs**: 5
- **Batch Size**: 16
- **Learning Rate**: 2e-5
- **Optimizer**: AdamW
- **Loss**: BCEWithLogitsLoss (multi-label)

## Complementary to PhoBERT

This model is designed to work alongside PhoBERT:
- **PhoBERT**: Vietnamese comments (79% of data, F1: 56.38%)
- **XLM-RoBERTa**: Non-Vietnamese comments (21% of data, F1: 53.74%)

## Limitations

- Limited training data (462 samples)
- Performance lower than PhoBERT due to less data
- May not work well for very rare languages
- Trained on tourism domain only

## Citation

```bibtex
@misc{{xlm-tourism-classifier,
  author = {{Strawberry0604}},
  title = {{XLM-RoBERTa Tourism Topic Classifier}},
  year = {{2025}},
  publisher = {{HuggingFace}},
  howpublished = {{\\url{{https://huggingface.co/{HF_USERNAME}/xlm-tourism-topic-classifier}}}}
}}
```

## Contact

- Repository: [tourism-data-monitor](https://github.com/TechmoNoway/tourism-data-monitor)
- HuggingFace: [@{HF_USERNAME}](https://huggingface.co/{HF_USERNAME})
"""

def create_config_json():
    """Create config.json for model"""
    config = {
        "model_type": "xlm-roberta",
        "num_labels": 7,
        "id2label": {str(i): label for i, label in enumerate(TOPICS)},
        "label2id": {label: i for i, label in enumerate(TOPICS)},
        "problem_type": "multi_label_classification",
        "threshold": 0.3,
        "max_length": 256,
        "dropout": 0.3,
        "base_model": "xlm-roberta-base",
        "languages": "multilingual (39+ languages)"
    }
    return config

def main():
    """Main upload function"""
    print("\n" + "="*80)
    print("UPLOAD XLM-ROBERTA MODEL TO HUGGINGFACE")
    print("="*80)
    
    # Check if model exists
    if not os.path.exists(XLM_MODEL_PATH):
        print(f"\nERROR: XLM model not found at {XLM_MODEL_PATH}")
        print("Please train the model first: python training/train_xlm_topic_classifier.py")
        return
    
    # Initialize API
    print("\nInitializing HuggingFace API...")
    try:
        api = HfApi()
        repo_name = f"{HF_USERNAME}/xlm-tourism-topic-classifier"
        
        # Create repository
        print(f"\nCreating repository: {repo_name}")
        try:
            create_repo(repo_name, exist_ok=True, private=False)
            print("✓ Repository created/verified successfully!")
        except Exception as e:
            print(f"✓ Repository already exists: {e}")
        
        # Upload model file
        print(f"\n[1/4] Uploading model file ({os.path.getsize(XLM_MODEL_PATH) / (1024**3):.2f} GB)...")
        upload_file(
            path_or_fileobj=XLM_MODEL_PATH,
            path_in_repo="xlm_best_model.pt",
            repo_id=repo_name,
            repo_type="model"
        )
        print("✓ Model file uploaded!")
        
        # Create and upload README
        print("\n[2/4] Creating and uploading README.md...")
        readme_content = create_model_card()
        with open("temp_readme_xlm.md", "w", encoding="utf-8") as f:
            f.write(readme_content)
        
        upload_file(
            path_or_fileobj="temp_readme_xlm.md",
            path_in_repo="README.md",
            repo_id=repo_name,
            repo_type="model"
        )
        os.remove("temp_readme_xlm.md")
        print("✓ README uploaded!")
        
        # Create and upload config
        print("\n[3/4] Creating and uploading config.json...")
        config = create_config_json()
        with open("temp_config_xlm.json", "w") as f:
            json.dump(config, f, indent=2)
        
        upload_file(
            path_or_fileobj="temp_config_xlm.json",
            path_in_repo="config.json",
            repo_id=repo_name,
            repo_type="model"
        )
        os.remove("temp_config_xlm.json")
        print("✓ Config uploaded!")
        
        # Upload topic statistics
        stats_path = "training/models/xlm_topic_statistics.csv"
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
        print("✅ XLM-ROBERTA MODEL UPLOADED SUCCESSFULLY!")
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
