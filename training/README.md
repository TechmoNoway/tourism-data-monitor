# Tourism Topic Classifier Training

Training scripts for fine-tuning PhoBERT (Vietnamese) and XLM-RoBERTa (Multilingual) on tourism comment topic classification.

## 📋 Requirements

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install transformers datasets scikit-learn deep-translator langdetect
```

**GPU Requirements:**
- NVIDIA GPU with CUDA support (tested on RTX 4060)
- 8GB+ VRAM
- CUDA 11.8+

## 🚀 Quick Start

### 1. Train PhoBERT (Vietnamese)

```bash
# Basic training (5 epochs, batch size 16)
python training/train_phobert_topic_classifier.py

# Custom configuration
python training/train_phobert_topic_classifier.py \
    --epochs 10 \
    --batch_size 32 \
    --learning_rate 3e-5 \
    --min_quality medium
```

### 2. Train XLM-RoBERTa (Multilingual)

```bash
# Train on non-Vietnamese comments
python training/train_xlm_topic_classifier.py --exclude_vietnamese --epochs 5

# Train on all languages
python training/train_xlm_topic_classifier.py --epochs 5
```

**Training Parameters:**

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--epochs` | 5 | Number of training epochs |
| `--batch_size` | 16 | Batch size (reduce if OOM) |
| `--learning_rate` | 2e-5 | Learning rate |
| `--max_length` | 256 | Max token sequence length |
| `--min_quality` | low | Minimum quality tier (low/medium/high) |
| `--train_split` | 0.8 | Train/validation split ratio |
| `--dropout` | 0.3 | Dropout rate |
| `--threshold` | 0.5 | Classification threshold |
| `--exclude_vietnamese` | False | (XLM only) Exclude Vietnamese comments |

### 3. Test Models

```bash
# Test PhoBERT (Vietnamese)
python training/test_phobert_topic_classifier.py

# Interactive mode
python training/test_phobert_topic_classifier.py --interactive

# Custom model path
python training/test_phobert_topic_classifier.py --model_path training/models/phobert_best_model.pt
```

## 📊 Expected Performance

**Rule-Based System (Baseline):**
- F1 Score: ~0.60-0.65
- Method: Keyword matching + Translation

**PhoBERT Fine-tuning (Vietnamese):**
- F1 Macro: **56.38%**
- Training: 951 Vietnamese comments
- Best Topics: Scenery (87%), Activities (68%), Food (61%)
- Training Time (RTX 4060): ~30-45 minutes

**XLM-RoBERTa Fine-tuning (Multilingual):**
- F1 Macro: **53.74%**
- Training: 462 non-Vietnamese comments (40+ languages)
- Best Topics: Activities (77%), Facilities (76%), Scenery (75%), Food (67%)
- Training Time (RTX 4060): ~20-30 minutes

**Dual-Model System:**
- Vietnamese → PhoBERT (56% F1)
- Other languages → XLM-RoBERTa (54% F1)
- Fallback → Rule-based with translation

## 🏷️ Topics

The model classifies comments into 7 tourism-related topics:

1. **scenery** (Phong cảnh): Views, landscapes, natural beauty
2. **food** (Ẩm thực): Food, restaurants, cuisine
3. **service** (Dịch vụ): Staff, customer service
4. **pricing** (Giá cả): Prices, value for money
5. **facilities** (Cơ sở vật chất): Accommodation, cleanliness, amenities
6. **activities** (Hoạt động): Entertainment, things to do
7. **accessibility** (Khả năng tiếp cận): Transportation, parking, wheelchair access

## 📁 Output Files

After training:

```
training/
├── models/
│   ├── phobert_best_model.pt           # PhoBERT model checkpoint
│   ├── phobert_topic_statistics.csv    # PhoBERT topic stats
│   ├── xlm_best_model.pt               # XLM-RoBERTa model checkpoint
│   └── xlm_topic_statistics.csv        # XLM-RoBERTa topic stats
└── train_phobert_topic_classifier.py   # Vietnamese training script
    train_xlm_topic_classifier.py       # Multilingual training script
    test_phobert_topic_classifier.py    # Testing script
```

## 🔄 Integration with Main System

After training, integrate the model into your system:

```python
# app/services/topic_classifier.py
import torch
from transformers import AutoTokenizer
from training.train_topic_classifier import PhoBERTTopicClassifier, TOPICS

class TopicClassifier:
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Load trained model
        checkpoint = torch.load('training/models/best_model.pt', map_location=self.device)
        self.model = PhoBERTTopicClassifier()
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.to(self.device)
        self.model.eval()
        
        self.tokenizer = AutoTokenizer.from_pretrained('vinai/phobert-base')
        self.threshold = checkpoint['threshold']
    
    def classify(self, text: str) -> List[str]:
        encoding = self.tokenizer(
            text,
            max_length=256,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        input_ids = encoding['input_ids'].to(self.device)
        attention_mask = encoding['attention_mask'].to(self.device)
        
        with torch.no_grad():
            logits = self.model(input_ids, attention_mask)
            probs = torch.sigmoid(logits).cpu().numpy()[0]
        
        # Return topics above threshold
        return [TOPICS[i] for i, prob in enumerate(probs) if prob >= self.threshold]
```

## 📈 Monitoring Training

Watch for these indicators:

**Good Training:**
- ✅ Train loss decreasing steadily
- ✅ Val F1 improving
- ✅ No large gap between train/val loss (no overfitting)
- ✅ Per-topic F1 scores balanced

**Problems:**
- ❌ Val loss increasing → Reduce learning rate or add regularization
- ❌ Train loss not decreasing → Increase learning rate
- ❌ Large train/val gap → Reduce model complexity or add dropout
- ❌ Some topics always 0 → Increase threshold or collect more data

## 🐛 Troubleshooting

### CUDA Out of Memory
```bash
# Reduce batch size
python training/train_topic_classifier.py --batch_size 8

# Reduce sequence length
python training/train_topic_classifier.py --max_length 128
```

### Not Enough Data
```bash
# Collect more data first
python scripts/collect_data.py

# Use lower quality threshold
python training/train_topic_classifier.py --min_quality low
```

### Poor Performance on Specific Topics
- Check topic distribution in data
- Collect more examples for underrepresented topics
- Adjust classification threshold
- Try weighted loss (modify code)

## 🌐 Upload to HuggingFace (Optional)

Share your trained model publicly:

```python
from huggingface_hub import login

# 1. Login
login(token="hf_xxxxxxxxxxxxx")

# 2. Upload
model.push_to_hub("TechmoNoway/phobert-vietnamese-tourism-topics")
tokenizer.push_to_hub("TechmoNoway/phobert-vietnamese-tourism-topics")

# 3. Anyone can use:
# from transformers import AutoModelForSequenceClassification
# model = AutoModelForSequenceClassification.from_pretrained(
#     "TechmoNoway/phobert-vietnamese-tourism-topics"
# )
```

## 📚 Additional Resources

- [PhoBERT Paper](https://arxiv.org/abs/2003.00744)
- [Transformers Documentation](https://huggingface.co/docs/transformers)
- [Multi-label Classification Guide](https://huggingface.co/docs/transformers/tasks/sequence_classification)
- [PyTorch CUDA Setup](https://pytorch.org/get-started/locally/)

## 🤝 Contributing

To improve the model:
1. Collect more diverse comments
2. Manually label uncertain predictions (active learning)
3. Experiment with different hyperparameters
4. Try ensemble methods (PhoBERT + rule-based)
