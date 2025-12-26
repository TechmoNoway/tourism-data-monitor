"""
--epochs 5 --batch_size 16
"""

import os
import sys
import argparse
import pandas as pd
import numpy as np
from typing import List, Tuple
import warnings
import torch
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from transformers import (
    AutoTokenizer,
    AutoModel,
    get_linear_schedule_with_warmup
)
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, hamming_loss
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from app.core.config import settings
from app.models.comment import Comment

warnings.filterwarnings('ignore')

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))



# Topic labels
TOPICS = ['scenery', 'food', 'service', 'pricing', 'facilities', 'activities', 'accessibility']


class TourismCommentDataset(Dataset):
    """Dataset for tourism comments with multi-label topics"""
    
    def __init__(self, texts: List[str], labels: np.ndarray, tokenizer, max_length: int = 256):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = self.labels[idx]
        
        encoding = self.tokenizer(
            text,
            add_special_tokens=True,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.FloatTensor(label)
        }


class PhoBERTTopicClassifier(torch.nn.Module):
    """PhoBERT with multi-label classification head"""
    
    def __init__(self, n_classes: int = 7, dropout: float = 0.3):
        super(PhoBERTTopicClassifier, self).__init__()
        self.phobert = AutoModel.from_pretrained('vinai/phobert-base')
        self.dropout = torch.nn.Dropout(dropout)
        self.classifier = torch.nn.Linear(self.phobert.config.hidden_size, n_classes)
        
    def forward(self, input_ids, attention_mask):
        outputs = self.phobert(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        pooled_output = outputs[1]  # [CLS] token representation
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)
        return logits


def load_data_from_db(min_quality: str = 'low') -> pd.DataFrame:
    """Load Vietnamese comments from database with their rule-based topic labels"""
    print("Loading data from database...")
    
    engine = create_engine(settings.DATABASE_URL)
    Session = sessionmaker(bind=engine)
    session = Session()
    
    try:
        quality_tiers = ['high', 'medium', 'low']
        if min_quality == 'medium':
            quality_tiers = ['high', 'medium']
        elif min_quality == 'high':
            quality_tiers = ['high']
        
        comments = session.query(Comment).filter(
            Comment.quality_tier.in_(quality_tiers),
            Comment.topics.isnot(None),
            Comment.language == 'vi'  # Only Vietnamese comments
        ).all()
        
        data = []
        for comment in comments:
            if comment.topics and len(comment.topics) > 0:
                data.append({
                    'text': comment.content,
                    'topics': comment.topics,
                    'quality_tier': comment.quality_tier,
                    'sentiment': comment.sentiment
                })
        
        df = pd.DataFrame(data)
        print(f"Loaded {len(df)} Vietnamese comments")
        print(f"   Quality distribution: {df['quality_tier'].value_counts().to_dict()}")
        
        total_comments = session.query(Comment).filter(
            Comment.quality_tier.in_(quality_tiers),
            Comment.topics.isnot(None)
        ).count()
        non_vi_count = total_comments - len(df)
        if non_vi_count > 0:
            print(f"   Skipped {non_vi_count} non-Vietnamese comments")
        
        return df
        
    finally:
        session.close()


def prepare_labels(df: pd.DataFrame) -> Tuple[np.ndarray, pd.DataFrame]:
    """Convert topic lists to multi-hot encoded labels"""
    print("\nPreparing labels...")
    
    labels = np.zeros((len(df), len(TOPICS)))
    
    for idx, topics in enumerate(df['topics']):
        if topics:
            for topic in topics:
                if topic in TOPICS:
                    topic_idx = TOPICS.index(topic)
                    labels[idx, topic_idx] = 1
    
    topic_counts = labels.sum(axis=0)
    stats_df = pd.DataFrame({
        'topic': TOPICS,
        'count': topic_counts.astype(int),
        'percentage': (topic_counts / len(df) * 100).round(1)
    })
    
    print("\nTopic distribution:")
    print(stats_df.to_string(index=False))
    print(f"\n   Average topics per comment: {labels.sum() / len(labels):.2f}")
    
    return labels, stats_df


def train_epoch(model, dataloader, optimizer, scheduler, device, epoch):
    """Train for one epoch"""
    model.train()
    total_loss = 0
    
    for batch_idx, batch in enumerate(dataloader):
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)
        
        optimizer.zero_grad()
        
        logits = model(input_ids, attention_mask)
        
        loss_fn = torch.nn.BCEWithLogitsLoss()
        loss = loss_fn(logits, labels)
        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()
        
        total_loss += loss.item()
        
        if (batch_idx + 1) % 10 == 0:
            print(f"   Batch {batch_idx + 1}/{len(dataloader)} - Loss: {loss.item():.4f}")
    
    avg_loss = total_loss / len(dataloader)
    return avg_loss


def evaluate(model, dataloader, device, threshold: float = 0.5):
    """Evaluate model on validation set"""
    model.eval()
    all_preds = []
    all_labels = []
    total_loss = 0
    
    loss_fn = torch.nn.BCEWithLogitsLoss()
    
    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            
            logits = model(input_ids, attention_mask)
            loss = loss_fn(logits, labels)
            total_loss += loss.item()
            
            probs = torch.sigmoid(logits)
            preds = (probs > threshold).float()
            
            all_preds.append(preds.cpu().numpy())
            all_labels.append(labels.cpu().numpy())
    
    all_preds = np.vstack(all_preds)
    all_labels = np.vstack(all_labels)
    
    avg_loss = total_loss / len(dataloader)
    f1_micro = f1_score(all_labels, all_preds, average='micro', zero_division=0)
    f1_macro = f1_score(all_labels, all_preds, average='macro', zero_division=0)
    f1_samples = f1_score(all_labels, all_preds, average='samples', zero_division=0)
    hamming = hamming_loss(all_labels, all_preds)
    
    topic_f1 = {}
    for idx, topic in enumerate(TOPICS):
        f1 = f1_score(all_labels[:, idx], all_preds[:, idx], zero_division=0)
        topic_f1[topic] = f1
    
    return {
        'loss': avg_loss,
        'f1_micro': f1_micro,
        'f1_macro': f1_macro,
        'f1_samples': f1_samples,
        'hamming_loss': hamming,
        'topic_f1': topic_f1
    }


def main(args):
    """Main training function"""
    print("=" * 80)
    print("PhoBERT Topic Classifier Training")
    print("=" * 80)
    print("\nConfiguration:")
    print(f"   Epochs: {args.epochs}")
    print(f"   Batch size: {args.batch_size}")
    print(f"   Learning rate: {args.learning_rate}")
    print(f"   Max length: {args.max_length}")
    print(f"   Threshold: {args.threshold}")
    print(f"   Device: {args.device}")
    
    # Set device
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"\nUsing device: {device}")
    if device.type == 'cuda':
        print(f"   GPU: {torch.cuda.get_device_name(0)}")
        print(f"   CUDA version: {torch.version.cuda}")
    
    # Load data
    df = load_data_from_db(min_quality=args.min_quality)
    
    if len(df) < 100:
        print(f"\nERROR: Not enough Vietnamese data! Found {len(df)} comments, need at least 100.")
        print("   Solutions:")
        print("      1. Run data collection: python scripts/collect_data.py")
        print("      2. Collect more Vietnamese comments from social media")
        print("      3. Use --min_quality low to include all quality tiers")
        return
    
    labels, stats_df = prepare_labels(df)
    
    print(f"\nSplitting data (train: {args.train_split}, val: {1-args.train_split})...")
    X_train, X_val, y_train, y_val = train_test_split(
        df['text'].values,
        labels,
        test_size=1-args.train_split,
        random_state=42,
        stratify=None  # Can't stratify multi-label easily
    )
    print(f"   Train: {len(X_train)} samples")
    print(f"   Val: {len(X_val)} samples")
    
    print("\nLoading PhoBERT tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained('vinai/phobert-base')
    
    train_dataset = TourismCommentDataset(X_train, y_train, tokenizer, args.max_length)
    val_dataset = TourismCommentDataset(X_val, y_val, tokenizer, args.max_length)
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)
    
    print("\nInitializing PhoBERT model...")
    model = PhoBERTTopicClassifier(n_classes=len(TOPICS), dropout=args.dropout)
    model.to(device)
    
    optimizer = AdamW(model.parameters(), lr=args.learning_rate, weight_decay=0.01)
    total_steps = len(train_loader) * args.epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(0.1 * total_steps),
        num_training_steps=total_steps
    )
    
    print("\n" + "=" * 80)
    print("Starting Training")
    print("=" * 80)
    
    best_f1 = 0
    best_model_path = None
    
    for epoch in range(args.epochs):
        print(f"\nEpoch {epoch + 1}/{args.epochs}")
        print("-" * 80)
        
        train_loss = train_epoch(model, train_loader, optimizer, scheduler, device, epoch)
        print(f"\n   Train Loss: {train_loss:.4f}")
        
        print("\n   Evaluating on validation set...")
        metrics = evaluate(model, val_loader, device, args.threshold)
        
        print("\n   Validation Metrics:")
        print(f"      Loss: {metrics['loss']:.4f}")
        print(f"      F1 Micro: {metrics['f1_micro']:.4f}")
        print(f"      F1 Macro: {metrics['f1_macro']:.4f}")
        print(f"      F1 Samples: {metrics['f1_samples']:.4f}")
        print(f"      Hamming Loss: {metrics['hamming_loss']:.4f}")
        
        print("\n   Per-Topic F1:")
        for topic, f1 in metrics['topic_f1'].items():
            print(f"      {topic:12s}: {f1:.4f}")
        
        if metrics['f1_macro'] > best_f1:
            best_f1 = metrics['f1_macro']
            
            output_dir = args.output_dir
            os.makedirs(output_dir, exist_ok=True)
            
            model_path = os.path.join(output_dir, 'phobert_best_model.pt')
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'f1_macro': best_f1,
                'metrics': metrics,
                'topics': TOPICS,
                'threshold': args.threshold
            }, model_path)
            
            best_model_path = model_path
            print(f"\n   New best model saved! F1 Macro: {best_f1:.4f}")
    
    print("\n" + "=" * 80)
    print("Training Complete!")
    print("=" * 80)
    print(f"\nBest F1 Macro: {best_f1:.4f}")
    print(f"Model saved to: {best_model_path}")
    
    stats_path = os.path.join(args.output_dir, 'phobert_topic_statistics.csv')
    stats_df.to_csv(stats_path, index=False)
    print(f"Topic statistics saved to: {stats_path}")
    
    print("\nNext steps:")
    print("   1. Test model: python training/test_phobert_topic_classifier.py")
    print("   2. Integrate: Update app/services/topic_classifier.py")
    print("   3. Deploy: Upload to HuggingFace (optional)")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train PhoBERT Topic Classifier')
    
    # Data parameters
    parser.add_argument('--min_quality', type=str, default='low', 
                       choices=['low', 'medium', 'high'],
                       help='Minimum quality tier for training data')
    parser.add_argument('--train_split', type=float, default=0.8,
                       help='Train/val split ratio')
    
    # Model parameters
    parser.add_argument('--max_length', type=int, default=256,
                       help='Maximum sequence length')
    parser.add_argument('--dropout', type=float, default=0.3,
                       help='Dropout rate')
    parser.add_argument('--threshold', type=float, default=0.5,
                       help='Classification threshold')
    
    # Training parameters
    parser.add_argument('--epochs', type=int, default=5,
                       help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=16,
                       help='Batch size')
    parser.add_argument('--learning_rate', type=float, default=2e-5,
                       help='Learning rate')
    
    # Output parameters
    parser.add_argument('--output_dir', type=str, default='training/models',
                       help='Output directory for saved models')
    parser.add_argument('--device', type=str, default='cuda',
                       help='Device to use (cuda or cpu)')
    
    args = parser.parse_args()
    
    main(args)
