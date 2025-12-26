"""
Train PhoBERT Sentiment Classifier for Vietnamese Comments
3-class classification: positive, negative, neutral

Usage:
    python training/train_phobert_sentiment.py --epochs 5 --batch_size 16 --device cuda
"""

import os
import sys
import argparse
import pandas as pd
import numpy as np
from typing import List, Tuple
import warnings

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from transformers import (
    AutoTokenizer,
    AutoModel,
    get_linear_schedule_with_warmup
)
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, f1_score, accuracy_score, confusion_matrix

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.core.config import settings
from app.models.comment import Comment
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

warnings.filterwarnings('ignore')

# Sentiment labels
SENTIMENTS = ['positive', 'neutral', 'negative']
SENTIMENT_TO_IDX = {sentiment: idx for idx, sentiment in enumerate(SENTIMENTS)}
IDX_TO_SENTIMENT = {idx: sentiment for sentiment, idx in SENTIMENT_TO_IDX.items()}


class SentimentDataset(Dataset):
    """Dataset for sentiment classification"""
    
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
            'labels': torch.LongTensor([label])
        }


class PhoBERTSentimentClassifier(nn.Module):
    """PhoBERT with sentiment classification head"""
    
    def __init__(self, n_classes: int = 3, dropout: float = 0.3):
        super(PhoBERTSentimentClassifier, self).__init__()
        self.phobert = AutoModel.from_pretrained('vinai/phobert-base')
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(self.phobert.config.hidden_size, n_classes)
        
    def forward(self, input_ids, attention_mask):
        outputs = self.phobert(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        pooled_output = outputs[1]  # [CLS] token representation
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)
        return logits


def load_data_from_db() -> pd.DataFrame:
    """Load Vietnamese comments with sentiment labels from database"""
    print("Loading Vietnamese comments from database...")
    
    engine = create_engine(settings.DATABASE_URL)
    Session = sessionmaker(bind=engine)
    session = Session()
    
    try:
        comments = session.query(Comment).filter(
            Comment.language == 'vi',
            Comment.sentiment.isnot(None),
            Comment.content.isnot(None),
            Comment.is_meaningful.is_(True)
        ).all()
        
        data = []
        for comment in comments:
            content = comment.cleaned_content or comment.content
            if content and len(content.strip()) > 10:  # Min length
                data.append({
                    'text': content.strip(),
                    'sentiment': comment.sentiment
                })
        
        df = pd.DataFrame(data)
        
        print(f"\nLoaded {len(df)} Vietnamese comments")
        print("\nSentiment distribution:")
        print(df['sentiment'].value_counts())
        print(f"\n{df['sentiment'].value_counts(normalize=True)*100}")
        
        return df
        
    finally:
        session.close()


def train_epoch(model, data_loader, criterion, optimizer, device, scheduler):
    """Train for one epoch"""
    model.train()
    
    losses = []
    correct_predictions = 0
    
    for batch in data_loader:
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device).flatten()
        
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        
        loss = criterion(outputs, labels)
        
        _, preds = torch.max(outputs, dim=1)
        correct_predictions += torch.sum(preds == labels)
        
        losses.append(loss.item())
        
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()
    
    return correct_predictions.double() / len(data_loader.dataset), np.mean(losses)


def eval_model(model, data_loader, criterion, device):
    """Evaluate model"""
    model.eval()
    
    losses = []
    correct_predictions = 0
    predictions_list = []
    labels_list = []
    
    with torch.no_grad():
        for batch in data_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device).flatten()
            
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            
            loss = criterion(outputs, labels)
            
            _, preds = torch.max(outputs, dim=1)
            correct_predictions += torch.sum(preds == labels)
            
            losses.append(loss.item())
            predictions_list.extend(preds.cpu().numpy())
            labels_list.extend(labels.cpu().numpy())
    
    accuracy = correct_predictions.double() / len(data_loader.dataset)
    f1_macro = f1_score(labels_list, predictions_list, average='macro')
    f1_weighted = f1_score(labels_list, predictions_list, average='weighted')
    
    return accuracy, np.mean(losses), f1_macro, f1_weighted, predictions_list, labels_list


def main(args):
    """Main training function"""
    print("="*80)
    print("PhoBERT Sentiment Classifier Training")
    print("="*80)
    
    # Set device
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"\nUsing device: {device}")
    
    # Load data
    df = load_data_from_db()
    
    if len(df) < 100:
        print("\n❌ Not enough data to train sentiment classifier")
        return
    
    # Convert sentiment to indices
    df['label'] = df['sentiment'].map(SENTIMENT_TO_IDX)
    
    # Split data
    train_texts, val_texts, train_labels, val_labels = train_test_split(
        df['text'].values,
        df['label'].values,
        test_size=0.2,
        random_state=42,
        stratify=df['label'].values
    )
    
    print(f"\nTrain set: {len(train_texts)} samples")
    print(f"Val set:   {len(val_texts)} samples")
    
    # Load tokenizer
    print("\nLoading PhoBERT tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained('vinai/phobert-base')
    
    # Create datasets
    train_dataset = SentimentDataset(train_texts, train_labels, tokenizer, args.max_length)
    val_dataset = SentimentDataset(val_texts, val_labels, tokenizer, args.max_length)
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size)
    
    # Create model
    print("\nInitializing PhoBERT model...")
    model = PhoBERTSentimentClassifier(n_classes=len(SENTIMENTS), dropout=args.dropout)
    model = model.to(device)
    
    # Optimizer and scheduler
    optimizer = AdamW(model.parameters(), lr=args.learning_rate)
    total_steps = len(train_loader) * args.epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(0.1 * total_steps),
        num_training_steps=total_steps
    )
    
    # Loss function
    criterion = nn.CrossEntropyLoss().to(device)
    
    # Training loop
    print("\n" + "="*80)
    print("Starting Training")
    print("="*80)
    
    best_f1_macro = 0
    best_epoch = 0
    
    for epoch in range(args.epochs):
        print(f"\nEpoch {epoch + 1}/{args.epochs}")
        print("-" * 80)
        
        train_acc, train_loss = train_epoch(
            model, train_loader, criterion, optimizer, device, scheduler
        )
        
        print(f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f}")
        
        val_acc, val_loss, val_f1_macro, val_f1_weighted, preds, labels = eval_model(
            model, val_loader, criterion, device
        )
        
        print(f"Val Loss:   {val_loss:.4f} | Val Acc: {val_acc:.4f}")
        print(f"Val F1 Macro: {val_f1_macro:.4f} | Val F1 Weighted: {val_f1_weighted:.4f}")
        
        # Save best model
        if val_f1_macro > best_f1_macro:
            best_f1_macro = val_f1_macro
            best_epoch = epoch + 1
            
            # Save model
            os.makedirs('training/models', exist_ok=True)
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'f1_macro': val_f1_macro,
                'f1_weighted': val_f1_weighted,
                'accuracy': val_acc.item(),
                'sentiments': SENTIMENTS
            }, 'training/models/phobert_sentiment_best_model.pt')
            
            print(f"\n✅ Saved best model! F1 Macro: {val_f1_macro:.4f}")
    
    # Final evaluation
    print("\n" + "="*80)
    print("Training Complete!")
    print("="*80)
    print(f"\nBest Epoch: {best_epoch}")
    print(f"Best F1 Macro: {best_f1_macro:.4f}")
    
    # Load best model for final evaluation
    checkpoint = torch.load('training/models/phobert_sentiment_best_model.pt')
    model.load_state_dict(checkpoint['model_state_dict'])
    
    val_acc, val_loss, val_f1_macro, val_f1_weighted, preds, labels = eval_model(
        model, val_loader, criterion, device
    )
    
    print("\n" + "="*80)
    print("Final Evaluation on Validation Set")
    print("="*80)
    
    print(f"\nAccuracy:     {val_acc:.4f}")
    print(f"F1 Macro:     {val_f1_macro:.4f}")
    print(f"F1 Weighted:  {val_f1_weighted:.4f}")
    
    print("\nClassification Report:")
    print(classification_report(labels, preds, target_names=SENTIMENTS, digits=4))
    
    print("\nConfusion Matrix:")
    cm = confusion_matrix(labels, preds)
    print(cm)
    
    # Save statistics
    stats_df = pd.DataFrame({
        'sentiment': SENTIMENTS,
        'count': [len(df[df['sentiment'] == s]) for s in SENTIMENTS],
        'percentage': [len(df[df['sentiment'] == s])/len(df)*100 for s in SENTIMENTS]
    })
    
    stats_df.to_csv('training/models/phobert_sentiment_statistics.csv', index=False)
    print("\n✅ Saved statistics to training/models/phobert_sentiment_statistics.csv")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train PhoBERT Sentiment Classifier')
    parser.add_argument('--epochs', type=int, default=5, help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size')
    parser.add_argument('--learning_rate', type=float, default=2e-5, help='Learning rate')
    parser.add_argument('--max_length', type=int, default=256, help='Max sequence length')
    parser.add_argument('--dropout', type=float, default=0.3, help='Dropout rate')
    parser.add_argument('--device', type=str, default='cuda', choices=['cuda', 'cpu'], help='Device')
    
    args = parser.parse_args()
    main(args)
