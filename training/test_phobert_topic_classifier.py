"""
Test script for trained PhoBERT Topic Classifier
Loads the trained model and tests on sample comments.

Usage:
    python training/test_phobert_topic_classifier.py
    python training/test_phobert_topic_classifier.py --model_path training/models/phobert_best_model.pt
"""

import os
import sys
import argparse
import torch
from transformers import AutoTokenizer

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from training.train_phobert_tourism_topic_classifier import PhoBERTTopicClassifier, TOPICS


def load_model(model_path: str, device: str = 'cuda'):
    """Load trained model from checkpoint"""
    print(f"📦 Loading model from {model_path}...")
    
    checkpoint = torch.load(model_path, map_location=device)
    
    model = PhoBERTTopicClassifier(n_classes=len(TOPICS))
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    
    print(f"✅ Model loaded successfully!")
    print(f"   Trained epoch: {checkpoint['epoch']}")
    print(f"   F1 Macro: {checkpoint['f1_macro']:.4f}")
    print(f"   Threshold: {checkpoint['threshold']}")
    
    return model, checkpoint['threshold']


def predict(model, tokenizer, text: str, threshold: float, device: str, max_length: int = 256):
    """Predict topics for a single comment"""
    # Tokenize
    encoding = tokenizer(
        text,
        add_special_tokens=True,
        max_length=max_length,
        padding='max_length',
        truncation=True,
        return_attention_mask=True,
        return_tensors='pt'
    )
    
    input_ids = encoding['input_ids'].to(device)
    attention_mask = encoding['attention_mask'].to(device)
    
    # Predict
    with torch.no_grad():
        logits = model(input_ids, attention_mask)
        probs = torch.sigmoid(logits)
    
    # Get predictions above threshold
    probs = probs.cpu().numpy()[0]
    predictions = []
    
    for idx, (topic, prob) in enumerate(zip(TOPICS, probs)):
        if prob >= threshold:
            predictions.append({
                'topic': topic,
                'confidence': float(prob)
            })
    
    # Sort by confidence
    predictions = sorted(predictions, key=lambda x: x['confidence'], reverse=True)
    
    return predictions, probs


def main(args):
    """Main testing function"""
    print("=" * 80)
    print("🧪 PhoBERT Topic Classifier - Testing")
    print("=" * 80)
    
    # Set device
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"\n🔧 Using device: {device}")
    
    # Load model
    model, threshold = load_model(args.model_path, device)
    
    # Load tokenizer
    print("\n📝 Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained('vinai/phobert-base')
    
    # Test samples
    test_samples = [
        "Cảnh đẹp tuyệt vời, view núi non hùng vĩ!",
        "Đồ ăn ngon, nhất là món hải sản tươi rói",
        "Nhân viên phục vụ thái độ tệ, không chuyên nghiệp",
        "Giá cả hơi cao so với chất lượng dịch vụ",
        "Khách sạn sạch sẽ, tiện nghi đầy đủ",
        "Nhiều hoạt động vui chơi giải trí cho trẻ em",
        "Đường đi khó khăn, không phù hợp người già",
        "Phong cảnh đẹp, đồ ăn ngon nhưng giá hơi mắc",
        "View biển tuyệt đẹp, nhà hàng hải sản tươi ngon",
        "Nhân viên nhiệt tình, phòng ốc sạch sẽ, giá cả hợp lý"
    ]
    
    print("\n" + "=" * 80)
    print("🎯 Testing on Sample Comments")
    print("=" * 80)
    
    for idx, text in enumerate(test_samples, 1):
        print(f"\n📝 Comment {idx}:")
        print(f"   \"{text}\"")
        
        predictions, all_probs = predict(model, tokenizer, text, threshold, device)
        
        if predictions:
            print(f"\n   ✅ Predicted Topics:")
            for pred in predictions:
                print(f"      {pred['topic']:12s}: {pred['confidence']:.2%}")
        else:
            print(f"\n   ⚠️  No topics detected (all below threshold {threshold})")
        
        print(f"\n   📊 All Topic Scores:")
        for topic, prob in zip(TOPICS, all_probs):
            emoji = "✅" if prob >= threshold else "❌"
            print(f"      {emoji} {topic:12s}: {prob:.2%}")
    
    # Interactive mode
    if args.interactive:
        print("\n" + "=" * 80)
        print("💬 Interactive Mode (type 'quit' to exit)")
        print("=" * 80)
        
        while True:
            try:
                text = input("\n📝 Enter comment: ").strip()
                
                if text.lower() in ['quit', 'exit', 'q']:
                    break
                
                if not text:
                    continue
                
                predictions, all_probs = predict(model, tokenizer, text, threshold, device)
                
                if predictions:
                    print(f"\n✅ Predicted Topics:")
                    for pred in predictions:
                        print(f"   {pred['topic']:12s}: {pred['confidence']:.2%}")
                else:
                    print(f"\n⚠️  No topics detected")
                
                print(f"\n📊 All Scores:")
                for topic, prob in zip(TOPICS, all_probs):
                    emoji = "✅" if prob >= threshold else "❌"
                    print(f"   {emoji} {topic:12s}: {prob:.2%}")
                    
            except KeyboardInterrupt:
                break
            except Exception as e:
                print(f"❌ Error: {e}")
    
    print("\n" + "=" * 80)
    print("✨ Testing Complete!")
    print("=" * 80)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Test PhoBERT Topic Classifier')
    
    parser.add_argument('--model_path', type=str, 
                       default='training/models/phobert_best_model.pt',
                       help='Path to trained model checkpoint')
    parser.add_argument('--device', type=str, default='cuda',
                       help='Device to use (cuda or cpu)')
    parser.add_argument('--interactive', action='store_true',
                       help='Enable interactive testing mode')
    
    args = parser.parse_args()
    
    main(args)
