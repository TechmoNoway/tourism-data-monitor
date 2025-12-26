"""
Test script for trained XLM-RoBERTa Topic Classifier
Loads the trained model and tests on multilingual sample comments.

Usage:
    python training/test_xlm_topic_classifier.py
    python training/test_xlm_topic_classifier.py --model_path training/models/xlm_best_model.pt
"""

import os
import sys
import argparse
import torch
from transformers import AutoTokenizer

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from training.train_xlm_tourism_topic_classifier import XLMRoBERTaTopicClassifier, TOPICS


def load_model(model_path: str, device: str = 'cuda'):
    """Load trained model from checkpoint"""
    print(f"📦 Loading model from {model_path}...")
    
    checkpoint = torch.load(model_path, map_location=device)
    
    model = XLMRoBERTaTopicClassifier(n_classes=len(TOPICS))
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
    print("🧪 XLM-RoBERTa Topic Classifier - Testing")
    print("=" * 80)
    
    # Set device
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"\n🔧 Using device: {device}")
    
    # Load model
    model, threshold = load_model(args.model_path, device)
    
    # Load tokenizer
    print("\n📝 Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained('xlm-roberta-base')
    
    # Test samples (multilingual)
    test_samples = [
        # English
        "Amazing scenery! The mountain view is breathtaking and the sunset is gorgeous.",
        "The food was delicious, especially the fresh seafood dishes.",
        "Staff was rude and unprofessional. Very disappointed with the service.",
        "Too expensive for what you get. Not worth the price at all.",
        
        # Korean
        "경치가 정말 아름답네요! 사진 찍기 좋은 곳입니다.",
        "음식이 맛있어요. 특히 현지 요리가 훌륭합니다.",
        "직원들이 친절하고 시설이 깨끗합니다.",
        
        # Chinese
        "风景太美了！拍照超级好看的地方。",
        "食物很好吃，海鲜很新鲜。",
        "服务态度不好，房间也不干净。",
        
        # Japanese
        "景色が素晴らしい！写真映えする場所です。",
        "料理が美味しい、特に海鮮料理が新鮮です。",
        "スタッフが親切で、施設がきれいです。",
        
        # Russian
        "Красивые виды! Отличное место для фотографий.",
        "Еда вкусная, особенно морепродукты свежие.",
        "Персонал грубый, комнаты грязные.",
    ]
    
    print("\n" + "=" * 80)
    print("🎯 Testing on Multilingual Sample Comments")
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
                    print(f"\n   ✅ Predicted Topics:")
                    for pred in predictions:
                        print(f"      {pred['topic']:12s}: {pred['confidence']:.2%}")
                else:
                    print(f"\n   ⚠️  No topics detected (all below threshold {threshold})")
                
                print(f"\n   📊 All Topic Scores:")
                for topic, prob in zip(TOPICS, all_probs):
                    emoji = "✅" if prob >= threshold else "❌"
                    print(f"      {emoji} {topic:12s}: {prob:.2%}")
                    
            except KeyboardInterrupt:
                print("\n\n👋 Goodbye!")
                break
            except Exception as e:
                print(f"\n❌ Error: {e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Test XLM-RoBERTa Topic Classifier')
    parser.add_argument(
        '--model_path',
        type=str,
        default='training/models/xlm_best_model.pt',
        help='Path to trained model checkpoint'
    )
    parser.add_argument(
        '--device',
        type=str,
        default='cuda',
        choices=['cuda', 'cpu'],
        help='Device to use for inference'
    )
    parser.add_argument(
        '--interactive',
        action='store_true',
        help='Enable interactive mode for manual testing'
    )
    
    args = parser.parse_args()
    main(args)
