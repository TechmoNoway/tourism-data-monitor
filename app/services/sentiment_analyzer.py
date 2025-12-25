import re
import os
from typing import Dict, List, Optional
import logging

import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModel

try:
    from langdetect import detect, LangDetectException
    LANGDETECT_AVAILABLE = True
except ImportError:
    LANGDETECT_AVAILABLE = False

logger = logging.getLogger(__name__)


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


class XLMRobertaSentimentClassifier(nn.Module):
    """XLM-RoBERTa with sentiment classification head"""
    
    def __init__(self, n_classes: int = 3, dropout: float = 0.3):
        super(XLMRobertaSentimentClassifier, self).__init__()
        self.xlm_roberta = AutoModel.from_pretrained('xlm-roberta-base')
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(self.xlm_roberta.config.hidden_size, n_classes)
        
    def forward(self, input_ids, attention_mask):
        outputs = self.xlm_roberta(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        pooled_output = outputs[0][:, 0, :]  # [CLS] token representation
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)
        return logits


class MultilingualSentimentAnalyzer:
    """
    Multilingual sentiment analyzer with support for:
    - Vietnamese (PhoBERT - custom trained)
    - 100+ languages (XLM-RoBERTa - custom trained)
    - Rule-based fallback
    - Sentiment flipper handling ("nhưng", "but", "however")
    """
    
    def __init__(self, use_gpu: bool = False):
        if not LANGDETECT_AVAILABLE:
            raise ImportError(
                "langdetect library is required. Install with: pip install langdetect"
            )
        
        self.use_gpu = use_gpu
        self.device = torch.device('cuda' if use_gpu and torch.cuda.is_available() else 'cpu')
        
        logger.info(f"Initializing sentiment analysis models on {self.device}...")
        
        # Sentiment labels
        self.sentiments = ['positive', 'neutral', 'negative']
        
        # PhoBERT for Vietnamese
        self.vi_model = None
        self.vi_tokenizer = None
        try:
            phobert_path = 'training/models/phobert_sentiment_best_model.pt'
            if os.path.exists(phobert_path):
                self.vi_tokenizer = AutoTokenizer.from_pretrained('vinai/phobert-base')
                self.vi_model = PhoBERTSentimentClassifier(n_classes=3)
                
                checkpoint = torch.load(phobert_path, map_location=self.device)
                self.vi_model.load_state_dict(checkpoint['model_state_dict'])
                self.vi_model = self.vi_model.to(self.device)
                self.vi_model.eval()
                
                logger.info(f"✓ PhoBERT sentiment model loaded (F1: {checkpoint['f1_macro']:.4f})")
            else:
                logger.warning(f"PhoBERT model not found at {phobert_path}")
        except Exception as e:
            logger.error(f"Failed to load PhoBERT: {e}")
        
        # XLM-RoBERTa for other languages
        self.xlm_model = None
        self.xlm_tokenizer = None
        try:
            xlm_path = 'training/models/xlm_sentiment_best_model.pt'
            if os.path.exists(xlm_path):
                self.xlm_tokenizer = AutoTokenizer.from_pretrained('xlm-roberta-base')
                self.xlm_model = XLMRobertaSentimentClassifier(n_classes=3)
                
                checkpoint = torch.load(xlm_path, map_location=self.device)
                self.xlm_model.load_state_dict(checkpoint['model_state_dict'])
                self.xlm_model = self.xlm_model.to(self.device)
                self.xlm_model.eval()
                
                logger.info(f"✓ XLM-RoBERTa sentiment model loaded (F1: {checkpoint['f1_macro']:.4f})")
            else:
                logger.warning(f"XLM-RoBERTa model not found at {xlm_path}")
        except Exception as e:
            logger.error(f"Failed to load XLM-RoBERTa: {e}")
        
        # Sentiment flipper keywords (priority to text AFTER these words)
        self.flipper_keywords = {
            'vi': ['nhưng', 'tuy nhiên', 'nhưng mà', 'song', 'nhưng lại'],
            'en': ['but', 'however', 'though', 'although', 'yet'],
        }
    
    def detect_language(self, text: str) -> str:
        try:
            cleaned = self._clean_for_detection(text)
            if len(cleaned) < 3:
                return 'unknown'
            
            lang = detect(cleaned)
            
            lang_map = {
                'vi': 'vi',
                'en': 'en',
                'zh-cn': 'zh-cn',
                'zh-tw': 'zh-cn',
                'ko': 'ko',
                'ja': 'ja',
                'th': 'th'
            }
            
            return lang_map.get(lang, lang)
        
        except (LangDetectException, Exception) as e:
            logger.debug(f"Language detection failed: {e}")
            return 'unknown'
    
    def _clean_for_detection(self, text: str) -> str:
        text = re.sub(r'http\S+|www\.\S+', '', text)
        text = re.sub(r'[@#]\w+', '', text)
        text = ' '.join(text.split())
        return text
    
    def clean_text(self, text: str) -> str:
        cleaned = re.sub(r'http\S+|www\.\S+', '', text)
        
        cleaned = re.sub(r'\S+@\S+', '', cleaned)

        cleaned = re.sub(r'([\U0001F600-\U0001F64F]){3,}', r'\1\1', cleaned)
        
        cleaned = re.sub(r'([!?.]){3,}', r'\1\1', cleaned)
        
        cleaned = ' '.join(cleaned.split())
        
        return cleaned.strip()
    
    def count_words(self, text: str) -> int:
        return len(text.split())
    
    def _split_by_flipper(self, text: str, language: str) -> tuple[str, str, bool]:
        """
        Split text by sentiment flipper keywords (nhưng, but, however).
        
        Returns:
            tuple: (before_text, after_text, has_flipper)
            - If flipper found: returns text before and after the FIRST flipper
            - If no flipper: returns (full_text, '', False)
        
        Strategy: Text AFTER flipper has higher priority in final sentiment.
        
        Examples:
            "Đẹp nhưng đắt" → ("Đẹp", "đắt", True) → prioritize "đắt" (negative)
            "Tốt nhưng xa" → ("Tốt", "xa", True) → prioritize "xa" (negative/neutral)
        """
        text_lower = text.lower()
        
        # Get flipper keywords for language
        flippers = self.flipper_keywords.get(language, self.flipper_keywords['en'])
        
        # Find first flipper (use earliest occurrence)
        earliest_pos = len(text)
        earliest_flipper = None
        
        for flipper in flippers:
            pos = text_lower.find(flipper)
            if pos != -1 and pos < earliest_pos:
                earliest_pos = pos
                earliest_flipper = flipper
        
        # No flipper found
        if earliest_flipper is None:
            return (text, '', False)
        
        # Split at flipper position
        before = text[:earliest_pos].strip()
        after = text[earliest_pos + len(earliest_flipper):].strip()
        
        logger.debug(f"Flipper '{earliest_flipper}' found. Before: '{before[:30]}...', After: '{after[:30]}...'")
        
        return (before, after, True)
    
    def analyze_sentiment(self, text: str, language: Optional[str] = None) -> Dict:
        """
        Analyze sentiment with support for sentiment flippers.
        
        Strategy:
        1. Detect language if not provided
        2. Check for sentiment flipper keywords (nhưng, but, however)
        3. If flipper found:
           - Analyze text BEFORE flipper
           - Analyze text AFTER flipper
           - PRIORITIZE sentiment from AFTER part (70% weight)
        4. If no flipper: analyze normally
        """
        cleaned = self.clean_text(text)
        word_count = self.count_words(cleaned)
        
        if language is None:
            language = self.detect_language(text)
        
        # Check for sentiment flipper
        before, after, has_flipper = self._split_by_flipper(cleaned, language)
        
        if has_flipper and after:
            # Analyze both parts
            result_before = self._analyze_single_part(before, language, self.count_words(before))
            result_after = self._analyze_single_part(after, language, self.count_words(after))
            
            # Prioritize AFTER part (70% weight)
            sentiment_map = {'positive': 1, 'neutral': 0, 'negative': -1}
            
            before_value = sentiment_map.get(result_before['sentiment'], 0) * 0.3
            after_value = sentiment_map.get(result_after['sentiment'], 0) * 0.7
            
            combined_value = before_value + after_value
            
            # Convert back to sentiment
            if combined_value > 0.2:
                final_sentiment = 'positive'
            elif combined_value < -0.2:
                final_sentiment = 'negative'
            else:
                final_sentiment = 'neutral'
            
            # Average confidence
            confidence = (result_before.get('sentiment_score', 0.5) * 0.3 + 
                         result_after.get('sentiment_score', 0.5) * 0.7)
            
            return {
                'sentiment': final_sentiment,
                'sentiment_score': confidence,
                'language': language,
                'analysis_model': result_after.get('analysis_model', 'rule-based'),
                'cleaned_content': cleaned,
                'word_count': word_count,
                'flipper_detected': True,
                'before_sentiment': result_before['sentiment'],
                'after_sentiment': result_after['sentiment']
            }
        
        # No flipper, analyze normally
        result = self._analyze_single_part(cleaned, language, word_count)
        result['flipper_detected'] = False
        return result
    
    def _analyze_single_part(self, text: str, language: str, word_count: int) -> Dict:
        """Analyze a single text part (used for both flipper and non-flipper cases)"""
        if language == 'vi' and self.vi_model:
            return self._analyze_with_phobert(text, language, word_count)
        elif self.xlm_model:
            return self._analyze_with_xlm(text, language, word_count)
        else:
            return self._analyze_rule_based(text, language, word_count)
    
    def _analyze_with_phobert(self, text: str, language: str, word_count: int) -> Dict:
        try:
            # Tokenize
            encoding = self.vi_tokenizer(
                text,
                add_special_tokens=True,
                max_length=256,
                padding='max_length',
                truncation=True,
                return_attention_mask=True,
                return_tensors='pt'
            )
            
            input_ids = encoding['input_ids'].to(self.device)
            attention_mask = encoding['attention_mask'].to(self.device)
            
            # Predict
            with torch.no_grad():
                outputs = self.vi_model(input_ids=input_ids, attention_mask=attention_mask)
                probs = torch.softmax(outputs, dim=1)
                confidence, predicted = torch.max(probs, dim=1)
            
            sentiment = self.sentiments[predicted.item()]
            
            return {
                'sentiment': sentiment,
                'sentiment_score': float(confidence.item()),
                'language': language,
                'analysis_model': 'phobert',
                'cleaned_content': text,
                'word_count': word_count
            }
        
        except Exception as e:
            logger.error(f"PhoBERT analysis failed: {e}")
            return self._analyze_rule_based(text, language, word_count)
    
    def _analyze_with_xlm(self, text: str, language: str, word_count: int) -> Dict:
        try:
            # Tokenize
            encoding = self.xlm_tokenizer(
                text,
                add_special_tokens=True,
                max_length=256,
                padding='max_length',
                truncation=True,
                return_attention_mask=True,
                return_tensors='pt'
            )
            
            input_ids = encoding['input_ids'].to(self.device)
            attention_mask = encoding['attention_mask'].to(self.device)
            
            # Predict
            with torch.no_grad():
                outputs = self.xlm_model(input_ids=input_ids, attention_mask=attention_mask)
                probs = torch.softmax(outputs, dim=1)
                confidence, predicted = torch.max(probs, dim=1)
            
            sentiment = self.sentiments[predicted.item()]
            
            return {
                'sentiment': sentiment,
                'sentiment_score': float(confidence.item()),
                'language': language,
                'analysis_model': 'xlm-roberta',
                'cleaned_content': text,
                'word_count': word_count
            }
        
        except Exception as e:
            logger.error(f"XLM-RoBERTa analysis failed: {e}")
            return self._analyze_rule_based(text, language, word_count)
    
    def _analyze_rule_based(self, text: str, language: str, word_count: int) -> Dict:
        text_lower = text.lower()
        
        positive_keywords_vi = [
            'tốt', 'đẹp', 'hay', 'thích', 'tuyệt', 'xuất sắc', 'hoàn hảo',
            'ấn tượng', 'hài lòng', 'đáng', 'nên', 'recommend', 'good', 'great',
            'beautiful', 'amazing', 'wonderful', 'excellent', 'love', 'like'
        ]
        
        negative_keywords_vi = [
            'xấu', 'tệ', 'kém', 'không tốt', 'thất vọng', 'đắt', 'chán',
            'tồi', 'kém chất lượng', 'không nên', 'bad', 'poor', 'terrible',
            'disappointed', 'expensive', 'boring', 'dirty', 'rude', 'waste'
        ]
        
        positive_count = sum(1 for kw in positive_keywords_vi if kw in text_lower)
        negative_count = sum(1 for kw in negative_keywords_vi if kw in text_lower)
        
        if positive_count > negative_count:
            sentiment = 'positive'
            score = min(0.6 + (positive_count * 0.1), 0.9)
        elif negative_count > positive_count:
            sentiment = 'negative'
            score = min(0.6 + (negative_count * 0.1), 0.9)
        else:
            sentiment = 'neutral'
            score = 0.5
        
        return {
            'sentiment': sentiment,
            'sentiment_score': float(score),
            'language': language,
            'analysis_model': 'rule-based',
            'cleaned_content': text,
            'word_count': word_count
        }
    
    def analyze_batch(self, texts: List[str], batch_size: int = 32) -> List[Dict]:
        """Batch processing for faster analysis"""
        results = []
        
        # For simplicity, use single analysis for now
        # Can optimize later with true batch processing
        for text in texts:
            results.append(self.analyze_sentiment(text))
        
        return results
