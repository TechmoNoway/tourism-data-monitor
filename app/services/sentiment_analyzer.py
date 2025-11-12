import re
from typing import Dict, List, Optional
import logging

try:
    from transformers import pipeline
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False

try:
    from langdetect import detect, LangDetectException
    LANGDETECT_AVAILABLE = True
except ImportError:
    LANGDETECT_AVAILABLE = False

logger = logging.getLogger(__name__)


class MultilingualSentimentAnalyzer:
    """
    Multilingual sentiment analyzer with support for:
    - Vietnamese (PhoBERT)
    - 100+ languages (XLM-RoBERTa)
    - Rule-based fallback
    - Sentiment flipper handling ("nhưng", "but", "however")
    """
    
    def __init__(self, use_gpu: bool = False):
        if not TRANSFORMERS_AVAILABLE:
            raise ImportError(
                "transformers library is required. Install with: pip install transformers torch"
            )
        
        if not LANGDETECT_AVAILABLE:
            raise ImportError(
                "langdetect library is required. Install with: pip install langdetect"
            )
        
        self.use_gpu = use_gpu
        device = 0 if use_gpu else -1
        
        logger.info("Initializing sentiment analysis models...")
        
        # PhoBERT for Vietnamese
        try:
            self.vi_analyzer = pipeline(
                "sentiment-analysis",
                model="wonrax/phobert-base-vietnamese-sentiment",
                device=device
            )
            logger.info("✓ PhoBERT model loaded (Vietnamese)")
        except Exception as e:
            logger.error(f"Failed to load PhoBERT: {e}")
            self.vi_analyzer = None
        
        # XLM-RoBERTa for other languages
        try:
            self.multilingual_analyzer = pipeline(
                "sentiment-analysis",
                model="nlptown/bert-base-multilingual-uncased-sentiment",
                device=device
            )
            logger.info("✓ Multilingual BERT model loaded (100+ languages)")
        except Exception as e:
            logger.error(f"Failed to load multilingual model: {e}")
            self.multilingual_analyzer = None
        
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
        if language == 'vi' and self.vi_analyzer:
            return self._analyze_with_phobert(text, language, word_count)
        elif self.multilingual_analyzer:
            return self._analyze_with_xlm(text, language, word_count)
        else:
            return self._analyze_rule_based(text, language, word_count)
    
    def _analyze_with_phobert(self, text: str, language: str, word_count: int) -> Dict:
        try:
            result = self.vi_analyzer(text)[0]
            
            label_map = {
                'POS': 'positive',
                'NEG': 'negative',
                'NEU': 'neutral',
                'POSITIVE': 'positive',
                'NEGATIVE': 'negative',
                'NEUTRAL': 'neutral'
            }
            
            sentiment = label_map.get(result['label'].upper(), 'neutral')
            
            return {
                'sentiment': sentiment,
                'sentiment_score': float(result['score']),
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
            result = self.multilingual_analyzer(text)[0]
            
            label = result['label']
            
            if 'star' in label.lower():
                stars = int(label.split()[0])
                if stars >= 4:
                    sentiment = 'positive'
                elif stars <= 2:
                    sentiment = 'negative'
                else:
                    sentiment = 'neutral'
            else:
                sentiment = result['label'].lower()
            
            return {
                'sentiment': sentiment,
                'sentiment_score': float(result['score']),
                'language': language,
                'analysis_model': 'mbert',
                'cleaned_content': text,
                'word_count': word_count
            }
        
        except Exception as e:
            logger.error(f"Multilingual BERT analysis failed: {e}")
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
        results = []
        
        languages = [self.detect_language(text) for text in texts]
        
        vi_indices = [i for i, lang in enumerate(languages) if lang == 'vi']
        other_indices = [i for i, lang in enumerate(languages) if lang != 'vi']
        
        if vi_indices and self.vi_analyzer:
            vi_texts = [texts[i] for i in vi_indices]
            vi_cleaned = [self.clean_text(t) for t in vi_texts]
            
            try:
                vi_results = self.vi_analyzer(vi_cleaned, batch_size=batch_size)
                
                for i, idx in enumerate(vi_indices):
                    result = vi_results[i]
                    label_map = {
                        'POS': 'positive',
                        'NEG': 'negative',
                        'NEU': 'neutral',
                        'POSITIVE': 'positive',
                        'NEGATIVE': 'negative',
                        'NEUTRAL': 'neutral'
                    }
                    
                    results.append({
                        'index': idx,
                        'sentiment': label_map.get(result['label'].upper(), 'neutral'),
                        'sentiment_score': float(result['score']),
                        'language': 'vi',
                        'analysis_model': 'phobert',
                        'cleaned_content': vi_cleaned[i],
                        'word_count': self.count_words(vi_cleaned[i])
                    })
            except Exception as e:
                logger.error(f"Batch PhoBERT analysis failed: {e}")
                for idx in vi_indices:
                    results.append({
                        'index': idx,
                        **self.analyze_sentiment(texts[idx], 'vi')
                    })
        
        if other_indices and self.multilingual_analyzer:
            other_texts = [texts[i] for i in other_indices]
            other_cleaned = [self.clean_text(t) for t in other_texts]
            
            try:
                other_results = self.multilingual_analyzer(other_cleaned, batch_size=batch_size)
                
                for i, idx in enumerate(other_indices):
                    result = other_results[i]
                    
                    label = result['label']
                    if 'star' in label.lower():
                        stars = int(label.split()[0])
                        if stars >= 4:
                            sentiment = 'positive'
                        elif stars <= 2:
                            sentiment = 'negative'
                        else:
                            sentiment = 'neutral'
                    else:
                        sentiment = result['label'].lower()
                    
                    results.append({
                        'index': idx,
                        'sentiment': sentiment,
                        'sentiment_score': float(result['score']),
                        'language': languages[idx],
                        'analysis_model': 'mbert',
                        'cleaned_content': other_cleaned[i],
                        'word_count': self.count_words(other_cleaned[i])
                    })
            except Exception as e:
                logger.error(f"Batch multilingual BERT analysis failed: {e}")
                for idx in other_indices:
                    results.append({
                        'index': idx,
                        **self.analyze_sentiment(texts[idx], languages[idx])
                    })
        
        results.sort(key=lambda x: x['index'])
        
        return [{k: v for k, v in r.items() if k != 'index'} for r in results]
