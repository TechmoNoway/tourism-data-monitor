import re
import os
import torch
from typing import Dict, List
import logging

logger = logging.getLogger(__name__)

# Check if trained models exist
PHOBERT_MODEL_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'training', 'models', 'phobert_best_model.pt')
XLM_MODEL_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'training', 'models', 'xlm_best_model.pt')


class TopicClassifier:
    def __init__(self):
        # Try to load trained models
        self.use_phobert = False
        self.use_xlm = False
        self.phobert_model = None
        self.xlm_model = None
        self.phobert_tokenizer = None
        self.xlm_tokenizer = None
        self.translator = None
        self.phobert_threshold = 0.5
        self.xlm_threshold = 0.5
        self.ml_device = None
        self.ml_topics = ['scenery', 'food', 'service', 'pricing', 'facilities', 'activities', 'accessibility']
        
        # Load PhoBERT for Vietnamese
        if os.path.exists(PHOBERT_MODEL_PATH):
            try:
                self._load_phobert_model()
                logger.info("Loaded PhoBERT for Vietnamese")
            except Exception as e:
                logger.warning(f"Failed to load PhoBERT: {e}")
        
        # Load XLM-RoBERTa for other languages
        if os.path.exists(XLM_MODEL_PATH):
            try:
                self._load_xlm_model()
                logger.info("Loaded XLM-RoBERTa for multilingual")
            except Exception as e:
                logger.warning(f"Failed to load XLM-RoBERTa: {e}")
        
        if not self.use_phobert and not self.use_xlm:
            logger.info("Using rule-based classification (no trained models found)")
        
        # Vietnamese keywords for each topic
        self.topic_keywords = {
            'scenery': {
                'vi': [
                    'cảnh', 'đẹp', 'view', 'phong cảnh', 'thiên nhiên', 'núi', 'biển',
                    'bãi biển', 'hoàng hôn', 'bình minh', 'cảnh quan', 'khung cảnh',
                    'tuyệt đẹp', 'thơ mộng', 'hùng vĩ', 'yên bình', 'trong lành',
                    'cây cối', 'rừng', 'hoa', 'vườn', 'sông', 'hồ', 'thác',
                    'mây', 'trời', 'xanh', 'màu sắc', 'bầu trời', 'chụp ảnh',
                    'instagram', 'sống ảo', 'check in', 'checkin'
                ],
                'en': [
                    'view', 'scenery', 'landscape', 'nature', 'beautiful', 'gorgeous',
                    'stunning', 'breathtaking', 'picturesque', 'scenic', 'mountain',
                    'beach', 'sea', 'ocean', 'sunset', 'sunrise', 'sky', 'cloud',
                    'tree', 'forest', 'flower', 'garden', 'river', 'lake', 'waterfall',
                    'photo', 'picture', 'instagram', 'photography'
                ]
            },
            
            'food': {
                'vi': [
                    'đồ ăn', 'món ăn', 'ăn uống', 'thức ăn', 'nhà hàng', 'quán ăn',
                    'buffet', 'bữa sáng', 'bữa trưa', 'bữa tối', 'hải sản',
                    'ngon', 'tươi', 'vị', 'hương vị', 'chất lượng', 'phục vụ',
                    'menu', 'thực đơn', 'đặc sản', 'địa phương', 'truyền thống',
                    'nướng', 'lẩu', 'phở', 'cơm', 'bánh', 'canh', 'rau',
                    'thịt', 'cá', 'tôm', 'cua', 'ốc', 'mực', 'hàu',
                    'cafe', 'cà phê', 'trà', 'nước', 'đồ uống', 'bia', 'rượu'
                ],
                'en': [
                    'food', 'meal', 'dish', 'cuisine', 'restaurant', 'dining',
                    'breakfast', 'lunch', 'dinner', 'buffet', 'seafood',
                    'delicious', 'tasty', 'yummy', 'fresh', 'flavor', 'taste',
                    'menu', 'local', 'traditional', 'specialty', 'authentic',
                    'grilled', 'fried', 'soup', 'rice', 'noodle', 'bread',
                    'meat', 'fish', 'shrimp', 'crab', 'squid', 'oyster',
                    'cafe', 'coffee', 'tea', 'drink', 'beer', 'wine'
                ]
            },
            
            'service': {
                'vi': [
                    'nhân viên', 'phục vụ', 'staff', 'tiếp tân', 'lễ tân',
                    'thái độ', 'nhiệt tình', 'thân thiện', 'chu đáo', 'tận tâm',
                    'chuyên nghiệp', 'lịch sự', 'niềm nở', 'hỗ trợ', 'giúp đỡ',
                    'tư vấn', 'hướng dẫn', 'chăm sóc', 'quan tâm', 'check in',
                    'check out', 'đặt phòng', 'booking', 'tour', 'hướng dẫn viên',
                    'lạnh nhạt', 'thờ ơ', 'vô tâm', 'thiếu chuyên nghiệp',
                    'chậm', 'nhanh', 'kịp thời', 'đúng giờ'
                ],
                'en': [
                    'staff', 'service', 'employee', 'receptionist', 'front desk',
                    'attitude', 'friendly', 'helpful', 'attentive', 'professional',
                    'polite', 'kind', 'courteous', 'welcoming', 'support', 'assist',
                    'guide', 'care', 'check-in', 'checkout', 'booking', 'reservation',
                    'tour guide', 'rude', 'unfriendly', 'slow', 'fast', 'quick',
                    'efficient', 'responsive'
                ]
            },
            
            'pricing': {
                'vi': [
                    'giá', 'giá cả', 'tiền', 'chi phí', 'phí', 'hóa đơn', 'bill',
                    'đắt', 'rẻ', 'mắc', 'hợp lý', 'phải chăng', 'xứng đáng',
                    'tốt', 'value', 'worth', 'đáng giá', 'đắt đỏ', 'quá đắt',
                    'cắt cổ', 'chém', 'ép giá', 'discount', 'giảm giá', 'khuyến mãi',
                    'combo', 'package', 'gói', 'vé', 'ticket', 'entrance fee',
                    'phí vào cửa', 'miễn phí', 'free', 'tiết kiệm', 'tốn kém'
                ],
                'en': [
                    'price', 'cost', 'fee', 'charge', 'bill', 'payment', 'money',
                    'expensive', 'cheap', 'affordable', 'reasonable', 'worth',
                    'value', 'overpriced', 'pricey', 'costly', 'budget',
                    'discount', 'promotion', 'deal', 'package', 'combo',
                    'ticket', 'entrance', 'admission', 'free', 'complimentary'
                ]
            },
            
            'accessibility': {
                'vi': [
                    'đường', 'đường đi', 'đường xá', 'giao thông', 'di chuyển',
                    'xe', 'xe máy', 'ô tô', 'taxi', 'grab', 'xe bus', 'xe buýt',
                    'đậu xe', 'bãi đỗ', 'parking', 'xa', 'gần', 'khoảng cách',
                    'vị trí', 'location', 'địa điểm', 'trung tâm', 'trung tâm thành phố',
                    'sân bay', 'bến xe', 'nhà ga', 'bến tàu', 'cảng',
                    'dễ tìm', 'khó tìm', 'lạc đường', 'maps', 'google maps',
                    'chỉ đường', 'hướng dẫn', 'biển báo', 'cách', 'km', 'phút'
                ],
                'en': [
                    'road', 'street', 'traffic', 'transportation', 'transport',
                    'car', 'taxi', 'bus', 'shuttle', 'parking', 'park',
                    'location', 'distance', 'far', 'near', 'close', 'convenient',
                    'accessible', 'access', 'downtown', 'center', 'central',
                    'airport', 'station', 'port', 'terminal',
                    'easy to find', 'hard to find', 'lost', 'maps', 'direction',
                    'sign', 'signage', 'km', 'miles', 'minutes', 'walk'
                ]
            },
            
            'facilities': {
                'vi': [
                    'phòng', 'room', 'phòng ngủ', 'giường', 'nệm', 'ga trải giường',
                    'toilet', 'nhà vệ sinh', 'wc', 'phòng tắm', 'vòi sen', 'bồn tắm',
                    'nước nóng', 'điều hòa', 'máy lạnh', 'ac', 'quạt', 'tivi', 'tv',
                    'wifi', 'internet', 'mạng', 'bể bơi', 'pool', 'hồ bơi',
                    'gym', 'phòng tập', 'spa', 'sauna', 'massage',
                    'sạch sẽ', 'vệ sinh', 'bẩn', 'cũ', 'mới', 'hiện đại',
                    'tiện nghi', 'trang thiết bị', 'thiết bị', 'cơ sở vật chất',
                    'khăn', 'dầu gội', 'sữa tắm', 'kem đánh răng', 'bàn chải',
                    'rộng', 'rộng rãi', 'chật', 'hẹp', 'thoải mái', 'ấm cúng'
                ],
                'en': [
                    'room', 'bedroom', 'bed', 'mattress', 'sheet', 'pillow',
                    'toilet', 'bathroom', 'shower', 'bathtub', 'hot water',
                    'air conditioning', 'ac', 'fan', 'tv', 'television',
                    'wifi', 'internet', 'pool', 'swimming pool',
                    'gym', 'fitness', 'spa', 'sauna', 'massage',
                    'clean', 'dirty', 'hygiene', 'sanitation', 'old', 'new', 'modern',
                    'facilities', 'amenities', 'equipment', 'infrastructure',
                    'towel', 'shampoo', 'soap', 'toothpaste', 'toothbrush',
                    'spacious', 'cramped', 'comfortable', 'cozy'
                ]
            },
            
            'activities': {
                'vi': [
                    'hoạt động', 'activity', 'vui chơi', 'giải trí', 'thư giãn',
                    'tham quan', 'du lịch', 'khám phá', 'trải nghiệm', 'experience',
                    'chơi', 'lặn', 'bơi', 'leo núi', 'đi bộ', 'hiking', 'trekking',
                    'chèo thuyền', 'kayak', 'lướt ván', 'surfing', 'câu cá', 'fishing',
                    'xe đạp', 'cycling', 'zipline', 'cầu treo', 'cáp treo',
                    'tắm biển', 'tắm nắng', 'đi dạo', 'shopping', 'mua sắm',
                    'chụp ảnh', 'photo', 'tham quan', 'visit', 'tour',
                    'buồn', 'vui', 'thú vị', 'hấp dẫn', 'nhàm chán', 'boring'
                ],
                'en': [
                    'activity', 'activities', 'entertainment', 'fun', 'relax',
                    'sightseeing', 'tour', 'explore', 'experience', 'adventure',
                    'play', 'dive', 'diving', 'swim', 'swimming', 'climb', 'hiking',
                    'trekking', 'kayak', 'kayaking', 'surf', 'surfing', 'fishing',
                    'cycling', 'bike', 'zipline', 'cable car', 'gondola',
                    'beach', 'sunbathe', 'walk', 'stroll', 'shopping', 'shop',
                    'photo', 'photography', 'visit', 'interesting', 'exciting',
                    'boring', 'dull', 'amazing', 'awesome'
                ]
            }
        }
        
        logger.info("TopicClassifier initialized with 7 topics")
    
    def classify_topics(self, text: str, language: str = 'vi') -> List[str]:
        """Main topic classification method - uses appropriate ML model based on language"""
        if not text or not text.strip():
            return []
        
        # Use ML models if available
        if self.use_phobert or self.use_xlm:
            try:
                return self._classify_with_ml(text, language)
            except Exception as e:
                logger.error(f"ML classification failed: {e}, fallback to rule-based")
                return self._classify_rule_based(text, language)
        
        # Fallback to rule-based
        return self._classify_rule_based(text, language)
    
    def classify_with_scores(self, text: str, language: str = 'vi') -> Dict[str, int]:
        text_lower = text.lower()
        topic_scores = {}
        
        # Map language codes
        lang_key = 'vi' if language in ['vi', 'vie', 'vietnamese'] else 'en'
        
        for topic, keywords_dict in self.topic_keywords.items():
            keywords = keywords_dict.get(lang_key, keywords_dict['en'])
            
            match_count = sum(1 for keyword in keywords if keyword in text_lower)
            
            if match_count > 0:
                topic_scores[topic] = match_count
        
        return topic_scores
    
    def get_aspect_sentiments(
        self, 
        text: str, 
        topics: List[str], 
        overall_sentiment: str,
        language: str = 'vi'
    ) -> Dict[str, str]:
        if not topics:
            return {}
        
        if len(topics) == 1:
            return {topics[0]: overall_sentiment}
        
        aspect_sentiments = {}
        text_lower = text.lower()
        
        positive_words_vi = ['tốt', 'đẹp', 'hay', 'ok', 'ổn', 'ngon', 'sạch', 'rộng']
        positive_words_en = ['good', 'great', 'nice', 'excellent', 'amazing', 'clean']
        
        negative_words_vi = ['tệ', 'xấu', 'dở', 'kém', 'bẩn', 'cũ', 'đắt', 'chật']
        negative_words_en = ['bad', 'poor', 'terrible', 'dirty', 'old', 'expensive']
        
        positive_words = positive_words_vi if language == 'vi' else positive_words_en
        negative_words = negative_words_vi if language == 'vi' else negative_words_en
        
        for topic in topics:
            lang_key = 'vi' if language == 'vi' else 'en'
            topic_keywords = self.topic_keywords[topic].get(lang_key, [])
            
            sentences = re.split(r'[.!?;,]', text_lower)
            topic_sentences = [s for s in sentences if any(kw in s for kw in topic_keywords[:5])]
            
            if not topic_sentences:
                aspect_sentiments[topic] = overall_sentiment
                continue
            
            topic_text = ' '.join(topic_sentences)
            
            positive_count = sum(1 for word in positive_words if word in topic_text)
            negative_count = sum(1 for word in negative_words if word in topic_text)
            
            if positive_count > negative_count:
                aspect_sentiments[topic] = 'positive'
            elif negative_count > positive_count:
                aspect_sentiments[topic] = 'negative'
            else:
                aspect_sentiments[topic] = overall_sentiment
        
        return aspect_sentiments
    
    def classify_batch(self, texts: List[str], languages: List[str]) -> List[Dict]:
        results = []
        
        for text, lang in zip(texts, languages):
            topics = self.classify_topics(text, lang)
            topic_scores = self.classify_with_scores(text, lang)
            
            
            results.append({
                'topics': topics,
                'topic_scores': topic_scores
            })
        
        return results
    
    def _load_phobert_model(self):
        """Load trained PhoBERT model for Vietnamese"""
        from training.train_phobert_tourism_topic_classifier import PhoBERTTopicClassifier
        from transformers import AutoTokenizer
        
        if self.ml_device is None:
            self.ml_device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        checkpoint = torch.load(PHOBERT_MODEL_PATH, map_location=self.ml_device)
        
        self.phobert_model = PhoBERTTopicClassifier(n_classes=7)
        self.phobert_model.load_state_dict(checkpoint['model_state_dict'])
        self.phobert_model.to(self.ml_device)
        self.phobert_model.eval()
        
        self.phobert_tokenizer = AutoTokenizer.from_pretrained('vinai/phobert-base')
        self.phobert_threshold = checkpoint.get('threshold', 0.5)
        self.use_phobert = True
    
    def _load_xlm_model(self):
        """Load trained XLM-RoBERTa model for multilingual"""
        from training.train_xlm_tourism_topic_classifier import XLMRoBERTaTopicClassifier
        from transformers import AutoTokenizer
        
        if self.ml_device is None:
            self.ml_device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        checkpoint = torch.load(XLM_MODEL_PATH, map_location=self.ml_device)
        
        self.xlm_model = XLMRoBERTaTopicClassifier(n_classes=7)
        self.xlm_model.load_state_dict(checkpoint['model_state_dict'])
        self.xlm_model.to(self.ml_device)
        self.xlm_model.eval()
        
        self.xlm_tokenizer = AutoTokenizer.from_pretrained('xlm-roberta-base')
        self.xlm_threshold = checkpoint.get('threshold', 0.5)
        self.use_xlm = True
    
    def _classify_with_phobert(self, text: str) -> List[str]:
        """Classify using PhoBERT model (Vietnamese only)"""
        encoding = self.phobert_tokenizer(
            text,
            add_special_tokens=True,
            max_length=256,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt'
        )
        
        input_ids = encoding['input_ids'].to(self.ml_device)
        attention_mask = encoding['attention_mask'].to(self.ml_device)
        
        with torch.no_grad():
            logits = self.phobert_model(input_ids, attention_mask)
            probs = torch.sigmoid(logits).cpu().numpy()[0]
        
        topics = [self.ml_topics[i] for i, p in enumerate(probs) if p >= self.phobert_threshold]
        return topics if topics else self._classify_rule_based(text, 'vi')
    
    def _classify_with_xlm(self, text: str) -> List[str]:
        """Classify using XLM-RoBERTa model (multilingual)"""
        encoding = self.xlm_tokenizer(
            text,
            add_special_tokens=True,
            max_length=256,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt'
        )
        
        input_ids = encoding['input_ids'].to(self.ml_device)
        attention_mask = encoding['attention_mask'].to(self.ml_device)
        
        with torch.no_grad():
            logits = self.xlm_model(input_ids, attention_mask)
            probs = torch.sigmoid(logits).cpu().numpy()[0]
        
        topics = [self.ml_topics[i] for i, p in enumerate(probs) if p >= self.xlm_threshold]
        return topics if topics else self._classify_rule_based(text, 'en')
    
    def _classify_with_ml(self, text: str, language: str) -> List[str]:
        """Classify using appropriate ML model based on language"""
        # Vietnamese → Use PhoBERT
        if language == 'vi' and self.use_phobert:
            try:
                return self._classify_with_phobert(text)
            except Exception as e:
                logger.error(f"PhoBERT failed: {e}")
                return self._classify_rule_based(text, language)
        
        # Other languages → Use XLM-RoBERTa
        elif self.use_xlm:
            try:
                return self._classify_with_xlm(text)
            except Exception as e:
                logger.error(f"XLM-RoBERTa failed: {e}")
                return self._classify_rule_based(text, language)
        
        # Fallback
        return self._classify_rule_based(text, language)
    
    def _old_classify_with_ml(self, text: str, language: str) -> List[str]:
        """Old method - kept for backwards compatibility (translation approach)"""
        # Translate to Vietnamese if needed
        if language != 'vi':
            try:
                text = self.translator.translate(text[:500])
            except Exception as e:
                logger.warning(f"Translation failed: {e}")
                return self._classify_rule_based(text, language)
        
        # Tokenize
        encoding = self.ml_tokenizer(
            text,
            add_special_tokens=True,
            max_length=256,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt'
        )
        
        input_ids = encoding['input_ids'].to(self.ml_device)
        attention_mask = encoding['attention_mask'].to(self.ml_device)
        
        # Predict
        with torch.no_grad():
            logits = self.ml_model(input_ids, attention_mask)
            probs = torch.sigmoid(logits).cpu().numpy()[0]
        
        # Get topics above threshold
        topics = [self.ml_topics[i] for i, p in enumerate(probs) if p >= self.ml_threshold]
        
        # Fallback if no topics
        if not topics:
            return self._classify_rule_based(text, 'vi')
        
        return topics
    
    def _classify_rule_based(self, text: str, language: str) -> List[str]:
        """Rule-based classification with translation support for multilingual"""
        if not text or not isinstance(text, str):
            return []
            
        text_lower = text.lower()
        detected_topics = []
        
        # Check if we have keywords for this language
        lang_key = 'vi' if language in ['vi', 'vie', 'vietnamese'] else 'en'
        
        # If language is not Vi or En, translate to English first
        if language not in ['vi', 'vie', 'vietnamese', 'en', 'eng', 'english', None]:
            try:
                from deep_translator import GoogleTranslator
                translator = GoogleTranslator(source='auto', target='en')
                translated = translator.translate(text[:500])
                if translated and isinstance(translated, str):
                    text_lower = translated.lower()
                    lang_key = 'en'
                    logger.info(f"Translated {language} to English for rule-based classification")
            except Exception as e:
                logger.warning(f"Translation failed for {language}: {e}")
                # Fallback to English keywords without translation
                lang_key = 'en'
        
        # Apply keyword matching
        for topic, keywords_dict in self.topic_keywords.items():
            keywords = keywords_dict.get(lang_key, keywords_dict['en'])
            if any(keyword in text_lower for keyword in keywords):
                detected_topics.append(topic)
        
        return detected_topics
    
    def classify(self, text: str, language: str = 'vi') -> List[str]:
        """Public method for backwards compatibility"""
        return self.classify_topics(text, language)

