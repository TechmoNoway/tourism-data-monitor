from typing import Dict, Any, Optional
import logging
import re


class CommentFilter:
    
    def __init__(
        self, 
        min_length: int = 3,  
        min_chars: int = 10  
    ):
        self.min_length = min_length
        self.min_chars = min_chars
        self.logger = logging.getLogger("comment_filter")
        
        self.spam_keywords = {
            # Sales/promotional
            'mua ngay', 'order now', 'inbox', 'liên hệ', 'contact',
            'freeship', 'miễn phí ship', 'giảm giá', 'sale off', 'discount',
            'khuyến mãi', 'promotion', 'voucher', 'coupon', 'mã giảm',
            
            # Spam/scam
            'click link', 'theo dõi', 'follow me', 'subscribe',
            'nhấn vào', 'xem thêm tại', 'visit my', 'check out my',
            'casino', 'cá cược', 'betting', 'gambling',
            'kiếm tiền', 'make money', 'get rich', 'làm giàu',
            
            # Advertising
            'quảng cáo', 'bán', 'selling', 'advertisement',
            'dịch vụ', 'service', 'cho thuê', 'for rent',
            'phân phối', 'distributor', 'đại lý', 'agency',
            
            # Suspicious patterns
            'telegram', 'whatsapp', 'zalo', 'viber',
            'www.', 'http', '.com', '.vn', '.net',
            '0123', '0124', '0125', '0126', '0127', '0128', '0129',  # Phone patterns
            '090', '091', '092', '093', '094', '096', '097', '098', '099'
        }
        
        self.meaningless_patterns = {
            r'^[\U0001F300-\U0001F9FF\s]+$',  # Only emoji
            r'^[!@#$%^&*()_+=\-\[\]{};:\'",.<>?/\\|`~\s]+$',  # Only symbols
            
            r'^(.)\1{2,}$',
            
            r'^\d+$',
            
            r'^[a-zA-Z]\s*$',
        }
        
        self.meaningless_words = {
            'ok', 'okay', 'k', 'oke', 'okê',
            'yes', 'no', 'yep', 'nope',
            'ừ', 'à', 'ơ', 'ô', 'ồ', 'ừm',
            'good', 'nice', 'wow', 'lol', 'haha', 'hihi', 'hehe',
            'đẹp', 'hay', 'tốt', 'ngon',
            'first', 'đầu', '1st',
        }
        
    def is_valid_comment(self, comment_text: str, author_name: Optional[str] = None) -> tuple[bool, str]:
        if not comment_text:
            return False, "empty_content"
        
        text_clean = comment_text.strip()
        text_lower = text_clean.lower()
        
        if len(text_clean) < self.min_chars:
            return False, f"too_short_chars_{len(text_clean)}"
        
        words = re.findall(r'\b\w+\b', text_clean)
        if len(words) < self.min_length:
            return False, f"too_short_words_{len(words)}"
        
        for pattern in self.meaningless_patterns:
            if re.match(pattern, text_clean):
                return False, "meaningless_pattern"
        
        if len(words) == 1 and words[0].lower() in self.meaningless_words:
            return False, f"meaningless_word_{words[0]}"
        
        spam_found = []
        for keyword in self.spam_keywords:
            if keyword in text_lower:
                spam_found.append(keyword)
        
        if spam_found:
            if len(spam_found) >= 2:
                return False, f"spam_keywords_{','.join(spam_found[:3])}"
            
            strong_spam = ['http', 'www.', '.com', '.vn', '.net', 'telegram', 'whatsapp', 'zalo',
                          'mua ngay', 'order now', 'click link', 'inbox']
            if any(s in spam_found for s in strong_spam):
                return False, f"spam_strong_{spam_found[0]}"
        
      
        unique_chars = len(set(text_clean.replace(' ', '')))
        total_chars = len(text_clean.replace(' ', ''))
        if total_chars > 20: 
            if unique_chars / total_chars < 0.25: 
                return False, "excessive_repetition"
        
        if len(text_clean) > 20: 
            alpha_chars = [c for c in text_clean if c.isalpha()]
            if len(alpha_chars) > 10:
                caps_ratio = sum(1 for c in alpha_chars if c.isupper()) / len(alpha_chars)
                if caps_ratio > 0.7: 
                    return False, "all_caps_spam"
        
        special_chars = sum(1 for c in text_clean if not c.isalnum() and not c.isspace())
        if len(text_clean) > 0:
            special_ratio = special_chars / len(text_clean)
            if special_ratio > 0.5: 
                return False, "excessive_special_chars"
        
        return True, "valid"
    
    def filter_comments(
        self, 
        raw_comments: list[Dict[str, Any]], 
        text_field: str = 'text',
        author_field: str = 'author_name'
    ) -> tuple[list[Dict[str, Any]], Dict[str, int]]:
        filtered = []
        rejection_reasons = {}
        
        for comment in raw_comments:
            text = comment.get(text_field, '')
            author = comment.get(author_field)
            
            is_valid, reason = self.is_valid_comment(text, author)
            
            if is_valid:
                filtered.append(comment)
            else:
                rejection_reasons[reason] = rejection_reasons.get(reason, 0) + 1
        
        stats = {
            'total': len(raw_comments),
            'kept': len(filtered),
            'filtered': len(raw_comments) - len(filtered),
            'rejection_reasons': rejection_reasons,
            'keep_rate': f"{len(filtered)/len(raw_comments)*100:.1f}%" if raw_comments else "0%"
        }
        
        if len(filtered) < len(raw_comments):
            self.logger.info(
                f"Filtered {len(raw_comments)} → {len(filtered)} comments "
                f"({stats['keep_rate']} kept)"
            )
            if rejection_reasons:
                top_reasons = sorted(rejection_reasons.items(), key=lambda x: x[1], reverse=True)[:3]
                self.logger.debug(f"Top rejection reasons: {top_reasons}")
        
        return filtered, stats
    
    def get_quality_score(self, comment_text: str) -> float:
        if not comment_text:
            return 0.0
        
        is_valid, _ = self.is_valid_comment(comment_text)
        if not is_valid:
            return 0.0
        
        score = 0.5 
        
        text_clean = comment_text.strip()
        text_lower = text_clean.lower()
        words = re.findall(r'\b\w+\b', text_clean)
        
        if len(words) >= 10:
            score += 0.2
        elif len(words) >= 5:
            score += 0.1
        
        quality_keywords = [
            'đẹp', 'tuyệt', 'tốt', 'hay', 'ngon', 'beautiful', 'amazing', 'great',
            'trải nghiệm', 'experience', 'review', 'đánh giá',
            'thích', 'like', 'love', 'recommend', 'nên đến', 'should visit'
        ]
        quality_found = sum(1 for kw in quality_keywords if kw in text_lower)
        score += min(0.3, quality_found * 0.1)
        
        if any(p in text_clean for p in ['.', '!', '?', ',']):
            score += 0.05
        
        emoji_count = len(re.findall(r'[\U0001F300-\U0001F9FF]', text_clean))
        if 1 <= emoji_count <= 5:
            score += 0.05
        
        return min(1.0, score)
    
    def assess_comment_quality(
        self, 
        comment_text: str, 
        author_name: Optional[str] = None
    ) -> tuple[str, float, Optional[str]]:
        if not comment_text:
            return ('spam', 0.0)
        
        is_valid, reason = self.is_valid_comment(comment_text)
        
        if not is_valid:
            if any(indicator in reason for indicator in ['spam_', 'all_caps']):
                return ('spam', 0.0)
            else:
                return ('low', 0.2)
        
        score = self.get_quality_score(comment_text)
        
        if score >= 0.7:
            tier = 'high'
        elif score >= 0.4:
            tier = 'medium'
        else:
            tier = 'low'
        
        return (tier, score)
