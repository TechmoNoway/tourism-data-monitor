from typing import List, Dict, Any, Optional
import logging
from datetime import datetime, timedelta


class RelevanceFilter:
    
    def __init__(self, min_relevance_score: float = 0.2):  # Giảm từ 0.3 → 0.2
        self.min_relevance_score = min_relevance_score
        self.logger = logging.getLogger("relevance_filter")
        
        # Common spam/irrelevant keywords (Vietnamese + English)
        self.spam_keywords = {
            'spam', 'quảng cáo', 'bán hàng', 'mua ngay', 'liên hệ',
            'inbox', 'order', 'freeship', 'giảm giá', 'sale off',
            'click link', 'theo dõi', 'follow', 'subscribe',
            'giveaway', 'tặng quà', 'trúng thưởng', 'casino', 'cá cược'
        }
        
        # Tourism-related keywords (Vietnamese)
        self.tourism_keywords = {
            'du lịch', 'travel', 'tour', 'khách sạn', 'hotel', 'resort',
            'nghỉ dưỡng', 'tham quan', 'check in', 'checkin', 'điểm đến',
            'destination', 'đẹp', 'beautiful', 'phong cảnh', '景色',
            'kỳ nghỉ', 'vacation', 'holiday', 'trải nghiệm', 'experience',
            'review', 'đánh giá', 'thăm', 'visit', 'ghé', 'đi chơi'
        }
    
    def generate_keywords(
        self, 
        attraction_name: str, 
        province_name: str,
        additional_keywords: Optional[List[str]] = None
    ) -> List[str]:
        """
        Generate smart search keywords for an attraction
        
        Args:
            attraction_name: Name of tourist attraction (e.g., "Hồ Xuân Hương")
            province_name: Province name (e.g., "Lâm Đồng")
            additional_keywords: Extra keywords to include
            
        Returns:
            List of search keywords optimized for each platform
        """
        keywords = []
        
        # Core keywords - attraction name variations
        keywords.append(attraction_name)
        
        # Remove accents for alternative search
        keywords.append(self._remove_accents(attraction_name))
        
        # Location-based keywords
        keywords.append(f"{attraction_name} {province_name}")
        keywords.append(f"{attraction_name} du lịch")
        keywords.append(f"{attraction_name} travel")
        keywords.append(f"{attraction_name} review")
        keywords.append(f"{attraction_name} check in")
        
        # Platform-specific keywords
        # For YouTube
        keywords.append(f"du lịch {attraction_name}")
        keywords.append(f"review {attraction_name}")
        keywords.append(f"tham quan {attraction_name}")
        
        # For Facebook/TikTok
        keywords.append(f"#{attraction_name.replace(' ', '')}")
        keywords.append(f"#{province_name.replace(' ', '')}")
        
        # Additional keywords
        if additional_keywords:
            keywords.extend(additional_keywords)
        
        # Remove duplicates and empty strings
        keywords = list(set([k.strip() for k in keywords if k.strip()]))
        
        self.logger.info(f"Generated {len(keywords)} keywords for {attraction_name}")
        return keywords
    
    def score_relevance(
        self,
        content: str,
        attraction_name: str,
        province_name: str,
        title: Optional[str] = None,
        hashtags: Optional[List[str]] = None,
        author: Optional[str] = None
    ) -> float:
        """
        Calculate relevance score for a piece of content
        
        Args:
            content: Main text content (description, caption, etc.)
            attraction_name: Target attraction name
            province_name: Target province
            title: Optional title (for YouTube videos)
            hashtags: Optional list of hashtags
            author: Optional author name
            
        Returns:
            Score from 0 (not relevant) to 1 (very relevant)
        """
        score = 0.0
        content_lower = content.lower() if content else ""
        title_lower = title.lower() if title else ""
        
        #  Check attraction name mention (40% weight)
        attraction_lower = attraction_name.lower()
        if attraction_lower in content_lower:
            score += 0.4
        elif attraction_lower in title_lower:
            score += 0.3
        elif self._fuzzy_match(attraction_lower, content_lower):
            score += 0.2
        
        # Check province mention (20% weight)
        province_lower = province_name.lower()
        if province_lower in content_lower or province_lower in title_lower:
            score += 0.2
        
        # Check tourism keywords (20% weight)
        tourism_found = sum(1 for kw in self.tourism_keywords if kw in content_lower or kw in title_lower)
        if tourism_found > 0:
            score += min(0.2, tourism_found * 0.05)
        
        # Check hashtags (10% weight)
        if hashtags:
            hashtag_str = ' '.join(hashtags).lower()
            if attraction_lower in hashtag_str or province_lower in hashtag_str:
                score += 0.1
        
        # Bonus for quality indicators (10% weight)
        quality_keywords = ['review', 'đánh giá', 'trải nghiệm', 'tham quan', 'check in']
        quality_found = sum(1 for kw in quality_keywords if kw in content_lower or kw in title_lower)
        if quality_found > 0:
            score += min(0.1, quality_found * 0.03)
        
        # Penalty for spam (-0.5 max)
        spam_found = sum(1 for kw in self.spam_keywords if kw in content_lower or kw in title_lower)
        if spam_found > 0:
            score -= min(0.5, spam_found * 0.15)
        
        # Penalty for very short content (-0.2)
        if content and len(content) < 20:
            score -= 0.2
        
        # Normalize to 0-1 range
        return max(0.0, min(1.0, score))
    
    def filter_posts(
        self,
        posts: List[Dict[str, Any]],
        attraction_name: str,
        province_name: str
    ) -> List[Dict[str, Any]]:
        """
        Filter list of posts to keep only relevant ones
        
        Args:
            posts: List of raw posts from collectors
            attraction_name: Target attraction
            province_name: Target province
            
        Returns:
            Filtered list of relevant posts with relevance scores
        """
        filtered = []
        
        for post in posts:
            # Extract content fields
            content = post.get('message', '') or post.get('description', '') or post.get('text', '')
            title = post.get('title', '') or post.get('snippet', {}).get('title', '')
            hashtags = post.get('hashtags', [])
            author = post.get('author', '') or post.get('page_name', '')
            
            # Calculate relevance score
            score = self.score_relevance(
                content=content,
                attraction_name=attraction_name,
                province_name=province_name,
                title=title,
                hashtags=hashtags,
                author=author
            )
            
            # Keep if above threshold
            if score >= self.min_relevance_score:
                post['relevance_score'] = score
                filtered.append(post)
                self.logger.debug(f"Kept post (score={score:.2f}): {content[:50]}...")
            else:
                self.logger.debug(f"Filtered out post (score={score:.2f}): {content[:50]}...")
        
        # Sort by relevance score (highest first)
        filtered.sort(key=lambda x: x.get('relevance_score', 0), reverse=True)
        
        self.logger.info(f"Filtered {len(posts)} posts → {len(filtered)} relevant posts")
        return filtered
    
    def filter_by_date(
        self,
        posts: List[Dict[str, Any]],
        days_back: int = 90
    ) -> List[Dict[str, Any]]:
        """
        Filter posts by date - keep only recent content
        
        Args:
            posts: List of posts
            days_back: Number of days to look back (default: 90)
            
        Returns:
            Posts from last N days
        """
        cutoff_date = datetime.utcnow() - timedelta(days=days_back)
        filtered = []
        
        for post in posts:
            # Try different date fields
            post_date = None
            for field in ['created_time', 'post_date', 'timestamp', 'create_time', 'publishedAt']:
                if field in post and post[field]:
                    try:
                        if isinstance(post[field], str):
                            post_date = datetime.fromisoformat(post[field].replace('Z', '+00:00'))
                        elif isinstance(post[field], (int, float)):
                            post_date = datetime.fromtimestamp(post[field])
                        else:
                            post_date = post[field]
                        break
                    except Exception:
                        continue
            
            # Keep if recent or no date info
            if post_date is None or post_date >= cutoff_date:
                filtered.append(post)
        
        self.logger.info(f"Date filter: {len(posts)} → {len(filtered)} posts (last {days_back} days)")
        return filtered
    
    def remove_duplicates(
        self,
        posts: List[Dict[str, Any]],
        key_field: str = 'post_id'
    ) -> List[Dict[str, Any]]:
        seen = set()
        unique = []
        
        for post in posts:
            key = post.get(key_field) or post.get('id') or post.get('video_id')
            
            if not key:
                content = post.get('message', '') or post.get('description', '')
                key = hash(content[:100])
            
            if key not in seen:
                seen.add(key)
                unique.append(post)
        
        self.logger.info(f"Removed {len(posts) - len(unique)} duplicates")
        return unique
    
    def _fuzzy_match(self, target: str, text: str, threshold: float = 0.8) -> bool:
        """
        Check if target appears in text with fuzzy matching
        (handles typos, missing accents, etc.)
        """
        # Simple fuzzy: check if most words from target appear in text
        target_words = set(target.split())
        text_words = set(text.split())
        
        if not target_words:
            return False
        
        matches = len(target_words & text_words)
        similarity = matches / len(target_words)
        
        return similarity >= threshold
    
    def _remove_accents(self, text: str) -> str:
        """
        Remove Vietnamese accents for alternative search
        """
        # Vietnamese accent mapping
        accents = {
            'à': 'a', 'á': 'a', 'ả': 'a', 'ã': 'a', 'ạ': 'a',
            'ă': 'a', 'ằ': 'a', 'ắ': 'a', 'ẳ': 'a', 'ẵ': 'a', 'ặ': 'a',
            'â': 'a', 'ầ': 'a', 'ấ': 'a', 'ẩ': 'a', 'ẫ': 'a', 'ậ': 'a',
            'è': 'e', 'é': 'e', 'ẻ': 'e', 'ẽ': 'e', 'ẹ': 'e',
            'ê': 'e', 'ề': 'e', 'ế': 'e', 'ể': 'e', 'ễ': 'e', 'ệ': 'e',
            'ì': 'i', 'í': 'i', 'ỉ': 'i', 'ĩ': 'i', 'ị': 'i',
            'ò': 'o', 'ó': 'o', 'ỏ': 'o', 'õ': 'o', 'ọ': 'o',
            'ô': 'o', 'ồ': 'o', 'ố': 'o', 'ổ': 'o', 'ỗ': 'o', 'ộ': 'o',
            'ơ': 'o', 'ờ': 'o', 'ớ': 'o', 'ở': 'o', 'ỡ': 'o', 'ợ': 'o',
            'ù': 'u', 'ú': 'u', 'ủ': 'u', 'ũ': 'u', 'ụ': 'u',
            'ư': 'u', 'ừ': 'u', 'ứ': 'u', 'ử': 'u', 'ữ': 'u', 'ự': 'u',
            'ỳ': 'y', 'ý': 'y', 'ỷ': 'y', 'ỹ': 'y', 'ỵ': 'y',
            'đ': 'd', 'Đ': 'D'
        }
        
        for accented, plain in accents.items():
            text = text.replace(accented, plain)
            text = text.replace(accented.upper(), plain.upper())
        
        return text


class PlatformKeywordOptimizer:
    @staticmethod
    def optimize_for_youtube(keywords: List[str]) -> List[str]:
        """
        Optimize keywords for YouTube search
        YouTube prefers longer, more descriptive queries
        """
        optimized = []
        
        for keyword in keywords:
            optimized.append(keyword)
            
            if len(keyword.split()) == 1:
                optimized.append(f"{keyword} travel vlog")
                optimized.append(f"du lịch {keyword}")
                optimized.append(f"{keyword} review 2024")
        
        return list(set(optimized))
    
    @staticmethod
    def optimize_for_facebook(keywords: List[str]) -> List[str]:
        """
        Optimize for Facebook Pages search
        Facebook prefers exact page names and locations
        """
        optimized = []
        
        for keyword in keywords:
            optimized.append(keyword)
            
            # Add "page" suffix for better results
            if 'page' not in keyword.lower():
                optimized.append(f"{keyword} page")
            
            # Location-based
            optimized.append(f"{keyword} Vietnam")
        
        return list(set(optimized))
    
    @staticmethod
    def optimize_for_tiktok(keywords: List[str]) -> List[str]:
        """
        Optimize for TikTok hashtag search
        TikTok prefers hashtags and trending terms
        """
        optimized = []
        
        for keyword in keywords:
            # Add hashtag version
            hashtag = keyword.replace('#', '').replace(' ', '')
            optimized.append(f"#{hashtag}")
            optimized.append(keyword)
            
            # Add trending suffixes
            optimized.append(f"#{hashtag}travel")
            optimized.append(f"#{hashtag}fyp")
        
        return list(set(optimized))


# Helper function for easy use
def create_relevance_filter(min_score: float = 0.3) -> RelevanceFilter:
    return RelevanceFilter(min_relevance_score=min_score)
