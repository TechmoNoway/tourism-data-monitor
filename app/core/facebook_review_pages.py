VIETNAM_TOURISM_REVIEW_PAGES = {
    "foody": {
        "name": "Foody.vn",
        "url": "https://www.facebook.com/foody.vn",
        "type": "review_platform",
        "focus": "food_travel",
        "engagement": "very_high",
        "description": "Major food & travel review platform in Vietnam"
    },
    "vnexpress_travel": {
        "name": "VnExpress Du lịch",
        "url": "https://www.facebook.com/vnexpresstravel",
        "type": "news_travel",
        "focus": "travel_news",
        "engagement": "high",
        "description": "Travel section of Vietnam's biggest news site"
    },
    "dulich24": {
        "name": "Du lịch 24h",
        "url": "https://www.facebook.com/dulich24h",
        "type": "travel_media",
        "focus": "travel_guide",
        "engagement": "high",
        "description": "Popular travel guide and review page"
    },
    "kenh14_travel": {
        "name": "Kenh14 Travel",
        "url": "https://www.facebook.com/kenh14travel",
        "type": "youth_travel",
        "focus": "youth_tourism",
        "engagement": "very_high",
        "description": "Youth-focused travel content, high engagement"
    }
}

# Popular travel bloggers/influencers
VIETNAM_TRAVEL_INFLUENCERS = {
    # Add top travel bloggers here
    # Format: "username": {"name": "", "url": "", "followers": ""}
}

REVIEW_KEYWORDS_VI = [
    # Explicit review terms
    "review", "đánh giá", "nhận xét",
    "trải nghiệm", "experience",
    
    # Recommendation terms
    "có nên đi", "nên ghé", "đáng đi",
    "worth it", "recommend", "giới thiệu",
    
    # Descriptive review terms
    "tuyệt vời", "đẹp", "ấn tượng",
    "thất vọng", "không nên", "tệ",
    
    # Visit-related
    "đã đi", "vừa đi", "mới đi",
    "check-in", "checkin"
]

SKIP_KEYWORDS = [
    "giảm giá", "sale", "khuyến mãi",
    "mua ngay", "đặt tour", "liên hệ",
    "inbox", "zalo", "hotline"
]

def should_collect_post(post_text: str) -> bool:
    text_lower = post_text.lower()
    
    # Skip if contains spam keywords
    for skip_word in SKIP_KEYWORDS:
        if skip_word in text_lower:
            return False
    
    # Collect if contains review keywords
    for review_word in REVIEW_KEYWORDS_VI:
        if review_word in text_lower:
            return True
    
    return False
