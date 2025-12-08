"""
Classify tourism type for each attraction based on name and keywords
Automatically categorize attractions into: beach, mountain, historical, cultural, nature, urban, adventure
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from app.database.connection import get_db
from app.models.tourist_attraction import TouristAttraction
from app.models.comment import Comment
from app.models.social_post import SocialPost
from app.models.province import Province
from sqlalchemy import func

# Tourism type classification keywords
TOURISM_TYPE_KEYWORDS = {
    'beach': {
        'keywords': ['bãi biển', 'biển', 'beach', 'bờ biển', 'hòn', 'đảo', 'cồn cát', 'mũi', 'vịnh'],
        'priority': 1
    },
    'mountain': {
        'keywords': ['núi', 'đồi', 'cao nguyên', 'đèo', 'đỉnh', 'langbiang', 'bidoup', 'fansipan'],
        'priority': 2
    },
    'historical': {
        'keywords': ['di tích', 'lịch sử', 'cổ', 'dinh', 'phủ', 'thành', 'pháo đài', 'lăng', 'tháp chàm'],
        'priority': 3
    },
    'cultural': {
        'keywords': ['chùa', 'đền', 'thiền viện', 'pagoda', 'temple', 'làng', 'bảo tàng', 'museum'],
        'priority': 4
    },
    'nature': {
        'keywords': ['thác', 'hồ', 'vườn quốc gia', 'rừng', 'sinh thái', 'suối', 'khu bảo tồn', 'thung lũng'],
        'priority': 5
    },
    'urban': {
        'keywords': ['chợ', 'phố', 'trung tâm', 'quảng trường', 'công viên', 'khu phố', 'đường'],
        'priority': 6
    },
    'adventure': {
        'keywords': ['zipline', 'canyoning', 'paragliding', 'dù lượn', 'leo núi', 'mạo hiểm', 'thể thao'],
        'priority': 7
    }
}

def classify_tourism_type(name: str, description: str = None) -> str:
    """Classify tourism type based on name and description"""
    
    text = name.lower()
    if description:
        text += " " + description.lower()
    
    # Find matching types
    matches = []
    for tourism_type, data in TOURISM_TYPE_KEYWORDS.items():
        for keyword in data['keywords']:
            if keyword in text:
                matches.append((tourism_type, data['priority']))
                break
    
    # Return type with highest priority (lowest number = higher priority)
    if matches:
        matches.sort(key=lambda x: x[1])
        return matches[0][0]
    
    # Default based on category or name patterns
    if any(word in text for word in ['resort', 'khách sạn', 'hotel']):
        return 'urban'
    
    return None  # Unknown type

def update_tourism_types():
    """Update tourism_type for all attractions"""
    
    db = next(get_db())
    
    try:
        attractions = db.query(TouristAttraction).all()
        
        print(f"Processing {len(attractions)} attractions...\n")
        
        stats = {
            'beach': 0,
            'mountain': 0,
            'historical': 0,
            'cultural': 0,
            'nature': 0,
            'urban': 0,
            'adventure': 0,
            'unknown': 0
        }
        
        for attraction in attractions:
            tourism_type = classify_tourism_type(attraction.name, attraction.description)
            
            if tourism_type:
                attraction.tourism_type = tourism_type
                stats[tourism_type] += 1
                print(f"✅ {attraction.name[:50]:50s} → {tourism_type}")
            else:
                attraction.tourism_type = None
                stats['unknown'] += 1
                print(f"⚠️  {attraction.name[:50]:50s} → unknown")
        
        db.commit()
        
        print("\n" + "="*70)
        print("CLASSIFICATION SUMMARY")
        print("="*70)
        
        total = len(attractions)
        for tourism_type, count in sorted(stats.items(), key=lambda x: x[1], reverse=True):
            percentage = (count / total * 100) if total > 0 else 0
            print(f"{tourism_type:15s}: {count:3d} ({percentage:5.1f}%)")
        
        print(f"\nTotal: {total}")
        
        return True
        
    except Exception as e:
        db.rollback()
        print(f"\n❌ Error: {e}")
        return False
    finally:
        db.close()

def update_attraction_stats():
    """Update total_comments and total_posts for each attraction"""
    
    db = next(get_db())
    
    try:
        print("\n" + "="*70)
        print("UPDATING ATTRACTION STATISTICS")
        print("="*70 + "\n")
        
        attractions = db.query(TouristAttraction).all()
        
        for attraction in attractions:
            # Count comments
            comment_count = db.query(func.count(Comment.id)).filter(
                Comment.attraction_id == attraction.id
            ).scalar()
            
            # Count posts
            post_count = db.query(func.count(SocialPost.id)).filter(
                SocialPost.attraction_id == attraction.id
            ).scalar()
            
            # Update
            attraction.total_comments = comment_count
            attraction.total_reviews = post_count  # Using total_reviews for posts count
            
            print(f"{attraction.name[:40]:40s} → {post_count:4d} posts, {comment_count:5d} comments")
        
        db.commit()
        print(f"\n✅ Updated statistics for {len(attractions)} attractions")
        
        return True
        
    except Exception as e:
        db.rollback()
        print(f"\n❌ Error: {e}")
        return False
    finally:
        db.close()

def update_province_stats():
    """Update statistics for each province"""
    
    db = next(get_db())
    
    try:
        print("\n" + "="*70)
        print("UPDATING PROVINCE STATISTICS")
        print("="*70 + "\n")
        
        provinces = db.query(Province).all()
        
        for province in provinces:
            # Count attractions
            attraction_count = db.query(func.count(TouristAttraction.id)).filter(
                TouristAttraction.province_id == province.id,
                TouristAttraction.is_active.is_(True)
            ).scalar()
            
            # Count total comments
            comment_count = db.query(func.count(Comment.id)).join(
                TouristAttraction, Comment.attraction_id == TouristAttraction.id
            ).filter(
                TouristAttraction.province_id == province.id
            ).scalar()
            
            # Count total posts
            post_count = db.query(func.count(SocialPost.id)).join(
                TouristAttraction, SocialPost.attraction_id == TouristAttraction.id
            ).filter(
                TouristAttraction.province_id == province.id
            ).scalar()
            
            # Update
            province.total_attractions = attraction_count
            province.total_comments = comment_count
            province.total_posts = post_count
            
            print(f"{province.name:20s} → {attraction_count:3d} attractions, {post_count:4d} posts, {comment_count:5d} comments")
        
        db.commit()
        print(f"\n✅ Updated statistics for {len(provinces)} provinces")
        
        return True
        
    except Exception as e:
        db.rollback()
        print(f"\n❌ Error: {e}")
        return False
    finally:
        db.close()

if __name__ == "__main__":
    print("="*70)
    print("TOURISM DATA CLASSIFICATION & STATISTICS UPDATE")
    print("="*70 + "\n")
    
    # Step 1: Classify tourism types
    print("STEP 1: Classifying Tourism Types")
    print("-"*70)
    success = update_tourism_types()
    
    if not success:
        print("\n❌ Failed to classify tourism types")
        sys.exit(1)
    
    # Step 2: Update attraction statistics
    success = update_attraction_stats()
    
    if not success:
        print("\n❌ Failed to update attraction statistics")
        sys.exit(1)
    
    # Step 3: Update province statistics
    success = update_province_stats()
    
    if not success:
        print("\n❌ Failed to update province statistics")
        sys.exit(1)
    
    print("\n" + "="*70)
    print("✅ ALL UPDATES COMPLETED SUCCESSFULLY!")
    print("="*70)
    
    sys.exit(0)
