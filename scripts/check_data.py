"""
Utility script to check and verify data in the database.
Combines functionality from check_all_data.py and verify_db.py
"""
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from app.database.connection import get_db
from app.models.tourist_attraction import TouristAttraction
from app.models.social_post import SocialPost
from app.models.comment import Comment
from app.models.province import Province
from sqlalchemy import func


def check_database_stats():
    """Display comprehensive database statistics"""
    db = next(get_db())
    
    print("\n" + "="*70)
    print("DATABASE STATISTICS")
    print("="*70)
    
    total_posts = db.query(SocialPost).count()
    total_comments = db.query(Comment).count()
    total_attractions = db.query(TouristAttraction).count()
    total_provinces = db.query(Province).count()
    
    print("\nTOTALS:")
    print(f"  Provinces: {total_provinces}")
    print(f"  Attractions: {total_attractions}")
    print(f"  Posts: {total_posts}")
    print(f"  Comments: {total_comments}")
    
    print("\nPOSTS BY PLATFORM:")
    posts_by_platform = db.query(
        SocialPost.platform,
        func.count(SocialPost.id).label('count')
    ).group_by(SocialPost.platform).all()
    
    for platform, count in posts_by_platform:
        print(f"  {platform}: {count} posts")
    
    print("\nBREAKDOWN BY ATTRACTION:")
    attractions = db.query(TouristAttraction).all()
    
    for attraction in attractions:
        post_count = db.query(SocialPost).filter(
            SocialPost.attraction_id == attraction.id
        ).count()
        
        comment_count = db.query(Comment).filter(
            Comment.attraction_id == attraction.id
        ).count()
        
        province = db.query(Province).filter(Province.id == attraction.province_id).first()
        province_name = province.name if province else "Unknown"
        
        print(f"  {attraction.name} ({province_name}): {post_count} posts, {comment_count} comments")
    
    print("\nCOMMENTS BY PLATFORM:")
    comments_by_platform = db.query(
        Comment.platform,
        func.count(Comment.id).label('count')
    ).group_by(Comment.platform).all()
    
    for platform, count in comments_by_platform:
        print(f"  {platform}: {count} comments")
    
    print("\nAVERAGES:")
    avg_posts_per_attraction = total_posts / total_attractions if total_attractions > 0 else 0
    avg_comments_per_attraction = total_comments / total_attractions if total_attractions > 0 else 0
    avg_comments_per_post = total_comments / total_posts if total_posts > 0 else 0
    
    print(f"  Posts per attraction: {avg_posts_per_attraction:.1f}")
    print(f"  Comments per attraction: {avg_comments_per_attraction:.1f}")
    print(f"  Comments per post: {avg_comments_per_post:.1f}")
    
    print("\n" + "="*70 + "\n")
    
    db.close()


def verify_database_connection():
    """Verify database connection and structure"""
    try:
        db = next(get_db())
        
        provinces = db.query(Province).count()
        attractions = db.query(TouristAttraction).count()
        posts = db.query(SocialPost).count()
        comments = db.query(Comment).count()
        
        print("Database connection successful!")
        print(f"Found {provinces} provinces, {attractions} attractions, {posts} posts, {comments} comments")
        
        db.close()
        return True
        
    except Exception as e:
        print(f"Database connection failed: {e}")
        return False


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Check database data')
    parser.add_argument('--verify', action='store_true', help='Verify database connection only')
    args = parser.parse_args()
    
    if args.verify:
        verify_database_connection()
    else:
        check_database_stats()
