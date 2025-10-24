
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from app.database.connection import get_db
from app.models.tourist_attraction import TouristAttraction
from app.models.social_post import SocialPost
from app.models.comment import Comment
from app.models.province import Province
from sqlalchemy import func


def check_database_stats():
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
    
    print("\nSENTIMENT ANALYSIS:")
    analyzed_comments = db.query(Comment).filter(Comment.sentiment.isnot(None)).count()
    
    if analyzed_comments > 0:
        print(f"  Analyzed: {analyzed_comments}/{total_comments} comments ({analyzed_comments/total_comments*100:.1f}%)")
        
        sentiments = db.query(
            Comment.sentiment,
            func.count(Comment.id).label('count')
        ).filter(Comment.sentiment.isnot(None)).group_by(Comment.sentiment).all()
        
        print("\n  Sentiment Distribution:")
        for sentiment, count in sentiments:
            percentage = count / analyzed_comments * 100
            bar = "█" * int(percentage / 2)
            print(f"    {sentiment:8s}: {count:4d} ({percentage:5.1f}%) {bar}")
        
        languages = db.query(
            Comment.language,
            func.count(Comment.id).label('count')
        ).filter(Comment.language.isnot(None)).group_by(Comment.language).order_by(func.count(Comment.id).desc()).all()
        
        print("\n  Language Distribution (Top 10):")
        for lang, count in languages[:10]:
            percentage = count / analyzed_comments * 100
            print(f"    {lang:8s}: {count:4d} ({percentage:5.1f}%)")
        
        models = db.query(
            Comment.analysis_model,
            func.count(Comment.id).label('count')
        ).filter(Comment.analysis_model.isnot(None)).group_by(Comment.analysis_model).all()
        
        print("\n  Models Used:")
        for model, count in models:
            percentage = count / analyzed_comments * 100
            print(f"    {model:15s}: {count:4d} ({percentage:5.1f}%)")
        
        avg_positive_score = db.query(func.avg(Comment.sentiment_score)).filter(Comment.sentiment == 'positive').scalar()
        avg_negative_score = db.query(func.avg(Comment.sentiment_score)).filter(Comment.sentiment == 'negative').scalar()
        avg_neutral_score = db.query(func.avg(Comment.sentiment_score)).filter(Comment.sentiment == 'neutral').scalar()
        
        print("\n  Average Confidence Scores:")
        if avg_positive_score:
            print(f"    Positive: {avg_positive_score:.3f}")
        if avg_negative_score:
            print(f"    Negative: {avg_negative_score:.3f}")
        if avg_neutral_score:
            print(f"    Neutral:  {avg_neutral_score:.3f}")
        
        # Topic classification statistics
        comments_with_topics = db.query(Comment).filter(Comment.topics.isnot(None)).count()
        
        if comments_with_topics > 0:
            print(f"\nTOPIC CLASSIFICATION:")
            print(f"  Comments with topics: {comments_with_topics}/{analyzed_comments} ({comments_with_topics/analyzed_comments*100:.1f}%)")
            
            # Count topics (JSONB array contains)
            all_topics = {}
            comments_with_topics_data = db.query(Comment).filter(Comment.topics.isnot(None)).all()
            
            for comment in comments_with_topics_data:
                if comment.topics:
                    for topic in comment.topics:
                        all_topics[topic] = all_topics.get(topic, 0) + 1
            
            print("\n  Topic Distribution:")
            for topic, count in sorted(all_topics.items(), key=lambda x: x[1], reverse=True):
                percentage = count / comments_with_topics * 100
                bar = "▓" * int(percentage / 3)
                print(f"    {topic:15s}: {count:4d} ({percentage:5.1f}%) {bar}")
            
            print(f"\n  Total topic mentions: {sum(all_topics.values())}")
            print(f"  Avg topics per comment: {sum(all_topics.values())/comments_with_topics:.1f}")
    else:
        print("  No sentiment analysis performed yet")
        print("  Run: python scripts/analyze_sentiment.py")
    
    print("\n" + "="*70 + "\n")
    
    db.close()


def verify_database_connection():
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
