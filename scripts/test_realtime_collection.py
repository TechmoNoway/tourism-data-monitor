"""Quick test to add sample comments and verify chart updates"""
from datetime import datetime, timezone
from app.database.connection import SessionLocal
from app.models.comment import Comment
from app.models.tourist_attraction import TouristAttraction
from sqlalchemy import func

def main():
    print("🚀 Starting quick collection test...")
    
    db = SessionLocal()
    
    try:
        # Get Bà Nà Hills attraction (ID = 3)
        attraction = db.query(TouristAttraction).filter(TouristAttraction.id == 3).first()
        
        if not attraction:
            print("❌ Attraction not found")
            return
        
        print(f"\n📍 Target: {attraction.name}")
        
        # Add 3 sample comments with sentiment
        sample_comments = [
            {
                "content": "Bà Nà Hills rất đẹp và tuyệt vời! Cầu Vàng ấn tượng!",
                "sentiment": "positive",
                "sentiment_score": 0.95
            },
            {
                "content": "Cảnh đẹp nhưng đông người quá, vé hơi đắt",
                "sentiment": "neutral",
                "sentiment_score": 0.5
            },
            {
                "content": "Tuyệt vời! Phong cảnh đẹp như tranh vẽ",
                "sentiment": "positive",
                "sentiment_score": 0.92
            }
        ]
        
        added = 0
        for idx, comment_data in enumerate(sample_comments, 1):
            comment = Comment(
                platform="test",
                platform_comment_id=f"test_{datetime.now().timestamp()}_{idx}",
                attraction_id=attraction.id,
                content=comment_data["content"],
                cleaned_content=comment_data["content"],
                author=f"Test User {idx}",
                sentiment=comment_data["sentiment"],
                sentiment_score=comment_data["sentiment_score"],
                analysis_model="test",
                scraped_at=datetime.now(timezone.utc),
                is_meaningful=True,
                quality_tier="high",
                language="vi"
            )
            db.add(comment)
            added += 1
            print(f"  ✅ Added comment {idx}: {comment_data['content'][:50]}...")
        
        db.commit()
        
        # Update sentiment counts
        positive_count = db.query(func.count(Comment.id)).filter(
            Comment.attraction_id == attraction.id,
            Comment.sentiment == 'positive'
        ).scalar() or 0
        
        negative_count = db.query(func.count(Comment.id)).filter(
            Comment.attraction_id == attraction.id,
            Comment.sentiment == 'negative'
        ).scalar() or 0
        
        neutral_count = db.query(func.count(Comment.id)).filter(
            Comment.attraction_id == attraction.id,
            Comment.sentiment == 'neutral'
        ).scalar() or 0
        
        attraction.positive_count = positive_count
        attraction.negative_count = negative_count
        attraction.neutral_count = neutral_count
        db.commit()
        
        print(f"\n📊 Updated sentiment counts:")
        print(f"   Positive: {positive_count}")
        print(f"   Negative: {negative_count}")
        print(f"   Neutral: {neutral_count}")
        print(f"\n✅ Test complete! Refresh the page to see chart updates with the '7days' period.")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        db.rollback()
    finally:
        db.close()

if __name__ == "__main__":
    main()
