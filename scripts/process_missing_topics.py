"""
Process all comments without topics using rule-based classifier
This will generate weak labels for training XLM-RoBERTa on non-Vietnamese comments

Usage:
    python scripts/process_missing_topics.py
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from app.core.config import settings
from app.models.comment import Comment
from app.services.topic_classifier import TopicClassifier

def main():
    print("=" * 80)
    print("Processing Comments Without Topics")
    print("=" * 80)
    
    # Create database connection
    engine = create_engine(settings.DATABASE_URL)
    Session = sessionmaker(bind=engine)
    session = Session()
    
    # Initialize rule-based topic classifier (without ML models)
    print("\nInitializing rule-based topic classifier...")
    classifier = TopicClassifier()
    
    try:
        # Find comments without topics (None or empty array)
        print("\nFinding comments without topics...")
        all_comments = session.query(Comment).filter(
            Comment.is_meaningful == True
        ).all()
        
        comments_without_topics = [
            c for c in all_comments 
            if c.topics is None or (isinstance(c.topics, list) and len(c.topics) == 0)
        ]
        
        total = len(comments_without_topics)
        print(f"   Found {total} meaningful comments without topics")
        
        if total == 0:
            print("\nAll comments already have topics!")
            return
        
        # Count by language
        language_counts = {}
        for comment in comments_without_topics:
            lang = comment.language or 'unknown'
            language_counts[lang] = language_counts.get(lang, 0) + 1
        
        print(f"\nLanguage distribution:")
        for lang, count in sorted(language_counts.items(), key=lambda x: x[1], reverse=True)[:10]:
            print(f"      {lang}: {count}")
        
        # Process comments
        print(f"\nProcessing {total} comments...")
        print("   (This may take a few minutes...)")
        
        processed = 0
        skipped = 0
        updated = 0
        
        for idx, comment in enumerate(comments_without_topics, 1):
            try:
                # Classify using rule-based (even if ML models are loaded, force rule-based)
                topics = classifier._classify_rule_based(comment.content, comment.language or 'en')
                
                if topics:
                    comment.topics = topics
                    updated += 1
                else:
                    skipped += 1
                
                processed += 1
                
                # Progress indicator
                if processed % 100 == 0:
                    print(f"      Progress: {processed}/{total} ({processed/total*100:.1f}%) - Updated: {updated}, Skipped: {skipped}")
                
                # Commit in batches
                if processed % 500 == 0:
                    session.commit()
                    print(f"      Committed batch at {processed}")
                    
            except Exception as e:
                print(f"      Error processing comment {comment.id}: {e}")
                skipped += 1
                continue
        
        # Final commit
        session.commit()
        
        print("\n" + "=" * 80)
        print("Processing Complete!")
        print("=" * 80)
        print(f"\nResults:")
        print(f"   Total processed: {processed}")
        print(f"   Updated with topics: {updated}")
        print(f"   Skipped (no topics detected): {skipped}")
        print(f"   Success rate: {updated/processed*100:.1f}%")
        
        print("\nNext steps:")
        print("   1. Retrain XLM-RoBERTa with more data:")
        print("      python training/train_xlm_topic_classifier.py --exclude_vietnamese --epochs 5")
        print("   2. Compare performance before/after")
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        session.rollback()
        raise
    finally:
        session.close()

if __name__ == '__main__':
    main()
