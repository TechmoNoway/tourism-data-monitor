"""
Detect and update language for all comments without language info

This will detect language for 5,321 comments that currently have language=None
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from app.core.config import settings
from app.models.comment import Comment
from langdetect import detect, LangDetectException
import time

def detect_language_safe(text):
    """Safely detect language with fallback"""
    if not text or len(text.strip()) < 10:
        return 'unknown'
    
    try:
        lang = detect(text)
        return lang
    except LangDetectException:
        return 'unknown'

def main():
    print("=" * 80)
    print("Detecting Language for Comments")
    print("=" * 80)
    
    # Create database connection
    engine = create_engine(settings.DATABASE_URL)
    Session = sessionmaker(bind=engine)
    session = Session()
    
    try:
        # Find comments without language
        print("\nFinding comments without language...")
        comments_no_lang = session.query(Comment).filter(
            Comment.language.is_(None)
        ).all()
        
        total = len(comments_no_lang)
        print(f"   Found {total} comments without language")
        
        if total == 0:
            print("\nAll comments already have language!")
            return
        
        # Process comments
        print(f"\nDetecting language for {total} comments...")
        print("   (This may take a few minutes...)")
        
        processed = 0
        detected = 0
        unknown = 0
        errors = 0
        
        start_time = time.time()
        
        for idx, comment in enumerate(comments_no_lang, 1):
            try:
                lang = detect_language_safe(comment.content)
                comment.language = lang
                
                if lang == 'unknown':
                    unknown += 1
                else:
                    detected += 1
                
                processed += 1
                
                # Progress indicator
                if processed % 100 == 0:
                    elapsed = time.time() - start_time
                    rate = processed / elapsed
                    remaining = (total - processed) / rate if rate > 0 else 0
                    print(f"      Progress: {processed}/{total} ({processed/total*100:.1f}%) - "
                          f"Detected: {detected}, Unknown: {unknown} - "
                          f"ETA: {remaining:.0f}s")
                
                # Commit in batches
                if processed % 500 == 0:
                    session.commit()
                    print(f"      Committed batch at {processed}")
                    
            except Exception as e:
                print(f"      Error processing comment {comment.id}: {e}")
                comment.language = 'unknown'
                errors += 1
                unknown += 1
                continue
        
        # Final commit
        session.commit()
        
        elapsed = time.time() - start_time
        
        print("\n" + "=" * 80)
        print("Language Detection Complete!")
        print("=" * 80)
        print(f"\nResults:")
        print(f"   Total processed: {processed}")
        print(f"   Successfully detected: {detected}")
        print(f"   Unknown/Too short: {unknown}")
        print(f"   Errors: {errors}")
        print(f"   Time taken: {elapsed:.1f}s ({processed/elapsed:.1f} comments/sec)")
        
        # Show top languages
        print(f"\nChecking updated language distribution...")
        from sqlalchemy import func
        langs = session.query(
            Comment.language, 
            func.count(Comment.id)
        ).group_by(Comment.language).all()
        
        print(f"\n   Top languages now:")
        for lang, count in sorted(langs, key=lambda x: x[1], reverse=True)[:10]:
            print(f"      {lang or 'None':10s}: {count:5d}")
        
        print("\nNext step:")
        print("   Run process_missing_topics.py to classify topics for new non-Vietnamese comments")
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        session.rollback()
        raise
    finally:
        session.close()

if __name__ == '__main__':
    main()
