"""Check language distribution in comments"""
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from app.database.connection import SessionLocal
from sqlalchemy import text

def main():
    db = SessionLocal()
    try:
        # Get language distribution
        result = db.execute(text("""
            SELECT language, COUNT(*) 
            FROM comments 
            WHERE language IS NOT NULL 
            GROUP BY language 
            ORDER BY COUNT(*) DESC
        """))
        results = result.fetchall()
        
        print("\nLanguage distribution:")
        print("-" * 40)
        total = 0
        for lang, count in results:
            print(f"{lang}: {count:,}")
            total += count
        
        # Check NULL languages
        result = db.execute(text("SELECT COUNT(*) FROM comments WHERE language IS NULL"))
        null_count = result.fetchone()[0]
        if null_count > 0:
            print(f"NULL: {null_count:,}")
        
        print("-" * 40)
        print(f"Total: {total + null_count:,}")
        
        # Calculate Vietnamese vs Non-Vietnamese
        vi_count = next((c for lang, c in results if lang == 'vi'), 0)
        non_vi_count = total - vi_count
        
        print("\nSummary:")
        print(f"Vietnamese (vi): {vi_count:,}")
        print(f"Non-Vietnamese: {non_vi_count:,}")
        print(f"Percentage Vietnamese: {vi_count / total * 100:.1f}%")
        print(f"Percentage Non-Vietnamese: {non_vi_count / total * 100:.1f}%")
    finally:
        db.close()

if __name__ == "__main__":
    main()
