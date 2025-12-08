"""Check current database state"""
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from app.database.connection import SessionLocal
from app.models import TouristAttraction, Province, Comment, SocialPost
from sqlalchemy import text, func, or_

db = SessionLocal()

print("\n" + "="*80)
print("CURRENT DATABASE STATE")
print("="*80)

# Provinces and attractions
print("\nATTRACTIONS PER PROVINCE:")
print("-" * 80)
provinces = db.query(Province).all()
total_attractions = 0
for p in provinces:
    count = db.query(TouristAttraction).filter(TouristAttraction.province_id == p.id).count()
    print(f"{p.name:20s}: {count:3d} attractions")
    total_attractions += count

print(f"\n{'Total':20s}: {total_attractions:3d} attractions")
print(f"Target: 100 attractions (25 per province x 4 provinces)")

# Comments
result = db.execute(text("SELECT COUNT(*) FROM comments")).fetchone()
total_comments = result[0]
print(f"\nCOMMENTS:")
print(f"Current: {total_comments:,} comments")
print(f"Target:  20,000 comments")
print(f"Need:    {20000 - total_comments:,} more comments")

# Posts by platform
print(f"\nPOSTS BY PLATFORM:")
print("-" * 80)
platforms = db.query(SocialPost.platform, func.count(SocialPost.id)).group_by(SocialPost.platform).all()
for platform, count in platforms:
    print(f"{platform:20s}: {count:3d} posts")

# Comments by platform
print(f"\nCOMMENTS BY PLATFORM:")
print("-" * 80)
comment_platforms = db.query(Comment.platform, func.count(Comment.id)).group_by(Comment.platform).all()
for platform, count in comment_platforms:
    avg_per_post = count / dict(platforms).get(platform, 1)
    print(f"{platform:20s}: {count:5d} comments (avg {avg_per_post:.1f} per post)")

# Check null columns
print(f"\nNULL/EMPTY COLUMNS CHECK:")
print("-" * 80)

# Check attractions table
attr_sample = db.query(TouristAttraction).first()
if attr_sample:
    print("\nTouristAttraction columns:")
    for col in ['description', 'google_place_id', 'total_reviews', 'total_comments']:
        if col in ['total_reviews', 'total_comments']:
            null_count = db.query(TouristAttraction).filter(
                or_(
                    getattr(TouristAttraction, col).is_(None),
                    getattr(TouristAttraction, col) == 0
                )
            ).count()
        else:
            null_count = db.query(TouristAttraction).filter(
                getattr(TouristAttraction, col).is_(None)
            ).count()
        pct = (null_count / total_attractions * 100) if total_attractions > 0 else 0
        print(f"  {col:20s}: {null_count:3d} null/0 ({pct:.1f}%)")

# Check posts table
total_posts = db.query(SocialPost).count()
if total_posts > 0:
    print(f"\nSocialPost columns (total: {total_posts}):")
    for col in ['content', 'author']:
        null_count = db.query(SocialPost).filter(
            getattr(SocialPost, col).is_(None)
        ).count()
        pct = (null_count / total_posts * 100)
        print(f"  {col:20s}: {null_count:3d} null/0 ({pct:.1f}%)")

db.close()

print("\n" + "="*80)
