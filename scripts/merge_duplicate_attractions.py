"""
Script to find and merge duplicate attractions
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.database.connection import SessionLocal
from app.models.tourist_attraction import TouristAttraction
from app.models.comment import Comment
from sqlalchemy import func, or_
import logging
from unidecode import unidecode

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def normalize_name(name: str) -> str:
    """Normalize attraction name for comparison"""
    # Remove accents
    normalized = unidecode(name.lower())
    # Remove special characters
    normalized = ''.join(c for c in normalized if c.isalnum() or c.isspace())
    # Remove extra spaces
    normalized = ' '.join(normalized.split())
    return normalized


def find_duplicate_attractions():
    """Find attractions with similar names in the same province"""
    db = SessionLocal()
    
    try:
        attractions = db.query(TouristAttraction).all()
        
        # Group by normalized name and province
        groups = {}
        for attr in attractions:
            key = (normalize_name(attr.name), attr.province_id)
            if key not in groups:
                groups[key] = []
            groups[key].append(attr)
        
        # Find duplicates
        duplicates = {k: v for k, v in groups.items() if len(v) > 1}
        
        if duplicates:
            logger.info(f"Found {len(duplicates)} groups of duplicate attractions:")
            for (normalized_name, province_id), attrs in duplicates.items():
                logger.info(f"\n  Group: {normalized_name} (Province {province_id})")
                for attr in attrs:
                    comment_count = db.query(func.count(Comment.id)).filter(
                        Comment.attraction_id == attr.id
                    ).scalar()
                    logger.info(f"    - ID {attr.id}: '{attr.name}' ({comment_count} comments)")
        else:
            logger.info("No duplicate attractions found!")
        
        return duplicates
        
    finally:
        db.close()


def merge_attractions(keep_id: int, remove_ids: list[int]):
    """Merge duplicate attractions by moving all comments to one"""
    db = SessionLocal()
    
    try:
        # Get attractions
        keep_attr = db.query(TouristAttraction).filter(TouristAttraction.id == keep_id).first()
        remove_attrs = db.query(TouristAttraction).filter(TouristAttraction.id.in_(remove_ids)).all()
        
        if not keep_attr:
            logger.error(f"Keep attraction {keep_id} not found!")
            return False
        
        if len(remove_attrs) != len(remove_ids):
            logger.error(f"Some remove attractions not found!")
            return False
        
        logger.info(f"Merging into: {keep_attr.name} (ID: {keep_id})")
        
        # Move comments
        total_moved = 0
        for remove_attr in remove_attrs:
            comment_count = db.query(func.count(Comment.id)).filter(
                Comment.attraction_id == remove_attr.id
            ).scalar()
            
            if comment_count > 0:
                db.query(Comment).filter(
                    Comment.attraction_id == remove_attr.id
                ).update({'attraction_id': keep_id})
                
                logger.info(f"  Moved {comment_count} comments from '{remove_attr.name}' (ID: {remove_attr.id})")
                total_moved += comment_count
            
            # Delete the duplicate
            db.delete(remove_attr)
            logger.info(f"  Deleted duplicate: '{remove_attr.name}' (ID: {remove_attr.id})")
        
        db.commit()
        
        logger.info(f"✓ Successfully merged {len(remove_ids)} duplicates, moved {total_moved} comments")
        return True
        
    except Exception as e:
        logger.error(f"Error merging: {e}")
        db.rollback()
        return False
    finally:
        db.close()


def main():
    """Find and offer to merge duplicates"""
    logger.info("Searching for duplicate attractions...")
    
    duplicates = find_duplicate_attractions()
    
    if not duplicates:
        return
    
    logger.info("\n" + "=" * 80)
    logger.info("MERGE RECOMMENDATIONS")
    logger.info("=" * 80)
    
    for (normalized_name, province_id), attrs in duplicates.items():
        # Sort by comment count (descending) to keep the one with most comments
        db = SessionLocal()
        try:
            attrs_with_counts = []
            for attr in attrs:
                count = db.query(func.count(Comment.id)).filter(
                    Comment.attraction_id == attr.id
                ).scalar()
                attrs_with_counts.append((attr, count))
            
            attrs_with_counts.sort(key=lambda x: x[1], reverse=True)
            
            keep = attrs_with_counts[0][0]
            remove = [a[0] for a in attrs_with_counts[1:]]
            
            logger.info(f"\nGroup: {normalized_name}")
            logger.info(f"  KEEP: '{keep.name}' (ID: {keep.id}, {attrs_with_counts[0][1]} comments)")
            logger.info(f"  MERGE:")
            for i, (attr, count) in enumerate(attrs_with_counts[1:], 1):
                logger.info(f"    {i}. '{attr.name}' (ID: {attr.id}, {count} comments)")
            
            # Auto-merge
            user_input = input(f"\n  Merge these duplicates? [y/N]: ").strip().lower()
            if user_input == 'y':
                success = merge_attractions(keep.id, [a.id for a in remove])
                if success:
                    logger.info("  ✓ Merge completed!")
                else:
                    logger.error("  ✗ Merge failed!")
            else:
                logger.info("  Skipped")
                
        finally:
            db.close()


if __name__ == "__main__":
    main()
