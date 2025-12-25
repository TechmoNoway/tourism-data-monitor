"""
Check for and merge duplicate attractions
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.database.connection import SessionLocal
from app.models.tourist_attraction import TouristAttraction
from app.models.comment import Comment
from app.models.social_post import SocialPost
from sqlalchemy import or_, func
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def find_duplicates():
    """Find duplicate attractions"""
    db = SessionLocal()
    
    try:
        # Check for Ba Na Hills duplicates
        bana_duplicates = db.query(TouristAttraction).filter(
            or_(
                TouristAttraction.name.like('%Ba Na Hills%'),
                TouristAttraction.name.like('%Bà Nà Hills%'),
                TouristAttraction.name.like('%Ba Nà Hills%')
            )
        ).all()
        
        logger.info(f"\n{'='*60}")
        logger.info("BA NA HILLS DUPLICATES")
        logger.info(f"{'='*60}")
        for attr in bana_duplicates:
            comments_count = db.query(func.count(Comment.id)).filter(
                Comment.attraction_id == attr.id,
                Comment.is_meaningful == True
            ).scalar()
            logger.info(f"ID: {attr.id} | Name: {attr.name} | Comments: {comments_count}")
        
        # Check for Quảng Trường Lâm Viên duplicates
        lamvien_duplicates = db.query(TouristAttraction).filter(
            TouristAttraction.name.like('%Quảng trường Lâm Viên%')
        ).all()
        
        logger.info(f"\n{'='*60}")
        logger.info("QUẢNG TRƯỜNG LÂM VIÊN DUPLICATES")
        logger.info(f"{'='*60}")
        for attr in lamvien_duplicates:
            comments_count = db.query(func.count(Comment.id)).filter(
                Comment.attraction_id == attr.id,
                Comment.is_meaningful == True
            ).scalar()
            logger.info(f"ID: {attr.id} | Name: {attr.name} | Comments: {comments_count}")
        
        # Check for Hồ Tuyền Lâm duplicates
        tuyenlam_duplicates = db.query(TouristAttraction).filter(
            or_(
                TouristAttraction.name.like('%Hồ Tuyền Lâm%'),
                TouristAttraction.name.like('%Tuyen Lam%')
            )
        ).all()
        
        logger.info(f"\n{'='*60}")
        logger.info("HỒ TUYỀN LÂM DUPLICATES")
        logger.info(f"{'='*60}")
        for attr in tuyenlam_duplicates:
            comments_count = db.query(func.count(Comment.id)).filter(
                Comment.attraction_id == attr.id,
                Comment.is_meaningful == True
            ).scalar()
            logger.info(f"ID: {attr.id} | Name: {attr.name} | Comments: {comments_count}")
        
        # Check for Thác Cam Ly duplicates
        camly_duplicates = db.query(TouristAttraction).filter(
            TouristAttraction.name.like('%Cam Ly%')
        ).all()
        
        logger.info(f"\n{'='*60}")
        logger.info("THÁC CAM LY DUPLICATES")
        logger.info(f"{'='*60}")
        for attr in camly_duplicates:
            comments_count = db.query(func.count(Comment.id)).filter(
                Comment.attraction_id == attr.id,
                Comment.is_meaningful == True
            ).scalar()
            logger.info(f"ID: {attr.id} | Name: {attr.name} | Comments: {comments_count}")
        
        # Check for Cầu Rồng duplicates
        caulong_duplicates = db.query(TouristAttraction).filter(
            or_(
                TouristAttraction.name.like('%Cầu Rồng%'),
                TouristAttraction.name.like('%Dragon Bridge%')
            )
        ).all()
        
        logger.info(f"\n{'='*60}")
        logger.info("CẦU RỒNG DUPLICATES")
        logger.info(f"{'='*60}")
        for attr in caulong_duplicates:
            comments_count = db.query(func.count(Comment.id)).filter(
                Comment.attraction_id == attr.id,
                Comment.is_meaningful == True
            ).scalar()
            logger.info(f"ID: {attr.id} | Name: {attr.name} | Comments: {comments_count}")
        
        # Check for Vườn Hoa Đà Lạt duplicates
        vuonhoa_duplicates = db.query(TouristAttraction).filter(
            TouristAttraction.name.like('%Vườn%hoa%')
        ).all()
        vuonhoa_duplicates = [a for a in vuonhoa_duplicates if 'Đà Lạt' in a.name or 'thành phố' in a.name]
        
        logger.info(f"\n{'='*60}")
        logger.info("VƯỜN HOA ĐÀ LẠT DUPLICATES")
        logger.info(f"{'='*60}")
        for attr in vuonhoa_duplicates:
            comments_count = db.query(func.count(Comment.id)).filter(
                Comment.attraction_id == attr.id,
                Comment.is_meaningful == True
            ).scalar()
            logger.info(f"ID: {attr.id} | Name: {attr.name} | Comments: {comments_count}")
        
        # Check for Làng Cù Lần duplicates
        culan_duplicates = db.query(TouristAttraction).filter(
            TouristAttraction.name.like('%Cù Lần%')
        ).all()
        
        logger.info(f"\n{'='*60}")
        logger.info("LÀNG CÙ LẦN DUPLICATES")
        logger.info(f"{'='*60}")
        for attr in culan_duplicates:
            comments_count = db.query(func.count(Comment.id)).filter(
                Comment.attraction_id == attr.id,
                Comment.is_meaningful == True
            ).scalar()
            logger.info(f"ID: {attr.id} | Name: {attr.name} | Comments: {comments_count}")
        
        # Check for Thung Lũng Tình Yêu duplicates
        thunglungtinhyeu_duplicates = db.query(TouristAttraction).filter(
            TouristAttraction.name.like('%Thung%lũng%Tình%')
        ).all()
        
        logger.info(f"\n{'='*60}")
        logger.info("THUNG LŨNG TÌNH YÊU DUPLICATES")
        logger.info(f"{'='*60}")
        for attr in thunglungtinhyeu_duplicates:
            comments_count = db.query(func.count(Comment.id)).filter(
                Comment.attraction_id == attr.id,
                Comment.is_meaningful == True
            ).scalar()
            logger.info(f"ID: {attr.id} | Name: {attr.name} | Comments: {comments_count}")
        
        return bana_duplicates, lamvien_duplicates, tuyenlam_duplicates, camly_duplicates, caulong_duplicates, vuonhoa_duplicates, culan_duplicates, thunglungtinhyeu_duplicates
        
    finally:
        db.close()


def merge_attractions(keep_id: int, remove_id: int):
    """Merge two attractions - move all data from remove_id to keep_id"""
    db = SessionLocal()
    
    try:
        keep_attr = db.query(TouristAttraction).filter(TouristAttraction.id == keep_id).first()
        remove_attr = db.query(TouristAttraction).filter(TouristAttraction.id == remove_id).first()
        
        if not keep_attr or not remove_attr:
            logger.error("One or both attractions not found!")
            return False
        
        logger.info(f"\nMerging:")
        logger.info(f"  KEEP: {keep_attr.name} (ID: {keep_id})")
        logger.info(f"  REMOVE: {remove_attr.name} (ID: {remove_id})")
        
        # Move all comments
        comments_moved = db.query(Comment).filter(Comment.attraction_id == remove_id).update(
            {Comment.attraction_id: keep_id}
        )
        logger.info(f"  ✓ Moved {comments_moved} comments")
        
        # Move all posts
        posts_moved = db.query(SocialPost).filter(SocialPost.attraction_id == remove_id).update(
            {SocialPost.attraction_id: keep_id}
        )
        logger.info(f"  ✓ Moved {posts_moved} posts")
        
        # Move all analysis logs
        from app.models.analysis_log import AnalysisLog
        logs_moved = db.query(AnalysisLog).filter(AnalysisLog.attraction_id == remove_id).update(
            {AnalysisLog.attraction_id: keep_id}
        )
        logger.info(f"  ✓ Moved {logs_moved} analysis logs")
        
        # Update keep attraction's stats
        meaningful_comments = db.query(func.count(Comment.id)).filter(
            Comment.attraction_id == keep_id,
            Comment.is_meaningful == True
        ).scalar()
        
        keep_attr.total_comments = meaningful_comments
        keep_attr.total_reviews = meaningful_comments
        
        # Use better image if available
        if remove_attr.image_url and not keep_attr.image_url:
            keep_attr.image_url = remove_attr.image_url
            logger.info(f"  ✓ Copied image from removed attraction")
        
        # Delete the duplicate
        db.delete(remove_attr)
        db.commit()
        
        logger.info(f"  ✓ Deleted duplicate attraction")
        logger.info(f"  ✓ Updated stats: {meaningful_comments} meaningful comments")
        
        return True
        
    except Exception as e:
        db.rollback()
        logger.error(f"Error merging attractions: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        return False
    finally:
        db.close()


def main():
    logger.info("="*60)
    logger.info("CHECKING FOR DUPLICATE ATTRACTIONS")
    logger.info("="*60)
    
    bana_duplicates, lamvien_duplicates, tuyenlam_duplicates, camly_duplicates, caulong_duplicates, vuonhoa_duplicates, culan_duplicates, thunglungtinhyeu_duplicates = find_duplicates()
    
    # Handle Ba Na Hills duplicates
    if len(bana_duplicates) > 1:
        logger.info(f"\n{'='*60}")
        logger.info("MERGING BA NA HILLS")
        logger.info(f"{'='*60}")
        
        db = SessionLocal()
        try:
            # Find which one has more comments to keep
            best_bana = max(bana_duplicates, key=lambda x: db.query(func.count(Comment.id)).filter(
                Comment.attraction_id == x.id,
                Comment.is_meaningful == True
            ).scalar())
            
            for duplicate in bana_duplicates:
                if duplicate.id != best_bana.id:
                    merge_attractions(best_bana.id, duplicate.id)
        finally:
            db.close()
    
    # Handle Quảng Trường Lâm Viên duplicates
    if len(lamvien_duplicates) > 1:
        logger.info(f"\n{'='*60}")
        logger.info("MERGING QUẢNG TRƯỜNG LÂM VIÊN")
        logger.info(f"{'='*60}")
        
        db = SessionLocal()
        try:
            # Find which one has more comments to keep
            best_lamvien = max(lamvien_duplicates, key=lambda x: db.query(func.count(Comment.id)).filter(
                Comment.attraction_id == x.id,
                Comment.is_meaningful == True
            ).scalar())
            
            for duplicate in lamvien_duplicates:
                if duplicate.id != best_lamvien.id:
                    merge_attractions(best_lamvien.id, duplicate.id)
        finally:
            db.close()
    
    # Handle Hồ Tuyền Lâm duplicates
    if len(tuyenlam_duplicates) > 1:
        logger.info(f"\n{'='*60}")
        logger.info("MERGING HỒ TUYỀN LÂM")
        logger.info(f"{'='*60}")
        
        db = SessionLocal()
        try:
            # Find which one has more comments to keep
            best_tuyenlam = max(tuyenlam_duplicates, key=lambda x: db.query(func.count(Comment.id)).filter(
                Comment.attraction_id == x.id,
                Comment.is_meaningful == True
            ).scalar())
            
            for duplicate in tuyenlam_duplicates:
                if duplicate.id != best_tuyenlam.id:
                    merge_attractions(best_tuyenlam.id, duplicate.id)
        finally:
            db.close()
    
    # Handle Thác Cam Ly duplicates
    if len(camly_duplicates) > 1:
        logger.info(f"\n{'='*60}")
        logger.info("MERGING THÁC CAM LY")
        logger.info(f"{'='*60}")
        
        db = SessionLocal()
        try:
            # Find which one has more comments to keep
            best_camly = max(camly_duplicates, key=lambda x: db.query(func.count(Comment.id)).filter(
                Comment.attraction_id == x.id,
                Comment.is_meaningful == True
            ).scalar())
            
            for duplicate in camly_duplicates:
                if duplicate.id != best_camly.id:
                    merge_attractions(best_camly.id, duplicate.id)
        finally:
            db.close()
    
    # Handle Cầu Rồng duplicates
    if len(caulong_duplicates) > 1:
        logger.info(f"\n{'='*60}")
        logger.info("MERGING CẦU RỒNG")
        logger.info(f"{'='*60}")
        
        db = SessionLocal()
        try:
            # Find which one has more comments to keep
            best_caulong = max(caulong_duplicates, key=lambda x: db.query(func.count(Comment.id)).filter(
                Comment.attraction_id == x.id,
                Comment.is_meaningful == True
            ).scalar())
            
            for duplicate in caulong_duplicates:
                if duplicate.id != best_caulong.id:
                    merge_attractions(best_caulong.id, duplicate.id)
        finally:
            db.close()
    
    # Handle Vườn Hoa Đà Lạt duplicates
    if len(vuonhoa_duplicates) > 1:
        logger.info(f"\n{'='*60}")
        logger.info("MERGING VƯỜN HOA ĐÀ LẠT")
        logger.info(f"{'='*60}")
        
        db = SessionLocal()
        try:
            best_vuonhoa = max(vuonhoa_duplicates, key=lambda x: db.query(func.count(Comment.id)).filter(
                Comment.attraction_id == x.id,
                Comment.is_meaningful == True
            ).scalar())
            
            for duplicate in vuonhoa_duplicates:
                if duplicate.id != best_vuonhoa.id:
                    merge_attractions(best_vuonhoa.id, duplicate.id)
        finally:
            db.close()
    
    # Handle Làng Cù Lần duplicates
    if len(culan_duplicates) > 1:
        logger.info(f"\n{'='*60}")
        logger.info("MERGING LÀNG CÙ LẦN")
        logger.info(f"{'='*60}")
        
        db = SessionLocal()
        try:
            best_culan = max(culan_duplicates, key=lambda x: db.query(func.count(Comment.id)).filter(
                Comment.attraction_id == x.id,
                Comment.is_meaningful == True
            ).scalar())
            
            for duplicate in culan_duplicates:
                if duplicate.id != best_culan.id:
                    merge_attractions(best_culan.id, duplicate.id)
        finally:
            db.close()
    
    # Handle Thung Lũng Tình Yêu duplicates
    if len(thunglungtinhyeu_duplicates) > 1:
        logger.info(f"\n{'='*60}")
        logger.info("MERGING THUNG LŨNG TÌNH YÊU")
        logger.info(f"{'='*60}")
        
        db = SessionLocal()
        try:
            best_thunglungtinhyeu = max(thunglungtinhyeu_duplicates, key=lambda x: db.query(func.count(Comment.id)).filter(
                Comment.attraction_id == x.id,
                Comment.is_meaningful == True
            ).scalar())
            
            for duplicate in thunglungtinhyeu_duplicates:
                if duplicate.id != best_thunglungtinhyeu.id:
                    merge_attractions(best_thunglungtinhyeu.id, duplicate.id)
        finally:
            db.close()
    
    logger.info(f"\n{'='*60}")
    logger.info("✓ DUPLICATE CLEANUP COMPLETE")
    logger.info(f"{'='*60}")


if __name__ == "__main__":
    main()
