"""
Script to recreate database tables
"""
import sys
sys.stdout.reconfigure(encoding='utf-8')

from app.database.connection import engine
from app.models.base import Base
# Import all models to register them
from app.models.province import Province
from app.models.tourist_attraction import TouristAttraction
from app.models.social_post import SocialPost
from app.models.comment import Comment
from app.models.analysis_log import AnalysisLog

def recreate_tables():
    """Drop all tables and recreate them"""
    from sqlalchemy import text
    
    print("Dropping materialized views...")
    with engine.connect() as conn:
        try:
            conn.execute(text("DROP MATERIALIZED VIEW IF EXISTS province_stats CASCADE"))
            conn.commit()
            print("✓ Materialized views dropped")
        except Exception as e:
            print(f"Warning: {e}")
    
    print("\nDropping all tables...")
    Base.metadata.drop_all(bind=engine)
    print("✓ Tables dropped")
    
    print("\nCreating all tables...")
    Base.metadata.create_all(bind=engine)
    print("✓ Tables created")
    
    print("\nDatabase tables recreated successfully!")

if __name__ == "__main__":
    recreate_tables()
