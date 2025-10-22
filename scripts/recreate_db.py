"""
Script to recreate database tables
"""
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from app.database.connection import engine
from app.models.base import Base

sys.stdout.reconfigure(encoding='utf-8')

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
