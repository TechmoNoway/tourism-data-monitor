"""
Script to keep only top N attractions per province
"""
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from app.database.connection import SessionLocal
from app.models.tourist_attraction import TouristAttraction
from app.models.province import Province

# Configuration
ATTRACTIONS_PER_PROVINCE = 20

db = SessionLocal()

try:
    provinces = db.query(Province).all()
    total_kept = 0
    total_deleted = 0
    
    print(f'\nKeeping top {ATTRACTIONS_PER_PROVINCE} attractions per province...\n')
    print('='*70)
    
    for province in provinces:
        # Get all attractions for this province ordered by rating
        attractions = db.query(TouristAttraction).filter(
            TouristAttraction.province_id == province.id
        ).order_by(TouristAttraction.rating.desc()).all()
        
        if not attractions:
            print(f'{province.name}: No attractions found')
            continue
        
        # Keep top N, delete the rest
        to_keep = attractions[:ATTRACTIONS_PER_PROVINCE]
        to_delete = attractions[ATTRACTIONS_PER_PROVINCE:]
        
        print(f'\n{province.name} ({province.code}):')
        print(f'  Total: {len(attractions)} attractions')
        print(f'  Keeping: {len(to_keep)} attractions')
        if to_keep:
            print(f'    Top rating: {to_keep[0].rating:.1f} - {to_keep[0].name[:40]}')
            print(f'    Lowest kept: {to_keep[-1].rating:.1f} - {to_keep[-1].name[:40]}')
        
        if to_delete:
            print(f'  Deleting: {len(to_delete)} attractions')
            print(f'    Highest deleted: {to_delete[0].rating:.1f} - {to_delete[0].name[:40]}')
            
            # Delete attractions outside top N
            for attr in to_delete:
                db.delete(attr)
        
        total_kept += len(to_keep)
        total_deleted += len(to_delete)
    
    # Commit changes
    db.commit()
    
    print('\n' + '='*70)
    print('SUMMARY:')
    print('='*70)
    print(f'Provinces processed: {len(provinces)}')
    print(f'Total kept: {total_kept} attractions')
    print(f'Total deleted: {total_deleted} attractions')
    print('='*70)
    print('\n[OK] Database updated successfully!')
    
except Exception as e:
    print(f'\n[ERROR] Error: {e}')
    db.rollback()
    import traceback
    traceback.print_exc()
finally:
    db.close()
