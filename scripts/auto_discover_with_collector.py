"""
Auto-Discovery Integration - Using GoogleMapsCollector
======================================================

Script này sử dụng method auto_discover_attractions() đã được tích hợp
vào GoogleMapsApifyCollector để tự động phát hiện attractions.

USAGE:
------
1. DRY RUN (Preview only):
   python scripts/auto_discover_with_collector.py --dry-run

2. Production (Save to database):
   python scripts/auto_discover_with_collector.py

3. Custom province:
   python scripts/auto_discover_with_collector.py --province "Quảng Nam" --city "Hội An" --code "QN"

4. Custom limit:
   python scripts/auto_discover_with_collector.py --limit 5
"""

import asyncio
import sys
from pathlib import Path
import argparse
import os

sys.path.insert(0, str(Path(__file__).parent.parent))

from dotenv import load_dotenv
from app.collectors.google_maps_collector import create_google_maps_collector
from app.database.connection import SessionLocal
from app.models.tourist_attraction import TouristAttraction
from app.models.province import Province

load_dotenv()

DEFAULT_ATTRACTION_TYPES = [
    "tourist attraction",
    "beach",
    "mountain",
    "waterfall",
    "temple",
    "pagoda",
    "museum",
    "amusement park",
    "national park",
    "historical site"
]


def save_attractions_to_database(
    attractions: list,
    province_code: str,
    province_name: str,
    main_city: str,
    dry_run: bool = True
):
    mode_str = "DRY RUN (Preview)" if dry_run else "PRODUCTION (Saving)"
    
    print(f"\n{'='*80}")
    print(f"💾 {mode_str}")
    print(f"{'='*80}\n")
    
    db = SessionLocal()
    
    try:
        # Get or create province
        province = db.query(Province).filter(Province.code == province_code).first()
        
        if not province:
            if dry_run:
                print(f"[INFO] Province '{province_name}' ({province_code}) not found")
                print("Will be created in production mode\n")
                province_id = None
            else:
                province = Province(
                    name=province_name,
                    code=province_code,
                    main_city=main_city
                )
                db.add(province)
                db.commit()
                province_id = province.id
                print(f"[SUCCESS] Created province: {province_name} ({province_code})\n")
        else:
            province_id = province.id
            print(f"[OK] Province exists: {province.name} (ID: {province_id})\n")
        
        # Process attractions
        added_count = 0
        skipped_count = 0
        
        print("-"*80)
        print("PROCESSING ATTRACTIONS:")
        print("-"*80 + "\n")
        
        for idx, attr_data in enumerate(attractions, 1):
            # Check duplicates
            existing = db.query(TouristAttraction).filter(
                TouristAttraction.google_place_id == attr_data['google_place_id']
            ).first()
            
            if existing:
                print(f"[{idx}/{len(attractions)}] ⏭️  SKIP: {attr_data['name'][:60]}")
                print(f"           Reason: Already exists (ID: {existing.id})\n")
                skipped_count += 1
                continue
            
            if dry_run:
                print(f"[{idx}/{len(attractions)}] ✨ WOULD ADD: {attr_data['name']}")
                print(f"           Category: {attr_data['category']}")
                print(f"           Address: {attr_data['address'][:60]}...")
                print(f"           Rating: {attr_data['rating']:.1f}/5.0")
                if attr_data.get('phone'):
                    print(f"           Phone: {attr_data['phone']}")
                if attr_data.get('website'):
                    print(f"           Website: {attr_data['website'][:50]}...")
                print()
            else:
                new_attraction = TouristAttraction(
                    name=attr_data['name'],
                    province_id=province_id,
                    address=attr_data['address'],
                    description=attr_data['description'],
                    google_place_id=attr_data['google_place_id'],
                    rating=attr_data['rating']
                )
                db.add(new_attraction)
                db.commit()
                
                print(f"[{idx}/{len(attractions)}] [SUCCESS] ADDED: {attr_data['name']}")
                print(f"           Database ID: {new_attraction.id}")
                print(f"           Category: {attr_data['category']}")
                print(f"           Rating: {attr_data['rating']:.1f}/5.0\n")
            
            added_count += 1
        
        # Final summary
        print("\n" + "="*80)
        print("📈 FINAL RESULTS")
        print("="*80)
        if dry_run:
            print(f"Would add: {added_count} new attractions")
        else:
            print(f"[SUCCESS] Added: {added_count} new attractions")
            print(f"Skipped: {skipped_count} existing attractions")
        print(f"[STATS] Total processed: {len(attractions)}")
        print("="*80 + "\n")
        
    except Exception as e:
        print(f"\n[FAILED] ERROR: {str(e)}")
        import traceback
        traceback.print_exc()
        db.rollback()
    finally:
        db.close()


async def main():
    parser = argparse.ArgumentParser(
        description='Auto-discover attractions using integrated collector method'
    )
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Preview mode (no database changes)'
    )
    parser.add_argument(
        '--all-provinces',
        action='store_true',
        help='Discover for all provinces in database (default behavior)'
    )
    parser.add_argument(
        '--province',
        type=str,
        default=None,
        help='Specific province name (e.g., "Lâm Đồng")'
    )
    parser.add_argument(
        '--city',
        type=str,
        default=None,
        help='Main city for specific province'
    )
    parser.add_argument(
        '--code',
        type=str,
        default=None,
        help='Province code for specific province'
    )
    parser.add_argument(
        '--limit',
        type=int,
        default=3,
        help='Attractions per type (default: 3)'
    )
    parser.add_argument(
        '--categories',
        type=str,
        nargs='+',
        help='Custom categories (space-separated)'
    )
    
    args = parser.parse_args()
    
    # Configuration
    attraction_types = args.categories if args.categories else DEFAULT_ATTRACTION_TYPES
    
    # Get API token
    apify_token = os.getenv('APIFY_API_TOKEN')
    if not apify_token:
        print("[FAILED] ERROR: APIFY_API_TOKEN not found in environment")
        print("Please add it to your .env file")
        return
    
    # Create collector
    print("🔐 Creating Google Maps collector...")
    try:
        collector = create_google_maps_collector(apify_token)
        print("[OK] Collector initialized\n")
    except Exception as e:
        print(f"[FAILED] Failed to create collector: {str(e)}")
        return
    
    # Determine provinces to process
    from app.database.connection import SessionLocal
    from app.models.province import Province
    
    db = SessionLocal()
    
    if args.province:
        # Single province mode
        provinces_to_process = [{
            'name': args.province,
            'city': args.city or 'Main City',
            'code': args.code or 'XX'
        }]
        print(f"[MODE] Mode: Single Province ({args.province})\n")
    else:
        # All provinces mode (default)
        all_provinces = db.query(Province).all()
        provinces_to_process = [{
            'name': p.name,
            'city': p.main_city,
            'code': p.code
        } for p in all_provinces]
        print(f"🌍 Mode: All Provinces ({len(provinces_to_process)} provinces)\n")
    
    db.close()
    
    if not provinces_to_process:
        print("[FAILED] No provinces to process")
        return
    
    # Process each province
    total_discovered = 0
    
    for prov_data in provinces_to_process:
        print("\n" + "="*80)
        print(f"🏙️  PROVINCE: {prov_data['name']} ({prov_data['code']})")
        print("="*80)
        print(f"Main City: {prov_data['city']}")
        print(f"Limit per type: {args.limit}")
        print(f"Categories: {len(attraction_types)} types")
        print(f"Mode: {'DRY RUN (Preview)' if args.dry_run else 'PRODUCTION (Will save to DB)'}")
        print("="*80 + "\n")
        
        # Auto-discover attractions
        print("[SEARCH] Starting auto-discovery...\n")
        
        attractions = await collector.auto_discover_attractions(
            province_name=prov_data['name'],
            main_city=prov_data['city'],
            attraction_types=attraction_types,
            limit_per_type=args.limit
        )
        
        if not attractions:
            print(f"\n[WARNING] No attractions discovered for {prov_data['name']}")
            continue
        
        # Show statistics
        print("\n" + "="*80)
        print(f"[STATS] DISCOVERY STATISTICS - {prov_data['name']}")
        print("="*80)
        
        # Group by category
        from collections import Counter
        category_counts = Counter(a['category'] for a in attractions)
        
        for category, count in sorted(category_counts.items(), key=lambda x: -x[1]):
            print(f"  • {category.title()}: {count} places")
        
        print(f"\nTotal discovered: {len(attractions)}")
        total_discovered += len(attractions)
        print("="*80)
        
        # Save to database
        save_attractions_to_database(
            attractions=attractions,
            province_name=prov_data['name'],
            province_code=prov_data['code'],
            main_city=prov_data['city'],
            dry_run=args.dry_run
        )
    
    # Final summary
    print("\n" + "="*80)
    print("🎉 AUTO-DISCOVERY COMPLETED")
    print("="*80)
    print(f"Provinces processed: {len(provinces_to_process)}")
    print(f"Total attractions discovered: {total_discovered}")
    if args.dry_run:
        print("\n[TIP] TIP: To save to database, run without --dry-run flag")
    print("="*80 + "\n")


if __name__ == '__main__':
    asyncio.run(main())
