import asyncio
import sys
from pathlib import Path
from typing import List

sys.path.insert(0, str(Path(__file__).parent.parent))

from app.database.connection import get_db
from app.models.province import Province
from app.models.tourist_attraction import TouristAttraction
from app.models.social_post import SocialPost
from app.models.comment import Comment
from app.collectors.data_pipeline import create_data_pipeline
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

# Scale 3x: Previous 24 posts -> 72 posts per attraction
# Previous 120 comments -> 360 comments per attraction
TARGET_POSTS_PER_ATTRACTION = 72  
TARGET_COMMENTS_PER_ATTRACTION = 360 
PLATFORMS_PRIORITY = ['facebook', 'google_maps', 'youtube', 'tiktok']


async def collect_for_attraction_multi_platform(
    pipeline,
    attraction_id: int,
    attraction_name: str,
    province_name: str
) -> dict:
    db = next(get_db())
    
    existing_posts = db.query(SocialPost).filter(
        SocialPost.attraction_id == attraction_id
    ).count()
    
    existing_comments = db.query(Comment).filter(
        Comment.attraction_id == attraction_id
    ).count()
    
    print("\n" + "="*70)
    print(f"{attraction_name} ({province_name})")
    print(f"   Existing: {existing_posts} posts, {existing_comments} comments")
    print(f"   Target: {TARGET_COMMENTS_PER_ATTRACTION} comments total")
    print("="*70)
    print()
    
    if existing_comments >= TARGET_COMMENTS_PER_ATTRACTION:
        print(f"\nTarget reached! Total comments: {existing_comments}/{TARGET_COMMENTS_PER_ATTRACTION}\n")
        return {
            'attraction_id': attraction_id,
            'attraction_name': attraction_name,
            'province': province_name,
            'posts_before': existing_posts,
            'comments_before': existing_comments,
            'posts_collected': 0,
            'comments_collected': 0,
            'posts_total': existing_posts,
            'comments_total': existing_comments,
            'platforms_used': []
        }
    
    results = {
        'attraction_id': attraction_id,
        'attraction_name': attraction_name,
        'province': province_name,
        'posts_before': existing_posts,
        'comments_before': existing_comments,
        'posts_collected': 0,
        'comments_collected': 0,
        'platforms_used': []
    }
    
    for platform in PLATFORMS_PRIORITY:
        current_posts = db.query(SocialPost).filter(
            SocialPost.attraction_id == attraction_id
        ).count()
        
        current_comments = db.query(Comment).filter(
            Comment.attraction_id == attraction_id
        ).count()
        
        comments_needed = TARGET_COMMENTS_PER_ATTRACTION - current_comments
        
        if comments_needed <= 0:
            print(f"Got {current_comments} comments, stopping here!")
            break
        
        if current_comments >= TARGET_COMMENTS_PER_ATTRACTION * 0.8:
            print(f"Got {current_comments} comments (80% of target), stopping here!")
            break
        
        print(f"\nTrying {platform.upper()}: Need {comments_needed} more comments...")
        
        try:
            # Scale 3x: Previous limit 50 -> 100 posts per platform
            # To reach 72 posts/attraction with multiple platforms
            await pipeline.collect_for_attraction(
                attraction_id=attraction_id,
                platforms=[platform], 
                limit_per_platform=100  
            )
            
            new_posts = db.query(SocialPost).filter(
                SocialPost.attraction_id == attraction_id
            ).count() - current_posts
            
            new_comments = db.query(Comment).filter(
                Comment.attraction_id == attraction_id
            ).count() - current_comments
            
            if new_posts > 0 or new_comments > 0:
                print(f"{platform.upper()}: +{new_posts} posts, +{new_comments} comments")
                results['posts_collected'] += new_posts
                results['comments_collected'] += new_comments
                results['platforms_used'].append(platform)
            else:
                print(f"{platform.upper()}: No new data")
                
        except Exception as e:
            print(f"{platform.upper()} failed: {e}")
    
    results['posts_total'] = db.query(SocialPost).filter(
        SocialPost.attraction_id == attraction_id
    ).count()
    
    results['comments_total'] = db.query(Comment).filter(
        Comment.attraction_id == attraction_id
    ).count()
    
    print("\n" + "="*70)
    print(f"COMPLETED: {attraction_name}")
    print(f"   Final: {results['posts_total']} posts, {results['comments_total']} comments")
    print(f"   New: +{results['posts_collected']} posts, +{results['comments_collected']} comments")
    print("="*70)
    
    db.close()
    return results


async def run_collection(province_names: List[str] = None, attractions_per_province: int = 3, all_attractions: bool = False):
    db = next(get_db())
    
    from app.core.config import settings
    
    pipeline = create_data_pipeline(
        apify_api_token=settings.APIFY_API_TOKEN,
        youtube_api_key=settings.YOUTUBE_API_KEY
    )
    
    print("\n" + "="*70)
    print("MULTI-PLATFORM TOURISM DATA COLLECTION")
    print("="*70)
    if province_names:
        print(f"Target provinces: {', '.join(province_names)}")
    else:
        print("Target: All provinces")
    print(f"Platform priority: {' -> '.join([p.upper() for p in PLATFORMS_PRIORITY])}")
    print(f"Target: {TARGET_POSTS_PER_ATTRACTION} posts, {TARGET_COMMENTS_PER_ATTRACTION} comments per attraction")
    print("="*70)
    print()
    
    # Get provinces
    query = db.query(Province)
    if province_names:
        query = query.filter(Province.name.in_(province_names))
    provinces = query.all()
    
    if not provinces:
        print("No provinces found!")
        db.close()
        return
    
    results = []
    
    for province in provinces:
        print(f"\n{province.name}: Processing attractions...")
        print()
        
        attraction_query = db.query(TouristAttraction).filter(
            TouristAttraction.province_id == province.id,
            TouristAttraction.is_active.is_(True)
        )
        
        if not all_attractions:
            attraction_query = attraction_query.limit(attractions_per_province)
        
        attractions = attraction_query.all()
        
        print(f"   Found {len(attractions)} attraction(s) to process")
        
        for attraction in attractions:
            result = await collect_for_attraction_multi_platform(
                pipeline,
                attraction.id,
                attraction.name,
                province.name
            )
            results.append(result)
            
            try:
                print("Waiting 5 seconds before next attraction...")
                await asyncio.sleep(5)
            except asyncio.CancelledError:
                print("Sleep cancelled, continuing...")
                pass
    
    db.close()
    
    print("\n" + "="*70)
    print("COLLECTION COMPLETE - FINAL SUMMARY")
    print("="*70)
    print()
    
    for result in results:
        print(f"{result['attraction_name']}")
        print(f"   Posts: {result['posts_total']} ({result['posts_collected']} new)")
        print(f"   Comments: {result['comments_total']} ({result['comments_collected']} new)")
        if result['platforms_used']:
            print(f"   Platforms: {', '.join(result['platforms_used'])}")
        print()
    
    total_posts_collected = sum(r['posts_collected'] for r in results)
    total_comments_collected = sum(r['comments_collected'] for r in results)
    total_attractions = len(results)
    avg_comments = total_comments_collected / total_attractions if total_attractions > 0 else 0
    
    print("TOTALS:")
    print(f"Attractions processed: {total_attractions}")
    print(f"New posts collected: {total_posts_collected}")
    print(f"New comments collected: {total_comments_collected}")
    print(f"Average comments/attraction: {avg_comments:.1f}")
    print()
    print()


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Collect tourism data from social platforms')
    parser.add_argument(
        '--provinces',
        type=str,
        help='Comma-separated list of province names (e.g., "Bình Thuận,Đà Nẵng")'
    )
    parser.add_argument(
        '--limit',
        type=int,
        default=3,
        help='Number of attractions to process per province (default: 3)'
    )
    parser.add_argument(
        '--all',
        action='store_true',
        help='Process all active attractions (ignore limit)'
    )
    
    args = parser.parse_args()
    
    province_list = None
    if args.provinces:
        province_list = [p.strip() for p in args.provinces.split(',')]
    
    asyncio.run(run_collection(
        province_names=province_list,
        attractions_per_province=args.limit,
        all_attractions=args.all
    ))
