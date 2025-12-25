"""
Fetch real images from Google Maps for all attractions and provinces
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.database.connection import SessionLocal
from app.models.province import Province
from app.models.tourist_attraction import TouristAttraction
from app.core.config import settings
from app.collectors.google_maps_collector import GoogleMapsApifyCollector
import asyncio
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Default province images (high quality)
DEFAULT_PROVINCE_IMAGES = {
    'Đà Nẵng': 'https://images.unsplash.com/photo-1583417319070-4a69db38a482?w=1200&q=80',
    'Hồ Chí Minh': 'https://images.unsplash.com/photo-1583417269936-3e97d8eb5597?w=1200&q=80',
    'Hà Nội': 'https://images.unsplash.com/photo-1509931373843-86df4097e6f3?w=1200&q=80',
    'Quảng Ninh': 'https://images.unsplash.com/photo-1528127269322-539801943592?w=1200&q=80',
    'Khánh Hòa': 'https://images.unsplash.com/photo-1559592413-7cec4d0cae2b?w=1200&q=80',
    'Lâm Đồng': 'https://images.unsplash.com/photo-1545308269-1f8e9a2b8c0a?w=1200&q=80',
    'Thừa Thiên Huế': 'https://images.unsplash.com/photo-1557409518-691ebcd96038?w=1200&q=80',
    'Bình Thuận': 'https://images.unsplash.com/photo-1559592413-7cec4d0cae2b?w=1200&q=80',
}


async def fetch_image_for_attraction(collector: GoogleMapsApifyCollector, attraction: TouristAttraction) -> str:
    """Fetch image from Google Maps for an attraction"""
    try:
        search_query = f"{attraction.name} vietnam"
        
        places = await collector.collect_posts(
            keywords=[search_query],
            location="Vietnam",
            limit=1
        )
        
        if places and len(places) > 0:
            place = places[0]
            media_urls = place.get('media_urls', [])
            
            if media_urls and len(media_urls) > 0:
                # Filter out invalid URLs
                valid_urls = [url for url in media_urls if url and url.startswith('http')]
                if valid_urls:
                    return valid_urls[0]
        
        return None
        
    except Exception as e:
        logger.error(f"Error fetching image for {attraction.name}: {str(e)}")
        return None


async def fetch_all_attraction_images(force_refresh: bool = False):
    """Fetch images for all attractions from Google Maps"""
    
    if not settings.APIFY_API_TOKEN:
        logger.error("❌ APIFY_API_TOKEN not configured!")
        return
    
    collector = GoogleMapsApifyCollector(settings.APIFY_API_TOKEN, skip_sentiment=True)
    
    if not collector.authenticate():
        logger.error("❌ Failed to authenticate with Apify")
        return
    
    db = SessionLocal()
    
    try:
        # Get attractions
        if force_refresh:
            attractions = db.query(TouristAttraction).all()
            logger.info(f"Force refresh: Fetching images for ALL {len(attractions)} attractions")
        else:
            attractions = db.query(TouristAttraction).filter(
                TouristAttraction.image_url == None
            ).all()
            logger.info(f"Fetching images for {len(attractions)} attractions without images")
        
        if not attractions:
            logger.info("All attractions already have images!")
            return
        
        updated_count = 0
        failed_count = 0
        
        for i, attraction in enumerate(attractions, 1):
            logger.info(f"[{i}/{len(attractions)}] {attraction.name}")
            
            image_url = await fetch_image_for_attraction(collector, attraction)
            
            if image_url:
                attraction.image_url = image_url
                updated_count += 1
                logger.info(f"  ✓ {image_url[:80]}...")
            else:
                failed_count += 1
                logger.warning(f"  ⚠ No image found")
            
            # Commit every 10 attractions
            if i % 10 == 0:
                db.commit()
                logger.info(f"  💾 Saved progress ({updated_count} updated so far)")
            
            # Small delay to avoid rate limits
            if i < len(attractions):
                await asyncio.sleep(2)
        
        db.commit()
        logger.info(f"\n✓ Updated {updated_count} attraction images")
        logger.info(f"⚠ Failed to fetch {failed_count} images")
        
    finally:
        db.close()


def populate_province_images():
    """Populate province images from defaults or popular attractions"""
    db = SessionLocal()
    
    try:
        provinces = db.query(Province).all()
        updated_count = 0
        
        for province in provinces:
            # Use default image if available
            if province.name in DEFAULT_PROVINCE_IMAGES:
                if province.image_url != DEFAULT_PROVINCE_IMAGES[province.name]:
                    province.image_url = DEFAULT_PROVINCE_IMAGES[province.name]
                    updated_count += 1
                    logger.info(f"✓ {province.name}: Using curated image")
            else:
                # Get image from most popular attraction
                attraction = db.query(TouristAttraction).filter(
                    TouristAttraction.province_id == province.id,
                    TouristAttraction.image_url != None
                ).order_by(TouristAttraction.total_comments.desc()).first()
                
                if attraction and attraction.image_url:
                    if province.image_url != attraction.image_url:
                        province.image_url = attraction.image_url
                        updated_count += 1
                        logger.info(f"✓ {province.name}: Using image from {attraction.name}")
        
        db.commit()
        logger.info(f"\n✓ Updated {updated_count} province images")
        
    finally:
        db.close()


async def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Fetch images from Google Maps')
    parser.add_argument('--force', action='store_true', help='Force refresh all images')
    args = parser.parse_args()
    
    logger.info("="*80)
    logger.info("FETCHING REAL IMAGES FROM GOOGLE MAPS")
    logger.info("="*80)
    
    # Step 1: Fetch attraction images
    logger.info("\n📸 Step 1: Fetching attraction images from Google Maps...")
    await fetch_all_attraction_images(force_refresh=args.force)
    
    # Step 2: Populate province images
    logger.info("\n📸 Step 2: Updating province images...")
    populate_province_images()
    
    logger.info("\n" + "="*80)
    logger.info("✓ IMAGE COLLECTION COMPLETE")
    logger.info("="*80)


if __name__ == "__main__":
    asyncio.run(main())
