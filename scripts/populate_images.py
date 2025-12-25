"""
Populate images for provinces and attractions from Google Maps data
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.database.connection import SessionLocal
from app.models.province import Province
from app.models.tourist_attraction import TouristAttraction
from app.models.social_post import SocialPost
from app.core.config import settings
from app.collectors.google_maps_collector import GoogleMapsApifyCollector
import asyncio
import logging
import json

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Default province images from Unsplash (fallback)
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


def get_image_from_post(post: SocialPost) -> str:
    """Extract image URL from social post"""
    if not post.content:
        return None
    
    try:
        # Parse JSON content that Google Maps collector stores
        data = json.loads(post.content) if isinstance(post.content, str) and post.content.startswith('{') else {}
        
        # Check for imageUrl in the data
        if 'imageUrl' in data and data['imageUrl']:
            return data['imageUrl']
        
        # Check for media_urls array
        if 'media_urls' in data and data['media_urls']:
            return data['media_urls'][0] if isinstance(data['media_urls'], list) else data['media_urls']
            
    except:
        pass
    
    return None


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
                image_url = media_urls[0]
                logger.info(f"✓ {attraction.name}: Fetched image")
                return image_url
        
        logger.warning(f"⚠ {attraction.name}: No image found")
        return None
        
    except Exception as e:
        logger.error(f"❌ {attraction.name}: Error - {str(e)}")
        return None


def populate_attraction_images_from_posts():
    """Populate attraction images from existing Google Maps posts"""
    db = SessionLocal()
    
    try:
        attractions = db.query(TouristAttraction).all()
        updated_count = 0
        
        for attraction in attractions:
            if attraction.image_url:
                continue  # Skip if already has image
            
            # Get the post for this attraction
            post = db.query(SocialPost).filter(
                SocialPost.attraction_id == attraction.id,
                SocialPost.platform == 'google_maps'
            ).first()
            
            if post:
                image_url = get_image_from_post(post)
                if image_url:
                    attraction.image_url = image_url
                    updated_count += 1
                    logger.info(f"✓ {attraction.name}: {image_url[:60]}...")
        
        db.commit()
        logger.info(f"\n✓ Updated {updated_count} attraction images from existing posts")
        return updated_count
        
    finally:
        db.close()


async def fetch_missing_attraction_images(collector: GoogleMapsApifyCollector):
    """Fetch images for attractions without images"""
    db = SessionLocal()
    
    try:
        attractions = db.query(TouristAttraction).filter(
            TouristAttraction.image_url == None
        ).all()
        
        if not attractions:
            logger.info("All attractions have images!")
            return 0
        
        logger.info(f"Fetching images for {len(attractions)} attractions...")
        updated_count = 0
        
        for i, attraction in enumerate(attractions, 1):
            logger.info(f"[{i}/{len(attractions)}] {attraction.name}")
            
            image_url = await fetch_image_for_attraction(collector, attraction)
            
            if image_url:
                attraction.image_url = image_url
                updated_count += 1
            
            # Small delay to avoid rate limits
            if i < len(attractions):
                await asyncio.sleep(2)
        
        db.commit()
        logger.info(f"\n✓ Fetched {updated_count} new attraction images")
        return updated_count
        
    finally:
        db.close()


def populate_province_images():
    """Populate province images from their attractions or defaults"""
    db = SessionLocal()
    
    try:
        provinces = db.query(Province).all()
        updated_count = 0
        
        for province in provinces:
            if province.image_url:
                continue  # Skip if already has image
            
            # Try default image first
            if province.name in DEFAULT_PROVINCE_IMAGES:
                province.image_url = DEFAULT_PROVINCE_IMAGES[province.name]
                updated_count += 1
                logger.info(f"✓ {province.name}: Using default image")
                continue
            
            # Get image from most popular attraction
            attraction = db.query(TouristAttraction).filter(
                TouristAttraction.province_id == province.id,
                TouristAttraction.image_url != None
            ).order_by(TouristAttraction.total_comments.desc()).first()
            
            if attraction and attraction.image_url:
                province.image_url = attraction.image_url
                updated_count += 1
                logger.info(f"✓ {province.name}: Using image from {attraction.name}")
            else:
                # Use default Vietnam landscape
                province.image_url = 'https://images.unsplash.com/photo-1528127269322-539801943592?w=1200&q=80'
                updated_count += 1
                logger.info(f"✓ {province.name}: Using default Vietnam image")
        
        db.commit()
        logger.info(f"\n✓ Updated {updated_count} province images")
        return updated_count
        
    finally:
        db.close()


async def main():
    logger.info("="*80)
    logger.info("POPULATING IMAGES FOR PROVINCES AND ATTRACTIONS")
    logger.info("="*80)
    
    # Step 1: Use existing post data
    logger.info("\n📸 Step 1: Populating from existing Google Maps posts...")
    from_posts = populate_attraction_images_from_posts()
    
    # Step 2: Fetch missing images from Google Maps
    logger.info("\n📸 Step 2: Fetching missing attraction images from Google Maps...")
    
    if not settings.APIFY_API_TOKEN:
        logger.warning("⚠ No Apify token - skipping live fetching")
        logger.info("Using fallback images for remaining attractions...")
        
        # Use fallback logic for remaining attractions
        db = SessionLocal()
        try:
            from scripts.populate_images import TOURISM_TYPE_IMAGES
            attractions = db.query(TouristAttraction).filter(
                TouristAttraction.image_url == None
            ).all()
            
            for attraction in attractions:
                tourism_type = attraction.tourism_type or 'nature'
                attraction.image_url = TOURISM_TYPE_IMAGES.get(
                    tourism_type,
                    'https://images.unsplash.com/photo-1506905925346-21bda4d32df4?w=1200&q=80'
                )
            
            db.commit()
            logger.info(f"✓ Set fallback images for {len(attractions)} attractions")
        finally:
            db.close()
    else:
        collector = GoogleMapsApifyCollector(settings.APIFY_API_TOKEN, skip_sentiment=True)
        
        if collector.authenticate():
            from_gmaps = await fetch_missing_attraction_images(collector)
            logger.info(f"✓ Total fetched: {from_gmaps}")
        else:
            logger.error("Failed to authenticate with Apify")
    
    # Step 3: Populate province images
    logger.info("\n📸 Step 3: Populating province images...")
    province_count = populate_province_images()
    
    logger.info("\n" + "="*80)
    logger.info("✓ IMAGE POPULATION COMPLETE")
    logger.info("="*80)
    logger.info(f"Attractions updated: {from_posts} from posts + fetched from Google Maps")
    logger.info(f"Provinces updated: {province_count}")


# Fallback tourism type images
TOURISM_TYPE_IMAGES = {
    'beach': 'https://images.unsplash.com/photo-1519046904884-53103b34b206?w=1200&q=80',
    'mountain': 'https://images.unsplash.com/photo-1506905925346-21bda4d32df4?w=1200&q=80',
    'historical': 'https://images.unsplash.com/photo-1555217851-064893f593e2?w=1200&q=80',
    'cultural': 'https://images.unsplash.com/photo-1548013146-72479768bada?w=1200&q=80',
    'nature': 'https://images.unsplash.com/photo-1441974231531-c6227db76b6e?w=1200&q=80',
    'urban': 'https://images.unsplash.com/photo-1480714378408-67cf0d13bc1b?w=1200&q=80',
    'adventure': 'https://images.unsplash.com/photo-1519904981063-b0cf448d479e?w=1200&q=80',
}


if __name__ == "__main__":
    asyncio.run(main())
