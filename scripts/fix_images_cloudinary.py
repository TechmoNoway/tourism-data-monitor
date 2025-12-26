"""
Re-fetch all images and upload to Cloudinary
Fixes broken image URLs by downloading and storing on Cloudinary
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
import requests
import cloudinary
import cloudinary.uploader
import cloudinary.api
from io import BytesIO
from typing import Optional

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configure Cloudinary
cloudinary.config(
    cloud_name=settings.CLOUDINARY_CLOUD_NAME,
    api_key=settings.CLOUDINARY_API_KEY,
    api_secret=settings.CLOUDINARY_API_SECRET
)


def upload_image_to_cloudinary(image_url: str, public_id: str, folder: str = "tourism") -> Optional[str]:
    """
    Download image from URL and upload to Cloudinary
    Returns Cloudinary URL or None if failed
    """
    try:
        # Download image
        logger.info(f"   Downloading image from: {image_url[:80]}...")
        response = requests.get(image_url, timeout=15, headers={
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        })
        response.raise_for_status()
        
        # Check if it's actually an image
        content_type = response.headers.get('content-type', '')
        if not content_type.startswith('image/'):
            logger.warning(f"   Not an image: {content_type}")
            return None
        
        # Upload to Cloudinary
        logger.info(f"   Uploading to Cloudinary...")
        result = cloudinary.uploader.upload(
            BytesIO(response.content),
            public_id=public_id,
            folder=folder,
            overwrite=True,
            resource_type="image",
            quality="auto:good",
            fetch_format="auto"
        )
        
        cloudinary_url = result.get('secure_url')
        logger.info(f"   ✅ Uploaded: {cloudinary_url}")
        return cloudinary_url
        
    except requests.exceptions.RequestException as e:
        logger.error(f"   ❌ Failed to download image: {e}")
        return None
    except Exception as e:
        logger.error(f"   ❌ Failed to upload to Cloudinary: {e}")
        return None


async def fetch_new_image_from_google_maps(collector: GoogleMapsApifyCollector, attraction: TouristAttraction) -> Optional[str]:
    """Fetch fresh image URL from Google Maps"""
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
                # Get first valid image URL
                valid_urls = [url for url in media_urls if url and url.startswith('http')]
                if valid_urls:
                    return valid_urls[0]
        
        return None
        
    except Exception as e:
        logger.error(f"   Error fetching from Google Maps: {str(e)}")
        return None


async def fix_attraction_images(force_all: bool = False, test_mode: bool = False):
    """
    Re-fetch and upload attraction images to Cloudinary
    
    Args:
        force_all: If True, re-fetch ALL images. If False, only fix broken ones.
        test_mode: If True, only process first 5 attractions
    """
    
    logger.info("\n" + "="*80)
    logger.info("🔧 FIXING ATTRACTION IMAGES - UPLOAD TO CLOUDINARY")
    logger.info("="*80 + "\n")
    
    # Check Cloudinary config
    if not all([settings.CLOUDINARY_CLOUD_NAME, settings.CLOUDINARY_API_KEY, settings.CLOUDINARY_API_SECRET]):
        logger.error("❌ Cloudinary credentials not configured in .env!")
        return
    
    logger.info(f"☁️  Cloudinary: {settings.CLOUDINARY_CLOUD_NAME}")
    
    # Initialize Google Maps collector for fetching new images
    if not settings.APIFY_API_TOKEN:
        logger.error("❌ APIFY_API_TOKEN not configured!")
        return
    
    collector = GoogleMapsApifyCollector(settings.APIFY_API_TOKEN, skip_sentiment=True)
    if not collector.authenticate():
        logger.error("❌ Failed to authenticate with Apify")
        return
    
    db = SessionLocal()
    
    try:
        # Get attractions to process
        if force_all:
            attractions = db.query(TouristAttraction).filter(
                TouristAttraction.is_active.is_(True)
            ).all()
            logger.info(f"🔄 Force mode: Processing ALL {len(attractions)} active attractions\n")
        else:
            # Only process attractions with broken or no images
            attractions = db.query(TouristAttraction).filter(
                TouristAttraction.is_active.is_(True)
            ).filter(
                (TouristAttraction.image_url.is_(None)) |
                (~TouristAttraction.image_url.contains('cloudinary'))
            ).all()
            logger.info(f"🔍 Processing {len(attractions)} attractions with non-Cloudinary images\n")
        
        if not attractions:
            logger.info("✅ All attractions already have Cloudinary images!")
            return
        
        if test_mode:
            attractions = attractions[:5]
            logger.info(f"🧪 TEST MODE: Processing only first {len(attractions)} attractions\n")
        
        success_count = 0
        failed_count = 0
        skipped_count = 0
        
        for i, attraction in enumerate(attractions, 1):
            logger.info(f"\n[{i}/{len(attractions)}] {attraction.name} (ID: {attraction.id})")
            logger.info(f"   Province: {attraction.province.name if attraction.province else 'Unknown'}")
            
            current_url = attraction.image_url
            if current_url:
                logger.info(f"   Current: {current_url[:80]}...")
            else:
                logger.info(f"   Current: No image")
            
            # Generate Cloudinary public ID
            safe_name = attraction.name.lower().replace(' ', '_').replace('(', '').replace(')', '')
            safe_name = ''.join(c for c in safe_name if c.isalnum() or c in ['_', '-'])[:50]
            public_id = f"attraction_{attraction.id}_{safe_name}"
            
            cloudinary_url = None
            
            # Try to re-upload existing image if it exists
            if current_url and current_url.startswith('http'):
                cloudinary_url = upload_image_to_cloudinary(
                    current_url,
                    public_id,
                    folder="tourism/attractions"
                )
            
            # If failed or no image, fetch new one from Google Maps
            if not cloudinary_url:
                logger.info(f"   Fetching new image from Google Maps...")
                new_image_url = await fetch_new_image_from_google_maps(collector, attraction)
                
                if new_image_url:
                    cloudinary_url = upload_image_to_cloudinary(
                        new_image_url,
                        public_id,
                        folder="tourism/attractions"
                    )
            
            # Update database
            if cloudinary_url:
                attraction.image_url = cloudinary_url
                db.commit()
                success_count += 1
                logger.info(f"   ✅ SUCCESS - Image saved to Cloudinary")
            else:
                failed_count += 1
                logger.warning(f"   ❌ FAILED - Could not fetch/upload image")
            
            # Rate limiting
            if i < len(attractions):
                logger.info(f"   Waiting 3 seconds...")
                await asyncio.sleep(3)
        
        # Summary
        logger.info("\n" + "="*80)
        logger.info("📊 SUMMARY")
        logger.info("="*80)
        logger.info(f"Total processed: {len(attractions)}")
        logger.info(f"✅ Success: {success_count}")
        logger.info(f"❌ Failed: {failed_count}")
        logger.info(f"📈 Success rate: {(success_count/len(attractions)*100):.1f}%")
        logger.info("="*80 + "\n")
        
    finally:
        db.close()


async def fix_province_images(force_all: bool = False):
    """
    Re-fetch and upload province images to Cloudinary
    """
    
    logger.info("\n" + "="*80)
    logger.info("🗺️  FIXING PROVINCE IMAGES - UPLOAD TO CLOUDINARY")
    logger.info("="*80 + "\n")
    
    db = SessionLocal()
    
    try:
        if force_all:
            provinces = db.query(Province).all()
        else:
            provinces = db.query(Province).filter(
                (Province.image_url.is_(None)) |
                (~Province.image_url.contains('cloudinary'))
            ).all()
        
        logger.info(f"Processing {len(provinces)} provinces\n")
        
        success_count = 0
        failed_count = 0
        
        for i, province in enumerate(provinces, 1):
            logger.info(f"[{i}/{len(provinces)}] {province.name}")
            
            if province.image_url:
                logger.info(f"   Current: {province.image_url[:80]}...")
                
                safe_name = province.name.lower().replace(' ', '_')
                public_id = f"province_{province.id}_{safe_name}"
                
                cloudinary_url = upload_image_to_cloudinary(
                    province.image_url,
                    public_id,
                    folder="tourism/provinces"
                )
                
                if cloudinary_url:
                    province.image_url = cloudinary_url
                    db.commit()
                    success_count += 1
                    logger.info(f"   ✅ SUCCESS")
                else:
                    failed_count += 1
                    logger.warning(f"   ❌ FAILED")
            else:
                logger.info(f"   No image URL")
                failed_count += 1
            
            await asyncio.sleep(2)
        
        logger.info(f"\n✅ Provinces: {success_count} success, {failed_count} failed\n")
        
    finally:
        db.close()


async def main():
    """Main entry point"""
    
    import argparse
    
    parser = argparse.ArgumentParser(description='Re-fetch images and upload to Cloudinary')
    parser.add_argument('--force', action='store_true', help='Re-upload ALL images (not just broken ones)')
    parser.add_argument('--test', action='store_true', help='Test mode - only process first 5 attractions')
    parser.add_argument('--provinces', action='store_true', help='Also fix province images')
    parser.add_argument('--provinces-only', action='store_true', help='Only fix province images')
    
    args = parser.parse_args()
    
    try:
        if args.provinces_only:
            # Only provinces
            await fix_province_images(force_all=args.force)
        else:
            # Attractions
            await fix_attraction_images(force_all=args.force, test_mode=args.test)
            
            # Provinces if requested
            if args.provinces:
                await fix_province_images(force_all=args.force)
        
        logger.info("🎉 Image fix completed!")
        
    except KeyboardInterrupt:
        logger.info("\n\n⚠️  Interrupted by user")
    except Exception as e:
        logger.error(f"\n\n❌ Error: {e}", exc_info=True)


if __name__ == "__main__":
    asyncio.run(main())
