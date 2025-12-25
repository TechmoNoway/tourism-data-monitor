"""
Collect images for all provinces from Google Maps
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.database.connection import SessionLocal
from app.models.province import Province
from app.collectors.google_maps_collector import GoogleMapsApifyCollector
from app.core.config import Settings
import asyncio
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

settings = Settings()


async def populate_province_images():
    """Fetch and populate images for all provinces"""
    db = SessionLocal()
    
    try:
        provinces = db.query(Province).all()
        logger.info(f"Found {len(provinces)} provinces")
        
        # Check which provinces need images
        provinces_without_images = [p for p in provinces if not p.image_url]
        logger.info(f"{len(provinces_without_images)} provinces need images")
        
        if not provinces_without_images:
            logger.info("All provinces already have images!")
            return
        
        collector = GoogleMapsApifyCollector(settings.APIFY_API_TOKEN)
        
        for province in provinces_without_images:
            logger.info(f"\nFetching image for: {province.name}")
            
            # Search for the province name + "Vietnam"
            search_query = f"{province.name} Vietnam"
            
            try:
                result = await collector.collect_posts([search_query])
                
                if result and len(result) > 0:
                    place = result[0]
                    images = place.get('imageUrls', [])
                    
                    if images:
                        # Use the first image
                        province.image_url = images[0]
                        db.commit()
                        logger.info(f"✓ Updated image for {province.name}")
                    else:
                        logger.warning(f"No images found for {province.name}")
                else:
                    logger.warning(f"No results for {province.name}")
                    
                # Small delay to avoid rate limiting
                await asyncio.sleep(2)
                
            except Exception as e:
                logger.error(f"Error fetching image for {province.name}: {e}")
                continue
        
        logger.info("\n✓ Province image collection complete")
        
    except Exception as e:
        logger.error(f"Error: {e}")
        import traceback
        logger.error(traceback.format_exc())
        
    finally:
        db.close()


if __name__ == "__main__":
    asyncio.run(populate_province_images())
