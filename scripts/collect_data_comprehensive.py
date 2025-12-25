"""
Comprehensive Tourism Data Collection Script
============================================

Combines all collection, analysis, and discovery features into one unified script.

Features:
---------
1. Multi-platform data collection (Facebook, TikTok, YouTube, Google Maps)
2. Auto-discovery of new attractions
3. Automatic sentiment analysis
4. Topic classification
5. Tourism type classification
6. Statistics and reports
7. Smart duplicate prevention
8. Image collection
9. Language detection
10. Progress tracking and reporting

Usage:
------
1. Basic collection:
   python scripts/collect_data_comprehensive.py

2. Specific provinces:
   python scripts/collect_data_comprehensive.py --provinces "Đà Nẵng,Bình Thuận"

3. Auto-discover new attractions:
   python scripts/collect_data_comprehensive.py --auto-discover --provinces "Quảng Nam" --city "Hội An"

4. Full collection mode:
   python scripts/collect_data_comprehensive.py --full-mode

5. With analysis:
   python scripts/collect_data_comprehensive.py --analyze

6. Complete workflow (discover + collect + analyze + report):
   python scripts/collect_data_comprehensive.py --complete --provinces "Đà Nẵng"

7. Custom limits:
   python scripts/collect_data_comprehensive.py --limit 5 --batch-size 32

8. Force re-analyze:
   python scripts/collect_data_comprehensive.py --analyze --force-reanalyze
"""

import asyncio
import sys
from pathlib import Path
from typing import List, Optional
import argparse
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent.parent))

from app.database.connection import get_db, SessionLocal
from app.models.province import Province
from app.models.tourist_attraction import TouristAttraction
from app.models.social_post import SocialPost
from app.models.comment import Comment
from app.services.analysis_service import AnalysisService
from app.collectors.google_maps_collector import create_google_maps_collector
from app.services.topic_classifier import TopicClassifier
from app.core.config import settings
from sqlalchemy import func
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class ComprehensiveCollector:
    """Unified collector with all features"""
    
    def __init__(self, full_mode: bool = False, use_gpu: bool = False, batch_size: int = 32):
        self.full_mode = full_mode or settings.FULL_COLLECTION_MODE
        self.use_gpu = use_gpu
        self.batch_size = batch_size
        
        # Load settings based on mode
        if self.full_mode:
            self.target_posts = settings.FULL_TARGET_POSTS
            self.target_comments = settings.FULL_TARGET_COMMENTS
            self.platform_limits = settings.FULL_PLATFORM_LIMITS
            logger.info("🚀 FULL COLLECTION MODE - High volume data collection")
        else:
            self.target_posts = settings.TARGET_POSTS_PER_ATTRACTION
            self.target_comments = settings.TARGET_COMMENTS_PER_ATTRACTION
            self.platform_limits = settings.PLATFORM_LIMITS
            logger.info("📊 DEMO/WEEKLY UPDATE MODE - Cost-efficient incremental updates")
        
        self.max_attractions_per_province = settings.MAX_ATTRACTIONS_PER_PROVINCE
        
        logger.info(f"Target: {self.target_comments} comments/attraction, {self.target_posts} posts/attraction")
        logger.info(f"Platform limits: {self.platform_limits}")
        logger.info(f"Max attractions per province: {self.max_attractions_per_province}")
        
        # Initialize services
        self.pipeline = None
        self.analysis_service = None
        self.topic_classifier = None
    
    def _initialize_pipeline(self):
        """Lazy initialization of collection pipeline"""
        if not self.pipeline:
            from app.collectors.smart_collection_pipeline import create_smart_pipeline
            self.pipeline = create_smart_pipeline()
            logger.info("✅ Collection pipeline initialized")
    
    def _initialize_analysis(self):
        """Lazy initialization of analysis services"""
        if not self.analysis_service:
            self.analysis_service = AnalysisService(
                use_gpu=self.use_gpu,
                batch_size=self.batch_size
            )
            logger.info(f"✅ Analysis service initialized (GPU: {self.use_gpu})")
        
        if not self.topic_classifier:
            self.topic_classifier = TopicClassifier()
            logger.info("✅ Topic classifier initialized")
    
    async def auto_discover_attractions(
        self,
        province_name: str,
        city_name: str = None,
        province_code: str = None,
        limit: int = 20,
        dry_run: bool = False
    ) -> dict:
        """Auto-discover new attractions using Google Maps"""
        
        print("\n" + "="*80)
        print("🔍 AUTO-DISCOVERY MODE")
        print("="*80)
        print(f"Province: {province_name}")
        if city_name:
            print(f"City: {city_name}")
        print(f"Limit: {limit} attractions")
        print(f"Mode: {'DRY RUN (Preview)' if dry_run else 'PRODUCTION (Save)'}")
        print("="*80 + "\n")
        
        collector = await create_google_maps_collector()
        
        # Discover attractions
        attractions = await collector.auto_discover_attractions(
            province_name=province_name,
            city_name=city_name,
            limit=limit
        )
        
        if not attractions:
            print("❌ No attractions discovered")
            return {'discovered': 0, 'saved': 0}
        
        print(f"\n✅ Discovered {len(attractions)} attractions")
        
        if dry_run:
            print("\n📋 PREVIEW (not saving to database):")
            for idx, attr in enumerate(attractions, 1):
                print(f"\n{idx}. {attr['name']}")
                print(f"   Category: {attr.get('category', 'N/A')}")
                print(f"   Rating: {attr.get('rating', 'N/A')} ⭐ ({attr.get('reviews_count', 0)} reviews)")
                print(f"   Address: {attr.get('address', 'N/A')}")
            return {'discovered': len(attractions), 'saved': 0}
        
        # Save to database
        db = SessionLocal()
        saved_count = 0
        
        try:
            # Get or create province
            province = db.query(Province).filter(Province.name == province_name).first()
            
            if not province:
                if not province_code:
                    province_code = ''.join([c[0].upper() for c in province_name.split()])[:3]
                
                province = Province(
                    name=province_name,
                    code=province_code,
                    main_city=city_name or province_name
                )
                db.add(province)
                db.commit()
                print(f"\n✅ Created province: {province_name} ({province_code})")
            
            province_id = province.id
            
            # Save attractions
            for attr_data in attractions:
                # Check for duplicates
                existing = db.query(TouristAttraction).filter(
                    TouristAttraction.province_id == province_id,
                    TouristAttraction.name == attr_data['name']
                ).first()
                
                if existing:
                    logger.debug(f"Skipped duplicate: {attr_data['name']}")
                    continue
                
                # Create new attraction
                attraction = TouristAttraction(
                    name=attr_data['name'],
                    province_id=province_id,
                    description=attr_data.get('description'),
                    category=attr_data.get('category'),
                    rating=attr_data.get('rating'),
                    address=attr_data.get('address'),
                    latitude=attr_data.get('latitude'),
                    longitude=attr_data.get('longitude'),
                    image_url=attr_data.get('image_url'),
                    source='google_maps_auto_discovery',
                    is_active=True
                )
                db.add(attraction)
                saved_count += 1
            
            db.commit()
            print(f"\n✅ Saved {saved_count} new attractions to database")
            
        except Exception as e:
            db.rollback()
            logger.error(f"Error saving attractions: {e}")
        finally:
            db.close()
        
        return {'discovered': len(attractions), 'saved': saved_count}
    
    async def collect_for_provinces(
        self,
        province_names: Optional[List[str]] = None,
        limit_per_province: int = None,
        all_attractions: bool = False
    ) -> dict:
        """Collect data for provinces using smart pipeline"""
        
        self._initialize_pipeline()
        
        print("\n" + "="*80)
        print("📊 MULTI-PLATFORM DATA COLLECTION")
        print("="*80)
        
        db = next(get_db())
        
        try:
            # Get provinces
            query = db.query(Province)
            if province_names:
                query = query.filter(Province.name.in_(province_names))
                print(f"Target provinces: {', '.join(province_names)}")
            else:
                print("Target: All provinces")
            
            provinces = query.all()
            
            if not provinces:
                print("❌ No provinces found!")
                return {'provinces': 0, 'attractions': 0, 'posts': 0, 'comments': 0}
            
            print(f"Found {len(provinces)} province(s)")
            print("="*80 + "\n")
            
            total_stats = {
                'provinces': len(provinces),
                'attractions': 0,
                'posts_collected': 0,
                'comments_collected': 0
            }
            
            for province in provinces:
                print(f"\n{'='*80}")
                print(f"Province: {province.name}")
                print(f"{'='*80}\n")
                
                # Determine limit
                if all_attractions:
                    limit = None
                elif limit_per_province:
                    limit = limit_per_province
                else:
                    limit = self.max_attractions_per_province if self.max_attractions_per_province > 0 else 3
                
                # Collect for province
                result = await self.pipeline.collect_for_province(
                    province_id=province.id,
                    limit=limit
                )
                
                total_stats['attractions'] += result.get('attractions_processed', 0)
                total_stats['posts_collected'] += result.get('posts_collected', 0)
                total_stats['comments_collected'] += result.get('comments_collected', 0)
                
                # Wait between provinces
                if len(provinces) > 1:
                    print("\nWaiting 10 seconds before next province...")
                    await asyncio.sleep(10)
            
            return total_stats
            
        finally:
            db.close()
    
    def analyze_all_comments(
        self,
        limit: Optional[int] = None,
        force_reanalyze: bool = False
    ) -> dict:
        """Analyze sentiment and topics for all comments"""
        
        self._initialize_analysis()
        
        print("\n" + "="*80)
        print("🧠 COMPREHENSIVE COMMENT ANALYSIS")
        print("="*80)
        print(f"Mode: {'Re-analyze ALL' if force_reanalyze else 'Only unanalyzed'}")
        if limit:
            print(f"Limit: {limit} comments")
        print("="*80 + "\n")
        
        # Analyze comments
        result = self.analysis_service.analyze_unanalyzed_comments(
            batch_size=self.batch_size,
            limit=limit,
            force_reanalyze=force_reanalyze
        )
        
        return result
    
    def classify_tourism_types(self) -> dict:
        """Classify tourism types for all attractions"""
        
        print("\n" + "="*80)
        print("🏷️  TOURISM TYPE CLASSIFICATION")
        print("="*80 + "\n")
        
        db = SessionLocal()
        
        try:
            attractions = db.query(TouristAttraction).all()
            
            print(f"Processing {len(attractions)} attractions...\n")
            
            # Tourism type keywords
            type_keywords = {
                'beach': ['bãi biển', 'biển', 'beach', 'bờ biển', 'hòn', 'đảo', 'cồn cát', 'mũi', 'vịnh'],
                'mountain': ['núi', 'đồi', 'cao nguyên', 'đèo', 'đỉnh', 'langbiang', 'bidoup', 'fansipan'],
                'historical': ['di tích', 'lịch sử', 'cổ', 'dinh', 'phủ', 'thành', 'pháo đài', 'lăng', 'tháp chàm'],
                'cultural': ['chùa', 'đền', 'thiền viện', 'pagoda', 'temple', 'làng', 'bảo tàng', 'museum'],
                'nature': ['thác', 'hồ', 'vườn quốc gia', 'rừng', 'sinh thái', 'suối', 'khu bảo tồn', 'thung lũng'],
                'urban': ['chợ', 'phố', 'trung tâm', 'quảng trường', 'công viên', 'khu phố', 'đường'],
                'adventure': ['zipline', 'canyoning', 'paragliding', 'dù lượn', 'leo núi', 'mạo hiểm', 'thể thao']
            }
            
            stats = {t: 0 for t in type_keywords.keys()}
            stats['unknown'] = 0
            
            for attraction in attractions:
                text = attraction.name.lower()
                if attraction.description:
                    text += " " + attraction.description.lower()
                
                # Find matches
                matched_type = None
                for tourism_type, keywords in type_keywords.items():
                    if any(keyword in text for keyword in keywords):
                        matched_type = tourism_type
                        break
                
                if matched_type:
                    attraction.tourism_type = matched_type
                    stats[matched_type] += 1
                else:
                    stats['unknown'] += 1
            
            db.commit()
            
            print("✅ Classification complete\n")
            print("Results:")
            for t_type, count in sorted(stats.items(), key=lambda x: x[1], reverse=True):
                if count > 0:
                    print(f"   {t_type:15s}: {count:4d} attractions")
            
            return stats
            
        finally:
            db.close()
    
    def generate_report(self) -> dict:
        """Generate comprehensive database report"""
        
        print("\n" + "="*80)
        print("📈 DATABASE STATISTICS REPORT")
        print("="*80 + "\n")
        
        db = SessionLocal()
        
        try:
            # Province stats
            provinces = db.query(Province).all()
            print(f"Provinces: {len(provinces)}")
            
            # Attraction stats
            attractions = db.query(TouristAttraction).filter(
                TouristAttraction.is_active.is_(True)
            ).all()
            print(f"Active Attractions: {len(attractions)}")
            
            # Posts stats
            posts_total = db.query(SocialPost).count()
            posts_by_platform = db.query(
                SocialPost.platform,
                func.count(SocialPost.id)
            ).group_by(SocialPost.platform).all()
            
            print(f"\nPosts: {posts_total:,}")
            for platform, count in posts_by_platform:
                print(f"   {platform:15s}: {count:,}")
            
            # Comments stats
            comments_total = db.query(Comment).count()
            comments_analyzed = db.query(Comment).filter(
                Comment.sentiment.isnot(None)
            ).count()
            comments_meaningful = db.query(Comment).filter(
                Comment.is_meaningful.is_(True)
            ).count()
            
            print(f"\nComments: {comments_total:,}")
            print(f"   Analyzed: {comments_analyzed:,} ({comments_analyzed/comments_total*100:.1f}%)")
            print(f"   Meaningful: {comments_meaningful:,} ({comments_meaningful/comments_total*100:.1f}%)")
            
            # Language distribution
            lang_dist = db.query(
                Comment.language,
                func.count(Comment.id)
            ).filter(
                Comment.language.isnot(None)
            ).group_by(Comment.language).all()
            
            if lang_dist:
                print("\nLanguage Distribution:")
                for lang, count in sorted(lang_dist, key=lambda x: x[1], reverse=True)[:10]:
                    print(f"   {lang:10s}: {count:,}")
            
            # Sentiment distribution
            sent_dist = db.query(
                Comment.sentiment,
                func.count(Comment.id)
            ).filter(
                Comment.sentiment.isnot(None)
            ).group_by(Comment.sentiment).all()
            
            if sent_dist:
                print("\nSentiment Distribution:")
                for sentiment, count in sent_dist:
                    print(f"   {sentiment:10s}: {count:,}")
            
            # Tourism type distribution
            type_dist = db.query(
                TouristAttraction.tourism_type,
                func.count(TouristAttraction.id)
            ).filter(
                TouristAttraction.tourism_type.isnot(None)
            ).group_by(TouristAttraction.tourism_type).all()
            
            if type_dist:
                print("\nTourism Type Distribution:")
                for t_type, count in sorted(type_dist, key=lambda x: x[1], reverse=True):
                    print(f"   {t_type:15s}: {count}")
            
            # Top attractions by engagement
            top_attractions = db.query(
                TouristAttraction.name,
                Province.name.label('province_name'),
                func.count(Comment.id).label('comment_count')
            ).join(
                Province, TouristAttraction.province_id == Province.id
            ).outerjoin(
                Comment, Comment.attraction_id == TouristAttraction.id
            ).group_by(
                TouristAttraction.id,
                TouristAttraction.name,
                Province.name
            ).order_by(
                func.count(Comment.id).desc()
            ).limit(10).all()
            
            if top_attractions:
                print("\nTop 10 Attractions by Engagement:")
                for idx, (name, province, count) in enumerate(top_attractions, 1):
                    print(f"   {idx:2d}. {name} ({province}): {count:,} comments")
            
            return {
                'provinces': len(provinces),
                'attractions': len(attractions),
                'posts': posts_total,
                'comments': comments_total,
                'comments_analyzed': comments_analyzed,
                'comments_meaningful': comments_meaningful
            }
            
        finally:
            db.close()


async def main():
    """Main entry point"""
    
    parser = argparse.ArgumentParser(
        description='Comprehensive Tourism Data Collection Script',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s --provinces "Đà Nẵng,Bình Thuận" --analyze
  %(prog)s --auto-discover --provinces "Quảng Nam" --city "Hội An"
  %(prog)s --complete --provinces "Đà Nẵng"
  %(prog)s --full-mode --all --analyze --gpu
        """
    )
    
    # Collection options
    parser.add_argument(
        '--provinces',
        type=str,
        help='Comma-separated list of province names'
    )
    parser.add_argument(
        '--limit',
        type=int,
        help='Number of attractions per province'
    )
    parser.add_argument(
        '--all',
        action='store_true',
        help='Process all active attractions'
    )
    parser.add_argument(
        '--full-mode',
        action='store_true',
        help='Use full collection mode (high volume)'
    )
    
    # Discovery options
    parser.add_argument(
        '--auto-discover',
        action='store_true',
        help='Auto-discover new attractions'
    )
    parser.add_argument(
        '--city',
        type=str,
        help='City name for auto-discovery'
    )
    parser.add_argument(
        '--province-code',
        type=str,
        help='Province code for auto-discovery'
    )
    parser.add_argument(
        '--discovery-limit',
        type=int,
        default=20,
        help='Limit for auto-discovery (default: 20)'
    )
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Preview auto-discovery without saving'
    )
    
    # Analysis options
    parser.add_argument(
        '--analyze',
        action='store_true',
        help='Analyze comments after collection'
    )
    parser.add_argument(
        '--force-reanalyze',
        action='store_true',
        help='Re-analyze all comments'
    )
    parser.add_argument(
        '--analysis-limit',
        type=int,
        help='Limit number of comments to analyze'
    )
    parser.add_argument(
        '--batch-size',
        type=int,
        default=32,
        help='Batch size for analysis (default: 32)'
    )
    parser.add_argument(
        '--gpu',
        action='store_true',
        help='Use GPU for analysis'
    )
    
    # Classification options
    parser.add_argument(
        '--classify-types',
        action='store_true',
        help='Classify tourism types for attractions'
    )
    
    # Reporting options
    parser.add_argument(
        '--report',
        action='store_true',
        help='Generate database statistics report'
    )
    
    # Complete workflow
    parser.add_argument(
        '--complete',
        action='store_true',
        help='Run complete workflow (discover + collect + analyze + classify + report)'
    )
    
    args = parser.parse_args()
    
    # Parse province list
    province_list = None
    if args.provinces:
        province_list = [p.strip() for p in args.provinces.split(',')]
    
    # Initialize collector
    collector = ComprehensiveCollector(
        full_mode=args.full_mode,
        use_gpu=args.gpu,
        batch_size=args.batch_size
    )
    
    start_time = datetime.now()
    
    try:
        # Complete workflow
        if args.complete:
            print("\n" + "="*80)
            print("🚀 COMPLETE WORKFLOW MODE")
            print("="*80 + "\n")
            
            # 1. Auto-discover (if provinces specified)
            if province_list and len(province_list) == 1:
                print("\n--- Step 1: Auto-Discovery ---")
                await collector.auto_discover_attractions(
                    province_name=province_list[0],
                    city_name=args.city,
                    province_code=args.province_code,
                    limit=args.discovery_limit,
                    dry_run=False
                )
            
            # 2. Collect data
            print("\n--- Step 2: Data Collection ---")
            await collector.collect_for_provinces(
                province_names=province_list,
                limit_per_province=args.limit,
                all_attractions=args.all
            )
            
            # 3. Analyze
            print("\n--- Step 3: Comment Analysis ---")
            collector.analyze_all_comments()
            
            # 4. Classify types
            print("\n--- Step 4: Tourism Type Classification ---")
            collector.classify_tourism_types()
            
            # 5. Generate report
            print("\n--- Step 5: Final Report ---")
            collector.generate_report()
        
        else:
            # Individual operations
            if args.auto_discover:
                if not province_list:
                    print("❌ Error: --provinces required for auto-discovery")
                    return
                
                for province_name in province_list:
                    await collector.auto_discover_attractions(
                        province_name=province_name,
                        city_name=args.city,
                        province_code=args.province_code,
                        limit=args.discovery_limit,
                        dry_run=args.dry_run
                    )
            
            # Normal collection
            elif not args.analyze and not args.classify_types and not args.report:
                await collector.collect_for_provinces(
                    province_names=province_list,
                    limit_per_province=args.limit,
                    all_attractions=args.all
                )
            
            # Analysis
            if args.analyze:
                collector.analyze_all_comments(
                    limit=args.analysis_limit,
                    force_reanalyze=args.force_reanalyze
                )
            
            # Classification
            if args.classify_types:
                collector.classify_tourism_types()
            
            # Report
            if args.report:
                collector.generate_report()
    
    except KeyboardInterrupt:
        print("\n\n⚠️  Collection interrupted by user")
    except Exception as e:
        logger.error(f"Error during execution: {e}", exc_info=True)
    finally:
        end_time = datetime.now()
        duration = end_time - start_time
        
        print("\n" + "="*80)
        print(f"⏱️  Total execution time: {duration}")
        print("="*80 + "\n")


if __name__ == "__main__":
    asyncio.run(main())
