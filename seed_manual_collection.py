"""
seed_manual_collection.py

Manual seeding script to run an on-demand data collection for one or more
attractions using the integrated pipeline. Replaces older root-level test
scripts by providing a single CLI-driven entrypoint.

Usage examples:
  # collect default three best-pages with defaults from settings
  python seed_manual_collection.py --all

  # collect for a single attraction (name matches DB or will create)
  python seed_manual_collection.py --attraction "Bà Nà Hills" --province "Đà Nẵng" --limit 10

  # force using best-pages or force using raw page URL
  python seed_manual_collection.py --attraction "Bà Nà Hills" --province "Đà Nẵng" --force-page-url "https://www.facebook.com/SunWorldBaNaHills"

"""

import argparse
import asyncio
import os
import sys
from datetime import datetime

# allow running from repo root
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from app.core.config import settings
from app.collectors.data_pipeline import DataCollectionPipeline
from app.database.connection import get_db
from app.models.province import Province
from app.models.tourist_attraction import TouristAttraction


async def collect_for(attraction_name: str, province_name: str, pipeline: DataCollectionPipeline, limit: int, force_page_url: str | None, use_best_pages: bool):
    db = next(get_db())

    # find or create province
    province = db.query(Province).filter(Province.name.ilike(f"%{province_name}%")).first()
    if not province:
        print(f"Creating province: {province_name}")
        province = Province(name=province_name, code=province_name[:3].upper())
        db.add(province)
        db.commit()
        db.refresh(province)

    # find or create attraction
    attraction = db.query(TouristAttraction).filter(
        TouristAttraction.name.ilike(f"%{attraction_name}%"),
        TouristAttraction.province_id == province.id
    ).first()

    if not attraction:
        print(f"Creating attraction: {attraction_name}")
        attraction = TouristAttraction(
            name=attraction_name,
            province_id=province.id,
            description=f"Manual-seeding attraction: {attraction_name}"
        )
        db.add(attraction)
        db.commit()
        db.refresh(attraction)

    print(f"Starting collection for '{attraction.name}' (id={attraction.id})")

    # Build platform options. DataCollectionPipeline will pick best-pages by default
    opts = {
        'force_page_url': force_page_url,
        'use_best_pages': use_best_pages
    }

    result = await pipeline.collect_for_attraction(
        attraction_id=attraction.id,
        platforms=['facebook'],
        limit_per_platform=limit,
        platform_options={'facebook': opts}
    )

    # Pretty print result
    print("\n=== RESULT ===")
    print(f"Started: {datetime.utcnow().isoformat()}")
    print(f"Attraction: {attraction.name} ({province.name})")
    print(f"Result summary: total_posts={result.get('total_posts')}, total_comments={result.get('total_comments')}")
    platforms = result.get('platforms', {})
    if 'facebook' in platforms:
        fb = platforms['facebook']
        print(f" facebook.strategy: {fb.get('strategy')}")
        print(f" facebook.best_page_used: {fb.get('best_page_used')}")
        print(f" facebook.posts_collected: {fb.get('posts_collected')}")
        print(f" facebook.comments_collected: {fb.get('comments_collected')}")

    if result.get('errors'):
        print('\nErrors:')
        for e in result['errors']:
            print(' -', e)

    db.close()


def parse_args():
    p = argparse.ArgumentParser(description='Manual seeding collection script')
    p.add_argument('--all', action='store_true', help='Run for all configured best-pages')
    p.add_argument('--attraction', type=str, help='Attraction name to collect')
    p.add_argument('--province', type=str, help='Province name for the attraction')
    p.add_argument('--limit', type=int, default=settings.FACEBOOK_POSTS_PER_LOCATION, help='Number of posts per location')
    p.add_argument('--comments', type=int, default=settings.FACEBOOK_COMMENTS_PER_POST, help='Comments per post')
    p.add_argument('--force-page-url', type=str, default=None, help='If provided, force using this page URL instead of keyword search')
    p.add_argument('--no-best-pages', action='store_true', help="Don't use best-pages, force keyword search")
    return p.parse_args()


def main():
    args = parse_args()

    apify_token = os.getenv('APIFY_API_TOKEN')
    if not apify_token:
        print('APIFY_API_TOKEN not set in environment. Set it and retry.')
        return

    pipeline = DataCollectionPipeline(apify_api_token=apify_token)

    # quick list of default best pages to run when --all
    default_cases = []
    for key, cfg in settings.FACEBOOK_BEST_PAGES.items():
        # try to infer a simple attraction/province label from key
        if key == 'ba_na_hills':
            default_cases.append({'name': 'Bà Nà Hills', 'province': 'Đà Nẵng'})
        elif key == 'da_lat':
            default_cases.append({'name': 'Đà Lạt', 'province': 'Lâm Đồng'})
        elif key == 'phu_quoc':
            default_cases.append({'name': 'Phú Quốc', 'province': 'Kiên Giang'})

    # decide which cases to run
    cases = []
    if args.all:
        cases = default_cases
    elif args.attraction and args.province:
        cases = [{'name': args.attraction, 'province': args.province}]
    else:
        print('Specify --all or both --attraction and --province')
        return

    use_best_pages = not args.no_best_pages

    # run sequentially to avoid rate-limits
    async def _run():
        for c in cases:
            await collect_for(
                attraction_name=c['name'],
                province_name=c['province'],
                pipeline=pipeline,
                limit=args.limit,
                force_page_url=args.force_page_url,
                use_best_pages=use_best_pages
            )
            # small pause between runs
            await asyncio.sleep(5)

    asyncio.run(_run())


if __name__ == '__main__':
    main()
