"""
Facebook Best Pages Configuration
Based on test_facebook_full_smart.py results from 2025-10-21

This configuration contains the best performing Facebook pages for each location,
validated through Smart Page Selection strategy with actual comment collection data.
"""

# Best pages configuration based on actual test results
FACEBOOK_BEST_PAGES = {
    "Bà Nà Hills": {
        "location_id": "ba_na_hills",
        "province": "Đà Nẵng",
        "priority": "high",
        "best_page": {
            "name": "Sun World Bà Nà Hills",
            "url": "https://www.facebook.com/SunWorldBaNaHills",
            "type": "official",
            "tested_avg_comments": 8.0,  # From sampling
            "actual_avg_comments": 6.1,  # From full collection (15 posts, 91 comments)
            "engagement_score": 80.0,
            "validation_date": "2025-10-21",
            "collection_success": True
        },
        "backup_pages": [
            {
                "name": "Đà Nẵng Tourism",
                "url": "https://www.facebook.com/DanangTourism",
                "tested_avg_comments": 2.0,
                "engagement_score": 20.0
            },
            {
                "name": "Bà Nà Hills",
                "url": "https://www.facebook.com/banahills",
                "tested_avg_comments": 0.4,
                "engagement_score": 4.0
            }
        ]
    },
    
    "Đà Lạt": {
        "location_id": "da_lat",
        "province": "Lâm Đồng",
        "priority": "high",
        "best_page": {
            "name": "Hồ Tuyền Lâm",
            "url": "https://www.facebook.com/TuyenLamLake",
            "type": "attraction",
            "tested_avg_comments": 13.0,  # From sampling  
            "actual_avg_comments": 11.2,  # From full collection (6 posts, 67 comments)
            "engagement_score": 130.0,  # HIGHEST!
            "validation_date": "2025-10-21",
            "collection_success": True
        },
        "backup_pages": [
            {
                "name": "Đà Lạt City",
                "url": "https://www.facebook.com/dalatcity",
                "tested_avg_comments": 2.4,
                "engagement_score": 24.0
            },
            {
                "name": "Thiền viện Trúc Lâm",
                "url": "https://www.facebook.com/ThienVienTrucLam",
                "tested_avg_comments": 0.6,
                "engagement_score": 6.0
            }
        ]
    },
    
    "Phú Quốc": {
        "location_id": "phu_quoc",
        "province": "Kiên Giang",
        "priority": "high",
        "best_page": {
            "name": "Phú Quốc Island",
            "url": "https://www.facebook.com/PhuQuocIsland",
            "type": "tourism_info",
            "tested_avg_comments": 4.0,  # From sampling
            "actual_avg_comments": 3.7,  # From full collection (3 posts, 11 comments)
            "engagement_score": 40.0,
            "validation_date": "2025-10-21",
            "collection_success": True
        },
        "backup_pages": [
            {
                "name": "Vinpearl Phú Quốc",
                "url": "https://www.facebook.com/VinpearlPhuQuoc",
                "tested_avg_comments": 0.4,
                "engagement_score": 4.0
            },
            {
                "name": "VinWonders Phú Quốc",
                "url": "https://www.facebook.com/VinWondersPhuQuoc",
                "tested_avg_comments": 0.0,
                "engagement_score": 0.0
            }
        ]
    }
}

# Summary statistics from validation
VALIDATION_SUMMARY = {
    "test_date": "2025-10-21",
    "total_locations_tested": 4,
    "successful_locations": 3,
    "failed_locations": 1,  # Nha Trang - low engagement
    "total_posts_collected": 24,
    "total_comments_collected": 169,
    "average_comments_per_post": 7.0,
    "estimated_cost_usd": 0.41,
    "test_duration_minutes": 18
}

# Production collection parameters (based on test results)
PRODUCTION_CONFIG = {
    "posts_per_location": 20,  # Increased from test's 15
    "comments_per_post": 50,
    "sample_size_for_validation": 5,
    "min_engagement_threshold": 0.5,  # Minimum avg comments/post to consider page viable
    "use_backup_if_best_fails": True,
    "retry_attempts": 3,
    "delay_between_locations_seconds": 3
}

def get_best_page_url(location_name):
    """Get the best page URL for a location"""
    if location_name in FACEBOOK_BEST_PAGES:
        return FACEBOOK_BEST_PAGES[location_name]["best_page"]["url"]
    return None

def get_all_best_pages():
    """Get all best pages as a simple dict mapping location to URL"""
    return {
        location: config["best_page"]["url"]
        for location, config in FACEBOOK_BEST_PAGES.items()
    }

def get_expected_comments_per_post(location_name):
    """Get expected comments per post for a location (for capacity planning)"""
    if location_name in FACEBOOK_BEST_PAGES:
        return FACEBOOK_BEST_PAGES[location_name]["best_page"]["actual_avg_comments"]
    return 0

def estimate_collection_cost(num_posts=20, comments_per_post=50):
    """
    Estimate cost for collection based on validation data
    
    Args:
        num_posts: Number of posts to collect per location
        comments_per_post: Max comments to collect per post
        
    Returns:
        dict with cost breakdown
    """
    # From validation: 7.0 avg comments/post across all locations
    # But use actual expected comments for more accurate estimate
    total_locations = len(FACEBOOK_BEST_PAGES)
    expected_comments_per_location = {
        "Bà Nà Hills": 6.1 * num_posts,
        "Đà Lạt": 11.2 * num_posts,
        "Phú Quốc": 3.7 * num_posts
    }
    
    total_expected_comments = sum(expected_comments_per_location.values())
    total_posts = num_posts * total_locations
    
    # Pricing: $0.01 per post, $0.001 per comment
    post_cost = total_posts * 0.01
    comment_cost = total_expected_comments * 0.001
    total_cost = post_cost + comment_cost
    
    return {
        "total_locations": total_locations,
        "posts_per_location": num_posts,
        "total_posts": total_posts,
        "expected_comments": expected_comments_per_location,
        "total_expected_comments": int(total_expected_comments),
        "post_cost_usd": round(post_cost, 2),
        "comment_cost_usd": round(comment_cost, 2),
        "total_cost_usd": round(total_cost, 2),
        "estimated_duration_minutes": int((total_posts + total_expected_comments / 10) / 2)  # Rough estimate
    }

if __name__ == "__main__":
    print("="*70)
    print("FACEBOOK BEST PAGES CONFIGURATION")
    print("="*70)
    print(f"\n📅 Validated: {VALIDATION_SUMMARY['test_date']}")
    print(f"✅ Successful locations: {VALIDATION_SUMMARY['successful_locations']}")
    print(f"📊 Test results: {VALIDATION_SUMMARY['total_posts_collected']} posts, "
          f"{VALIDATION_SUMMARY['total_comments_collected']} comments")
    
    print("\n" + "="*70)
    print("BEST PAGES BY LOCATION")
    print("="*70)
    
    for location, config in FACEBOOK_BEST_PAGES.items():
        best = config["best_page"]
        print(f"\n📍 {location} ({config['province']})")
        print(f"   🏆 Best Page: {best['name']}")
        print(f"   🔗 URL: {best['url']}")
        print(f"   📊 Engagement: {best['engagement_score']:.1f} score")
        print(f"   💬 Comments: {best['actual_avg_comments']:.1f} avg/post")
        print(f"   ✅ Validated: {best['validation_date']}")
        
        if config.get("backup_pages"):
            print(f"   📋 Backup pages: {len(config['backup_pages'])} available")
    
    print("\n" + "="*70)
    print("PRODUCTION COST ESTIMATE (20 posts/location)")
    print("="*70)
    
    estimate = estimate_collection_cost(
        num_posts=PRODUCTION_CONFIG["posts_per_location"],
        comments_per_post=PRODUCTION_CONFIG["comments_per_post"]
    )
    
    print(f"\n📝 Total posts: {estimate['total_posts']}")
    print(f"💬 Expected comments: {estimate['total_expected_comments']}")
    print(f"💰 Estimated cost: ${estimate['total_cost_usd']}")
    print(f"⏱️  Estimated time: ~{estimate['estimated_duration_minutes']} minutes")
    
    print("\n📊 Comments breakdown by location:")
    for loc, count in estimate['expected_comments'].items():
        print(f"   {loc}: ~{int(count)} comments")
    
    print("\n" + "="*70)
    print("READY FOR PRODUCTION INTEGRATION!")
    print("="*70)
