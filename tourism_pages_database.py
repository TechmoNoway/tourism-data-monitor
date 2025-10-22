"""
Comprehensive Facebook Pages Database for Vietnam Tourism Locations
===================================================================
Danh sách mở rộng các Facebook Pages có liên quan đến 6 địa điểm du lịch

Strategy: Smart Page Selection
- Test nhiều pages trước (sample 5-10 posts)
- Tính engagement score
- Chọn TOP pages có comments cao nhất
- Thu thập full posts + comments từ đó

Cập nhật: 2025-10-21
"""

# ============================================================================
# FACEBOOK PAGES DATABASE - 6 TOURISM LOCATIONS
# ============================================================================

TOURISM_PAGES_DATABASE = {
    # Location 1: Bà Nà Hills (Đà Nẵng)
    "Bà Nà Hills": {
        "province": "Đà Nẵng",
        "priority": "high",
        "pages": [
            # Official & Resort pages
            {
                "url": "https://www.facebook.com/banahills",
                "name": "Bà Nà Hills",
                "type": "official",
                "expected_engagement": "medium"
            },
            {
                "url": "https://www.facebook.com/SunWorldBaNaHills",
                "name": "Sun World Bà Nà Hills",
                "type": "official",
                "expected_engagement": "medium"
            },
            {
                "url": "https://www.facebook.com/DanangFantasyland",
                "name": "Danang Fantasyland",
                "type": "official",
                "expected_engagement": "low"
            },
            # Tourism & City pages
            {
                "url": "https://www.facebook.com/DanangTourism",
                "name": "Đà Nẵng Tourism",
                "type": "tourism_board",
                "expected_engagement": "medium"
            },
            {
                "url": "https://www.facebook.com/danangcity",
                "name": "Đà Nẵng City",
                "type": "city_page",
                "expected_engagement": "medium"
            },
            # Travel & Review pages
            {
                "url": "https://www.facebook.com/DaNangTravel",
                "name": "Da Nang Travel",
                "type": "travel_info",
                "expected_engagement": "medium"
            },
        ]
    },
    
    # Location 2: Đà Lạt & Hồ Tuyền Lâm (Lâm Đồng)
    "Đà Lạt": {
        "province": "Lâm Đồng",
        "priority": "high",
        "pages": [
            # Official Tourism
            {
                "url": "https://www.facebook.com/DaLatTourism",
                "name": "Đà Lạt Tourism",
                "type": "tourism_board",
                "expected_engagement": "high"
            },
            {
                "url": "https://www.facebook.com/DalatVietnam",
                "name": "Dalat Vietnam",
                "type": "tourism_info",
                "expected_engagement": "medium"
            },
            # Attractions
            {
                "url": "https://www.facebook.com/ThienVienTrucLam",
                "name": "Thiền viện Trúc Lâm",
                "type": "attraction",
                "expected_engagement": "medium"
            },
            {
                "url": "https://www.facebook.com/TuyenLamLake",
                "name": "Hồ Tuyền Lâm",
                "type": "attraction",
                "expected_engagement": "low"
            },
            # City & Travel
            {
                "url": "https://www.facebook.com/dalatcity",
                "name": "Đà Lạt City",
                "type": "city_page",
                "expected_engagement": "medium"
            },
            {
                "url": "https://www.facebook.com/DaLatTravel",
                "name": "Da Lat Travel Guide",
                "type": "travel_info",
                "expected_engagement": "medium"
            },
        ]
    },
    
    # Location 3: Phú Quốc (Kiên Giang)
    "Phú Quốc": {
        "province": "Kiên Giang",
        "priority": "high",
        "pages": [
            # Resorts & Attractions
            {
                "url": "https://www.facebook.com/VinpearlPhuQuoc",
                "name": "Vinpearl Phú Quốc",
                "type": "resort",
                "expected_engagement": "high"
            },
            {
                "url": "https://www.facebook.com/GrandWorldPhuQuoc",
                "name": "Grand World Phú Quốc",
                "type": "attraction",
                "expected_engagement": "high"
            },
            {
                "url": "https://www.facebook.com/VinWondersPhuQuoc",
                "name": "VinWonders Phú Quốc",
                "type": "attraction",
                "expected_engagement": "medium"
            },
            # Tourism
            {
                "url": "https://www.facebook.com/PhuQuocTourism",
                "name": "Phú Quốc Tourism",
                "type": "tourism_board",
                "expected_engagement": "medium"
            },
            {
                "url": "https://www.facebook.com/PhuQuocIsland",
                "name": "Phú Quốc Island",
                "type": "tourism_info",
                "expected_engagement": "medium"
            },
            {
                "url": "https://www.facebook.com/phuquoctravel",
                "name": "Phú Quốc Travel",
                "type": "travel_info",
                "expected_engagement": "medium"
            },
        ]
    },
    
    # Location 4: Mũi Né & Đồi cát bay (Bình Thuận)
    "Mũi Né": {
        "province": "Bình Thuận",
        "priority": "medium",
        "pages": [
            # Resorts
            {
                "url": "https://www.facebook.com/AnantaraMuiNe",
                "name": "Anantara Mũi Né Resort",
                "type": "resort",
                "expected_engagement": "medium"
            },
            {
                "url": "https://www.facebook.com/SeaLinksMuiNe",
                "name": "Sea Links Mũi Né",
                "type": "resort",
                "expected_engagement": "low"
            },
            # Tourism & Travel
            {
                "url": "https://www.facebook.com/MuiNeBeach",
                "name": "Mũi Né Beach",
                "type": "tourism_info",
                "expected_engagement": "medium"
            },
            {
                "url": "https://www.facebook.com/BinhThuanTourism",
                "name": "Bình Thuận Tourism",
                "type": "tourism_board",
                "expected_engagement": "medium"
            },
            {
                "url": "https://www.facebook.com/PhanThietTravel",
                "name": "Phan Thiết Travel",
                "type": "travel_info",
                "expected_engagement": "medium"
            },
            # Attractions
            {
                "url": "https://www.facebook.com/WhiteSandDunesMuiNe",
                "name": "White Sand Dunes Mũi Né",
                "type": "attraction",
                "expected_engagement": "low"
            },
        ]
    },
    
    # Location 5: Nha Trang (Khánh Hòa)
    "Nha Trang": {
        "province": "Khánh Hòa",
        "priority": "high",
        "pages": [
            # Resorts & Attractions
            {
                "url": "https://www.facebook.com/VinpearlNhaTrang",
                "name": "Vinpearl Nha Trang",
                "type": "resort",
                "expected_engagement": "high"
            },
            {
                "url": "https://www.facebook.com/VinWondersNhaTrang",
                "name": "VinWonders Nha Trang",
                "type": "attraction",
                "expected_engagement": "high"
            },
            {
                "url": "https://www.facebook.com/HonTremIsland",
                "name": "Hòn Tre Island",
                "type": "attraction",
                "expected_engagement": "medium"
            },
            # Tourism
            {
                "url": "https://www.facebook.com/NhaTrangTourism",
                "name": "Nha Trang Tourism",
                "type": "tourism_board",
                "expected_engagement": "high"
            },
            {
                "url": "https://www.facebook.com/nhatrangcity",
                "name": "Nha Trang City",
                "type": "city_page",
                "expected_engagement": "medium"
            },
            {
                "url": "https://www.facebook.com/NhaTrangTravel",
                "name": "Nha Trang Travel",
                "type": "travel_info",
                "expected_engagement": "medium"
            },
        ]
    },
    
    # Location 6: Mỹ Khê Beach (Đà Nẵng)
    "Mỹ Khê Beach": {
        "province": "Đà Nẵng",
        "priority": "medium",
        "pages": [
            # Beach & Tourism
            {
                "url": "https://www.facebook.com/MyKheBeach",
                "name": "Mỹ Khê Beach",
                "type": "attraction",
                "expected_engagement": "medium"
            },
            {
                "url": "https://www.facebook.com/DanangBeaches",
                "name": "Đà Nẵng Beaches",
                "type": "tourism_info",
                "expected_engagement": "medium"
            },
            # Shared with Bà Nà Hills (same city)
            {
                "url": "https://www.facebook.com/DanangTourism",
                "name": "Đà Nẵng Tourism",
                "type": "tourism_board",
                "expected_engagement": "medium"
            },
            {
                "url": "https://www.facebook.com/danangcity",
                "name": "Đà Nẵng City",
                "type": "city_page",
                "expected_engagement": "medium"
            },
            {
                "url": "https://www.facebook.com/DaNangTravel",
                "name": "Da Nang Travel",
                "type": "travel_info",
                "expected_engagement": "medium"
            },
        ]
    },
}

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def get_all_locations():
    """Get list of all location names"""
    return list(TOURISM_PAGES_DATABASE.keys())

def get_pages_for_location(location_name):
    """Get all pages for a specific location"""
    if location_name in TOURISM_PAGES_DATABASE:
        return TOURISM_PAGES_DATABASE[location_name]["pages"]
    return []

def get_high_priority_locations():
    """Get locations marked as high priority"""
    return [
        name for name, data in TOURISM_PAGES_DATABASE.items()
        if data.get("priority") == "high"
    ]

def get_all_page_urls():
    """Get all unique page URLs across all locations"""
    urls = set()
    for location_data in TOURISM_PAGES_DATABASE.values():
        for page in location_data["pages"]:
            urls.add(page["url"])
    return sorted(list(urls))

def get_pages_by_type(page_type):
    """Get all pages of a specific type (official, tourism_board, etc.)"""
    pages = []
    for location_name, location_data in TOURISM_PAGES_DATABASE.items():
        for page in location_data["pages"]:
            if page.get("type") == page_type:
                pages.append({
                    "location": location_name,
                    **page
                })
    return pages

def print_database_summary():
    """Print summary of the database"""
    print("=" * 70)
    print("VIETNAM TOURISM PAGES DATABASE - SUMMARY")
    print("=" * 70)
    print()
    
    total_pages = len(get_all_page_urls())
    print(f"📊 Total unique pages: {total_pages}")
    print(f"📍 Total locations: {len(get_all_locations())}")
    print(f"⭐ High priority locations: {len(get_high_priority_locations())}")
    print()
    
    for location_name in get_all_locations():
        location_data = TOURISM_PAGES_DATABASE[location_name]
        priority = location_data["priority"]
        num_pages = len(location_data["pages"])
        
        priority_emoji = "⭐" if priority == "high" else "📌"
        print(f"{priority_emoji} {location_name} ({location_data['province']})")
        print(f"   Pages: {num_pages} | Priority: {priority}")
        
        # Count by type
        types = {}
        for page in location_data["pages"]:
            page_type = page.get("type", "unknown")
            types[page_type] = types.get(page_type, 0) + 1
        
        type_summary = ", ".join([f"{count} {t}" for t, count in types.items()])
        print(f"   Types: {type_summary}")
        print()

if __name__ == "__main__":
    print_database_summary()
    
    print()
    print("=" * 70)
    print("EXAMPLE USAGE")
    print("=" * 70)
    print()
    
    # Example 1: Get pages for a location
    print("Example 1: Get pages for Bà Nà Hills")
    pages = get_pages_for_location("Bà Nà Hills")
    for page in pages[:3]:
        print(f"  - {page['name']}: {page['url']}")
    print()
    
    # Example 2: Get high priority locations
    print("Example 2: High priority locations")
    for loc in get_high_priority_locations():
        print(f"  - {loc}")
    print()
    
    # Example 3: Get pages by type
    print("Example 3: All official pages")
    official_pages = get_pages_by_type("official")
    for page in official_pages:
        print(f"  - {page['name']} ({page['location']})")
