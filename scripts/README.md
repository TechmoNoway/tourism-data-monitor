# Scripts Directory

This directory contains utility scripts for the tourism data monitoring system.

## Main Scripts

### `recreate_db.py` - Database Setup
Initialize or reset the database with clean schema and seed data.

**Usage:**
```bash
# Create/recreate database tables
python scripts/recreate_db.py
```

**Features:**
- Drops all existing tables
- Creates fresh schema from models
- Drops materialized views (if any)
- Safe to run multiple times

**When to use:**
- First-time setup
- After model changes
- When database is corrupted
- To reset all data

---

### `collect_data.py` - Data Collection
Main script to collect posts and comments from multiple social platforms.

**Usage:**
```bash
# Collect for specific provinces (3 attractions per province by default)
python scripts/collect_data.py --provinces "Bình Thuận,Đà Nẵng,Lâm Đồng"

# Collect for specific provinces with custom limit
python scripts/collect_data.py --provinces "Bình Thuận" --limit 5

# Collect for ALL active attractions in all provinces
python scripts/collect_data.py --all
```

**Features:**
- Multi-platform collection (Facebook, Google Maps, TikTok, YouTube)
- Platform priority order for optimal data gathering
- Automatic target-based stopping (40 comments per attraction)
- Rate limiting between attractions
- Comprehensive progress reporting

---

### `check_data.py` - Data Verification
Check database statistics and verify data integrity.

**Usage:**
```bash
# Show full database statistics
python scripts/check_data.py

# Just verify database connection
python scripts/check_data.py --verify
```

**Output:**
- Total counts (provinces, attractions, posts, comments)
- Breakdown by platform
- Breakdown by attraction
- Average statistics

---

## Collection Strategy

### Platform Priority
1. **Facebook** - Rich comments, good engagement
2. **Google Maps** - High-quality reviews (13+ places per attraction)
3. **YouTube** - Video comments
4. **TikTok** - Video posts (limited comment support)

### Targets
- **Posts:** 8 per attraction
- **Comments:** 40 per attraction
- **Stop condition:** 80% of target reached (32 comments)

### Rate Limiting
- 5-second delay between attractions
- Platform API rate limits handled automatically

---

## Development Notes

### Previous Files (Removed/Moved)
- `demo_facebook_integration.py` - Removed (integrated into main collector)
- `test_google_maps_reviews.py` - Removed (testing complete)
- `collect_three_provinces.py` - Removed (replaced by `collect_data.py`)
- `seed_manual_collection.py` - Removed (deprecated)
- `seed_three_provinces.py` - Removed (deprecated)
- `tourism_pages_database.py` - Removed (old database script)
- `verify_db.py` - Removed (merged into `check_data.py`)
- `check_all_data.py` - Removed (merged into `check_data.py`)
- `recreate_db.py` - Moved from root to `scripts/` (better organization)
- `facebook_best_pages_config.py` - Moved to `app/core/facebook_best_pages.py`

### Future Improvements (See TODO list)
- Facebook collector upgrades (direct page URLs only)
- Google Maps coverage expansion (20-30 places)
- TikTok comment collection fix
- YouTube collector testing
- Advanced relevance filtering (NLP, sentiment)
- Data quality validation layer
