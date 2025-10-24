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

### `analyze_sentiment.py` - Sentiment Analysis
Analyze comments with multi-language sentiment analysis using AI models.

**Usage:**
```bash
# Analyze all unanalyzed comments
python scripts/analyze_sentiment.py

# Analyze with custom batch size
python scripts/analyze_sentiment.py --batch-size 16

# Re-analyze all comments (force)
python scripts/analyze_sentiment.py --force

# Analyze only first 100 comments (testing)
python scripts/analyze_sentiment.py --limit 100

# Use GPU acceleration (requires CUDA)
python scripts/analyze_sentiment.py --gpu
```

**Features:**
- Multi-language support (22+ languages)
- PhoBERT for Vietnamese (~92% accuracy)
- mBERT for other languages (~85% accuracy)
- Batch processing for performance
- Automatic language detection
- Text cleaning and preprocessing
- Rule-based fallback

**Models:**
- **PhoBERT**: `wonrax/phobert-base-vietnamese-sentiment` (Vietnamese)
- **mBERT**: `nlptown/bert-base-multilingual-uncased-sentiment` (100+ languages)

---

### `check_data.py` - Data Verification & Statistics
Check database statistics and verify data integrity.

**Usage:**
```bash
# Show full database statistics (including sentiment analysis)
python scripts/check_data.py

# Just verify database connection
python scripts/check_data.py --verify
```

**Output:**
- Total counts (provinces, attractions, posts, comments)
- Breakdown by platform
- Breakdown by attraction
- Average statistics
- **Sentiment analysis results**
- **Language distribution**
- **Model usage statistics**
- **Confidence scores**

---

## Workflow

### 1. Initial Setup
```bash
# Setup database
python scripts/recreate_db.py
```

### 2. Data Collection
```bash
# Collect data from social platforms
python scripts/collect_data.py --provinces "Bình Thuận,Đà Nẵng,Lâm Đồng"
```

### 3. Sentiment Analysis
```bash
# Analyze collected comments
python scripts/analyze_sentiment.py --batch-size 16
```

### 4. Check Results
```bash
# View statistics and results
python scripts/check_data.py
```

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

## Sentiment Analysis Details

### Language Support
- Vietnamese (vi) - Primary language, PhoBERT model
- English (en), Korean (ko), Chinese (zh-cn), Japanese (ja), Thai (th)
- 100+ languages supported via mBERT

### Analysis Fields
Each comment gets:
- `sentiment`: positive/neutral/negative
- `sentiment_score`: confidence (0.0-1.0)
- `language`: detected language code
- `analysis_model`: phobert/mbert/rule-based
- `cleaned_content`: preprocessed text
- `word_count`: number of words
- `is_valid`: quality filter result

### Performance
- **CPU Mode**: ~16 comments/second (batch size 16)
- **GPU Mode**: ~100+ comments/second (requires CUDA)
- **PhoBERT**: ~400MB model, 135M parameters
- **mBERT**: ~669MB model, 110M parameters

### Performance
- **CPU Mode**: ~16 comments/second (batch size 16)
- **GPU Mode**: ~100+ comments/second (requires CUDA)
- **PhoBERT**: ~400MB model, 135M parameters
- **mBERT**: ~669MB model, 110M parameters

---

## Development Notes

### Removed Files (Cleanup Complete)
**Old test/demo files:**
- `demo_facebook_integration.py` - Removed (integrated into main collector)
- `test_google_maps_reviews.py` - Removed (testing complete)
- `collect_three_provinces.py` - Removed (replaced by `collect_data.py`)
- `seed_manual_collection.py` - Removed (deprecated)
- `seed_three_provinces.py` - Removed (deprecated)
- `tourism_pages_database.py` - Removed (old database script)
- `verify_db.py` - Removed (merged into `check_data.py`)
- `check_all_data.py` - Removed (merged into `check_data.py`)

**Temporary migration/test scripts:**
- `migrate_comment_schema.py` - Removed (migration complete)
- `test_sentiment.py` - Removed (testing complete)
- `verify_new_schema.py` - Removed (verification complete)

**Moved files:**
- `recreate_db.py` - Moved from root to `scripts/`
- `facebook_best_pages_config.py` - Moved to `app/core/facebook_best_pages.py`

### Future Improvements (See TODO list)
- Facebook collector upgrades (direct page URLs only)
- Google Maps coverage expansion (20-30 places)
- TikTok comment collection fix
- YouTube collector testing
- Topic classification implementation
- Spam detection
- Advanced relevance filtering (NLP)
- Data quality validation layer
- Analytics dashboard
