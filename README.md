# Tourism Data Monitor 🏖️

> A comprehensive system for monitoring and analyzing tourism data from multiple social media platforms for Vietnamese provinces.

## 🎯 Overview

This system collects and analyzes comments, posts, and reviews about tourist attractions from:
- ✅ **YouTube** (via YouTube Data API v3)
- ✅ **Google Reviews** (via Google Places API)
- ✅ **Facebook** (via Apify scraper)
- ✅ **TikTok** (via Apify scraper)

**Target provinces**: Lâm Đồng, Đà Nẵng, Bình Thuận

---

## 🚀 Quick Start

### 1. Prerequisites
- Python 3.9+
- PostgreSQL
- API keys (see setup below)

### 2. Installation

```bash
# Clone repository
git clone <repository-url>
cd tourism_data_monitor

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or
venv\Scripts\activate  # Windows

# Install dependencies
pip install -r requirements.txt
```

### 3. Setup Credentials

Copy the example env file:
```bash
cp .env.example .env
```

Edit `.env` and add your API keys:
```env
# Required APIs
YOUTUBE_API_KEY=your_youtube_key_here
GOOGLE_MAPS_API_KEY=your_google_maps_key_here
APIFY_API_TOKEN=apify_api_your_token_here

# Database
DATABASE_URL=sqlite:///./tourism.db
```

**Get API keys:**
- **YouTube & Google Maps**: [docs/API_CREDENTIALS_SETUP.md](docs/API_CREDENTIALS_SETUP.md)
- **Apify**: [docs/APIFY_SETUP.md](docs/APIFY_SETUP.md) or [docs/APIFY_QUICKSTART.md](docs/APIFY_QUICKSTART.md)

### 4. Test Your Setup

```bash
python test/test_credentials.py
```

Expected output:
```
✅ YouTube                WORKING
✅ Google Maps            WORKING
✅ Apify (FB/TikTok)      WORKING

🎉 All credentials are working!
```

### 5. Initialize Database

```bash
# Create database and seed initial data
python scripts/recreate_db.py
```

### 6. Collect Data

```bash
# Collect for specific provinces (recommended for testing)
python scripts/collect_data.py --provinces "Bình Thuận,Đà Nẵng,Lâm Đồng" --limit 3

# Collect for all active attractions
python scripts/collect_data.py --all
```

### 7. Verify Data

```bash
# Check database statistics
python scripts/check_data.py

# Just verify connection
python scripts/check_data.py --verify
```

### 8. Run API Server (Optional)

```bash
python run.py
# or
uvicorn app.main:app --reload
```

Visit: http://localhost:8000/docs for API documentation

---

## 📚 Documentation

- **[APIFY_QUICKSTART.md](docs/APIFY_QUICKSTART.md)** - 5-minute setup guide (START HERE!)
- **[APIFY_SETUP.md](docs/APIFY_SETUP.md)** - Complete Apify setup guide
- **[API_CREDENTIALS_SETUP.md](docs/API_CREDENTIALS_SETUP.md)** - YouTube & Google Maps setup
- **[COLLECTOR_CHANGES.md](docs/COLLECTOR_CHANGES.md)** - Platform changes and migration
- **[APIFY_INTEGRATION_SUMMARY.md](APIFY_INTEGRATION_SUMMARY.md)** - Technical summary

---

## 🏗️ Architecture

### Tech Stack
- **Backend**: FastAPI, SQLAlchemy, Pydantic v2
- **Database**: PostgreSQL or SQLite
- **Collectors**: YouTube API, Google Places API, Apify scrapers
- **Scheduling**: APScheduler
- **NLP**: PhoBERT (planned)

### Project Structure
```
tourism_data_monitor/
├── app/
│   ├── api/              # API endpoints
│   │   ├── routes.py
│   │   └── endpoints/    # Province, attraction, collection endpoints
│   ├── collectors/       # Data collection modules
│   │   ├── base_collector.py           # Base class with dict mapping
│   │   ├── data_pipeline.py            # Multi-platform orchestrator
│   │   ├── facebook_apify_collector.py # Facebook via Apify
│   │   ├── tiktok_apify_collector.py   # TikTok via Apify
│   │   ├── youtube_collector.py        # YouTube API
│   │   ├── google_maps_apify_collector.py # Google Maps via Apify
│   │   ├── relevance_filter.py         # Content filtering
│   │   └── scheduler.py                # Automated scheduling
│   ├── models/           # SQLAlchemy ORM models
│   ├── schemas/          # Pydantic v2 schemas
│   ├── services/         # Business logic
│   ├── core/             # Configuration
│   │   ├── config.py                # Main settings
│   │   └── facebook_best_pages.py   # Facebook pages config
│   └── database/         # Database connection
├── scripts/              # Utility scripts
│   ├── collect_data.py   # Main data collection script
│   ├── check_data.py     # Database verification
│   ├── recreate_db.py    # Database setup/reset
│   └── README.md         # Scripts documentation
├── docs/                 # Documentation
├── test/                 # Tests
├── run.py               # API server entry point
└── requirements.txt
```

### Database Schema
```
provinces
├── id, name, code

tourist_attractions
├── id, name, province_id
├── description, location

social_posts
├── id, platform, platform_post_id
├── attraction_id, content, author
├── post_date, engagement metrics

comments
├── id, platform, platform_comment_id
├── post_id, attraction_id, content
├── author, comment_date, sentiment

analysis_logs
├── id, attraction_id, analysis_type
├── results, created_at
```

---

## 💻 Usage Examples

### Using Command Line Scripts

```bash
# Collect data for specific provinces
python scripts/collect_data.py --provinces "Bình Thuận,Đà Nẵng" --limit 5

# Collect for all attractions
python scripts/collect_data.py --all

# Check database statistics
python scripts/check_data.py

# Verify connection only
python scripts/check_data.py --verify
```

### Using Python API

```python
from app.collectors.data_pipeline import create_data_pipeline

# Initialize pipeline
pipeline = create_data_pipeline()

# Collect from all platforms for an attraction
await pipeline.collect_for_attraction(
    attraction_id=1,
    platform='google_maps',  # or 'facebook', 'youtube', 'tiktok'
    max_posts=8,
    max_comments=20
)
```

### API Endpoints

```bash
# List provinces
GET /api/v1/provinces

# List attractions by province
GET /api/v1/attractions?province_id=1

# Trigger collection
POST /api/v1/collection/collect
{
  "attraction_id": 1,
  "platforms": ["facebook", "google_maps"],
  "limit_per_platform": 50
}

# Get collection status
GET /api/v1/collection/status/{task_id}
```

---

## 🎯 Features

### ✅ Implemented
- [x] Multi-platform data collection (Facebook, Google Maps, TikTok, YouTube)
- [x] Dict mapping strategy for comment collection on existing posts
- [x] Automatic duplicate detection (unique constraints)
- [x] Rate limiting and delay management
- [x] Platform priority-based collection
- [x] Target-based stopping (40 comments per attraction)
- [x] Comprehensive logging and progress reporting
- [x] Database models with proper relationships
- [x] Pydantic schemas for validation
- [x] FastAPI REST API
- [x] Duplicate detection (UniqueConstraint)
- [x] Automated scheduling support
- [x] Comprehensive documentation

### 🔜 Planned Improvements

**Collector Upgrades:**
- [ ] Facebook: Direct page URLs only (keyword search blocked)
- [ ] Google Maps: Increase coverage to 20-30 places
- [ ] TikTok: Fix comment collection (currently 0 comments)
- [ ] YouTube: Complete testing and optimization
- [ ] Add best page fallback strategies
- [ ] Implement scraping multiple related pages

**Data Quality:**
- [ ] NLP-based relevance filtering
- [ ] Sentiment analysis integration
- [ ] Spam/bot detection
- [ ] Comment length filtering
- [ ] Duplicate content detection across platforms

**Analytics:**
- [ ] PhoBERT integration for Vietnamese NLP
- [ ] Web dashboard for visualization
- [ ] Real-time monitoring
- [ ] Automated report generation
- [ ] Trend analysis

---

## 📊 Current Performance

**Latest Collection Results:**
- **Attractions processed:** 7/9 (2 duplicates in DB)
- **Total posts:** 54
- **Total comments:** 374
- **Average comments/attraction:** 53.4
- **Target achievement:** 7/7 attractions ≥30 comments ✅

**Platform Performance:**
- **Google Maps:** Excellent (165 comments from 13 places for one attraction)
- **Facebook:** Very good (60+ comments with Best Pages strategy)
- **TikTok:** Posts only (0 comments - needs fixing)
- **YouTube:** Not yet tested in production

---

## 🔧 Configuration

### Environment Variables

```env
# Application
DEBUG=True
HOST=0.0.0.0
PORT=8000

# Database
DATABASE_URL=postgresql://user:pass@localhost/tourism_db

# APIs (all required)
YOUTUBE_API_KEY=your_key
GOOGLE_MAPS_API_KEY=your_key
APIFY_API_TOKEN=your_token

# Scheduler (optional)
SCHEDULER_ENABLED=False
DAILY_COLLECTION_HOUR=2
DAILY_COLLECTION_MINUTE=0

# Collection Limits
DEFAULT_POSTS_LIMIT=50
DEFAULT_COMMENTS_LIMIT=100
```

---

## 🧪 Testing

### Run Credential Tests
```bash
python test/test_credentials.py
```

### Run Unit Tests (when implemented)
```bash
pytest
```

### Manual Testing
```bash
# Test a specific collector
python examples/collection_demo.py
```

---

## 🐛 Troubleshooting

### Import Errors
```bash
# If you see "Import apify_client could not be resolved"
pip install apify-client
```

### API Errors
- **YouTube quota exceeded**: Wait 24 hours or use different API key
- **Google Maps billing**: Enable billing in Google Cloud Console
- **Apify insufficient credit**: Add payment method in Apify Console

### Database Issues
```bash
# Reset database (CAUTION: deletes all data)
rm tourism.db
python -c "from app.database.connection import init_db; init_db()"
```

---

## 🤝 Contributing

Contributions welcome! Please:
1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests
5. Submit a pull request

---

## 📄 License

[Add your license here]

---

## 🙏 Acknowledgments

- **Apify** - For reliable web scraping platform
- **Google** - For YouTube and Maps APIs
- **FastAPI** - For excellent web framework
- **SQLAlchemy** - For powerful ORM
