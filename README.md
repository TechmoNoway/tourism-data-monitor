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
- PostgreSQL or SQLite
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
⚠️  Facebook API          DISABLED (using Apify)
⚠️  TikTok API            DISABLED (using Apify)

🎉 All credentials are working!
```

### 5. Run the Application

```bash
# Initialize database
python -c "from app.database.connection import init_db; init_db()"

# Start API server
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
│   ├── collectors/       # Data collection modules
│   │   ├── facebook_apify_collector.py
│   │   ├── tiktok_apify_collector.py
│   │   ├── youtube_collector.py
│   │   └── google_reviews_collector.py
│   ├── models/           # Database models
│   ├── schemas/          # Pydantic schemas
│   ├── services/         # Business logic
│   └── core/             # Configuration
├── docs/                 # Documentation
├── test/                 # Tests
├── examples/             # Usage examples
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

### Collect Data for an Attraction

```python
from app.collectors.data_pipeline import DataCollectionPipeline
from app.core.config import settings

# Initialize pipeline
pipeline = DataCollectionPipeline(
    youtube_api_key=settings.YOUTUBE_API_KEY,
    google_maps_api_key=settings.GOOGLE_MAPS_API_KEY,
    apify_api_token=settings.APIFY_API_TOKEN
)

# Collect from all platforms
result = await pipeline.collect_for_attraction(
    attraction_id=1,
    platforms=['youtube', 'google_reviews', 'facebook', 'tiktok'],
    limit_per_platform=50
)

print(f"Collected {result['total_posts']} posts")
print(f"Collected {result['total_comments']} comments")
```

### API Endpoints

```bash
# List provinces
GET /api/v1/provinces

# List attractions
GET /api/v1/attractions?province_id=1

# Collect data for attraction
POST /api/v1/collection/collect
{
  "attraction_id": 1,
  "platforms": ["youtube", "facebook"],
  "limit_per_platform": 50
}

# Get collection status
GET /api/v1/collection/status/{task_id}
```

See [examples/collection_usage_example.py](examples/collection_usage_example.py) for more.

---

## 🎯 Features

### ✅ Implemented
- [x] Multi-platform data collection (4 platforms)
- [x] Database models with proper relationships
- [x] Pydantic schemas for validation
- [x] FastAPI REST API
- [x] Duplicate detection (UniqueConstraint)
- [x] Automated scheduling support
- [x] Comprehensive documentation

### 🔜 Planned
- [ ] NLP analysis (sentiment, bot detection)
- [ ] PhoBERT integration
- [ ] Web dashboard
- [ ] Real-time monitoring
- [ ] Advanced analytics
- [ ] Report generation

---

## 💰 Cost Estimate

### Free Tier (Good for testing)
- YouTube: 10,000 quota units/day (free)
- Google Maps: $200 credit/month
- Apify: $5 free credit

### Monthly Production Costs
- **Small** (3 provinces, weekly): ~$10-15/month
- **Medium** (3 provinces, daily): ~$25-30/month
- **Large** (10 provinces, daily): ~$80-100/month

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

## 📞 Support

- **Documentation**: Check `docs/` folder
- **API Docs**: http://localhost:8000/docs (when server running)
- **Apify Support**: support@apify.com
- **Issues**: Create GitHub issue (if repo exists)

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

---

## 📈 Roadmap

### Phase 1: Data Collection ✅ (COMPLETE)
- [x] YouTube collector
- [x] Google Reviews collector
- [x] Facebook collector (Apify)
- [x] TikTok collector (Apify)
- [x] Data pipeline
- [x] Scheduler

### Phase 2: NLP Analysis 🔄 (IN PROGRESS)
- [ ] PhoBERT sentiment analysis
- [ ] Bot detection
- [ ] Topic modeling
- [ ] Trend analysis

### Phase 3: Visualization 📋 (PLANNED)
- [ ] Web dashboard
- [ ] Real-time charts
- [ ] Report generation
- [ ] Export to Excel/PDF

### Phase 4: Advanced Features 📋 (PLANNED)
- [ ] Alert system
- [ ] Competitor analysis
- [ ] Recommendation engine
- [ ] API rate limit optimization

---

**Built with ❤️ for Vietnamese Tourism**

**Start collecting data in 5 minutes**: [docs/APIFY_QUICKSTART.md](docs/APIFY_QUICKSTART.md)