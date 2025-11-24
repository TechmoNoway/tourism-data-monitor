# QUICKSTART - Tourism Data Monitor

Quick guide to get started with data collection in 5 minutes.

## 🚀 Setup (One-time)

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Configure API Keys
Edit `.env` file:
```env
YOUTUBE_API_KEY=your_key_here
GOOGLE_MAPS_API_KEY=your_key_here
APIFY_API_TOKEN=apify_api_your_token_here
DATABASE_URL=sqlite:///./tourism.db
```

### 3. Initialize Database
```bash
python scripts/recreate_db.py
```

✅ You're ready to collect data!

---

## 📥 Collect Data

### Option 1: Quick Test (Recommended for first run)
Collect for 3 provinces, 3 attractions each:
```bash
python scripts/collect_data.py --provinces "Bình Thuận,Đà Nẵng,Lâm Đồng" --limit 3
```

**Expected time:** 15-20 minutes  
**Expected result:** ~350-400 comments across 9 attractions

### Option 2: Single Province
```bash
python scripts/collect_data.py --provinces "Bình Thuận" --limit 5
```

### Option 3: All Attractions
```bash
python scripts/collect_data.py --all
```

⚠️ **Warning:** This will process ALL active attractions in database. Could take hours!

---

## 📊 Check Results

### View Statistics
```bash
python scripts/check_data.py
```

**Output example:**
```
📊 DATABASE STATISTICS
======================================================================

📈 TOTALS:
  Provinces: 3
  Attractions: 9
  Posts: 54
  Comments: 374

🏛️  BREAKDOWN BY ATTRACTION:
  Bãi biển Mũi Né (Bình Thuận): 23 posts, 165 comments
  Đồi cát bay Mũi Né (Bình Thuận): 1 posts, 12 comments
  Bà Nà Hills (Đà Nẵng): 13 posts, 63 comments
  ...

📊 AVERAGES:
  Posts per attraction: 6.0
  Comments per attraction: 41.6
  Comments per post: 6.9
```

### Quick Verify
```bash
python scripts/check_data.py --verify
```

---

## 🎯 What Gets Collected?

### Platforms (in priority order)
1. **Facebook** - Posts + Comments from tourism pages
2. **Google Maps** - Place reviews (13+ places per attraction)
3. **YouTube** - Videos + Comments
4. **TikTok** - Videos (limited comment support)

### Collection Strategy
- **Target:** 40 comments per attraction
- **Stops when:** 80% of target reached (32 comments)
- **Posts per attraction:** 8
- **Rate limiting:** 5 seconds between attractions

### What You'll Get
For each attraction:
- 5-23 social posts (varies by platform availability)
- 30-165 comments (target: 40, often exceeds)
- Mix of platforms (Google Maps typically provides most comments)

---

## 🔧 Common Issues

### "No data collected"
**Cause:** API keys not configured or invalid  
**Fix:** Run `python test/test_credentials.py` to verify API keys

### "Facebook keyword search blocked"
**Known issue:** Facebook blocks keyword search. System uses "Best Pages" strategy instead.  
**Status:** Working with direct page URLs

### "TikTok 0 comments"
**Known issue:** TikTok collector gets videos but no comments  
**Status:** Under investigation (see TODO list)

### "Rate limit exceeded"
**Cause:** Too many API calls  
**Fix:** Wait 1 hour and try again, or reduce `--limit` parameter

---

## 📚 Next Steps

### Run API Server
```bash
python run.py
```
Visit: http://localhost:8000/docs

### View Documentation
- Full README: `README.md`
- Scripts guide: `scripts/README.md`
- API setup: `docs/APIFY_QUICKSTART.md`

### Customize Collection
Edit `scripts/collect_data.py`:
- Change target comments: `TARGET_COMMENTS_PER_ATTRACTION`
- Change platform priority: `PLATFORMS_PRIORITY`
- Adjust rate limiting: `await asyncio.sleep(5)`

---

## 💡 Tips

**For Best Results:**
1. Start with `--limit 3` to test
2. Check `scripts/check_data.py` after each run
3. Google Maps provides highest quality reviews
4. Facebook works best with direct page URLs
5. Allow 5-10 minutes per attraction for collection

**Cost Optimization:**
- Use `--limit` to control scope
- Apify charges per API call (monitor at https://console.apify.com)
- YouTube & Google Maps have free tiers
- Expected cost: $0.50-$2.00 per full collection run

**Troubleshooting:**
1. Always check credentials first: `python test/test_credentials.py`
2. Verify database: `python scripts/check_data.py --verify`
3. Check logs in terminal output
4. See TODO list in VS Code for known issues

---

## 🎉 Success Indicators

After running `scripts/collect_data.py`, you should see:

```
✅ COMPLETED: Bãi biển Mũi Né
   Final: 23 posts, 165 comments
   New: +0 posts, +0 comments

📊 TOTALS:
   Attractions processed: 9
   New posts collected: 44
   New comments collected: 349
   Average comments/attraction: 38.8
```

✅ **You're all set!** Data is ready for analysis.
