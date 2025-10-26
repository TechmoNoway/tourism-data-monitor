"""
Test all collectors after refactoring
Verify that renamed files and updated imports work correctly
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.core.config import settings
from app.collectors.data_pipeline import create_data_pipeline

def test_collectors():
    """Test that all collectors initialize correctly"""
    
    print("\n" + "="*70)
    print("🧪 TESTING COLLECTORS AFTER REFACTOR")
    print("="*70 + "\n")
    
    # Check tokens
    print("📋 Checking API tokens...")
    has_apify = bool(settings.APIFY_API_TOKEN)
    has_youtube = bool(settings.YOUTUBE_API_KEY)
    
    print(f"   Apify token: {'✓' if has_apify else '✗'}")
    print(f"   YouTube API key: {'✓' if has_youtube else '✗'}")
    
    if not has_apify and not has_youtube:
        print("\n❌ No API tokens found!")
        print("💡 Add APIFY_API_TOKEN and/or YOUTUBE_API_KEY to .env")
        return
    
    print("\n🔧 Initializing DataPipeline...")
    
    try:
        # Create pipeline with simplified parameters
        pipeline = create_data_pipeline(
            apify_api_token=settings.APIFY_API_TOKEN,
            youtube_api_key=settings.YOUTUBE_API_KEY
        )
        
        print("✅ DataPipeline created successfully!\n")
        
        # Check which collectors were initialized
        print("📊 Initialized collectors:")
        for platform, collector in pipeline.collectors.items():
            collector_name = type(collector).__name__
            print(f"   ✓ {platform}: {collector_name}")
        
        if not pipeline.collectors:
            print("   ⚠️  No collectors initialized")
        
        # Expected collectors
        expected = []
        if has_apify:
            expected.extend(['facebook', 'tiktok', 'google_maps'])
        if has_youtube:
            expected.append('youtube')
        
        print(f"\n✅ Expected collectors: {', '.join(expected)}")
        print(f"✅ Actual collectors: {', '.join(pipeline.collectors.keys())}")
        
        # Check if all expected are present
        missing = set(expected) - set(pipeline.collectors.keys())
        if missing:
            print(f"\n⚠️  Missing collectors: {', '.join(missing)}")
        else:
            print(f"\n🎉 All expected collectors initialized successfully!")
        
        return True
        
    except Exception as e:
        print(f"\n❌ Error initializing pipeline:")
        print(f"   {str(e)}")
        import traceback
        traceback.print_exc()
        return False
    
    finally:
        print("\n" + "="*70 + "\n")

if __name__ == "__main__":
    success = test_collectors()
    sys.exit(0 if success else 1)
