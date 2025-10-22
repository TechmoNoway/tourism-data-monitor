import os
from dotenv import load_dotenv

load_dotenv()

def test_youtube():
    """Test YouTube Data API v3"""
    print("\nTesting YouTube Data API...")
    
    api_key = os.getenv('YOUTUBE_API_KEY')
    if not api_key:
        print("YOUTUBE_API_KEY not found in .env file")
        return False
    
    try:
        from googleapiclient.discovery import build
        
        youtube = build('youtube', 'v3', developerKey=api_key)

        response = youtube.search().list(
            q='travel',
            part='snippet',
            maxResults=1,
            type='video'
        ).execute()
        
        if response.get('items'):
            print("YouTube API is working!")
            print(f"Sample video: {response['items'][0]['snippet']['title']}")
            return True
        else:
            print("API working but no results returned")
            return True
            
    except ImportError:
        print("google-api-python-client not installed")
        print("Run: pip install google-api-python-client")
        return False
    except Exception as e:
        print(f"YouTube API error: {str(e)}")
        return False


def test_google_maps():
    """Test Google Maps/Places API"""
    print("\n🗺️  Testing Google Maps API...")
    
    api_key = os.getenv('GOOGLE_MAPS_API_KEY')
    if not api_key:
        print("GOOGLE_MAPS_API_KEY not found in .env file")
        return False
    
    try:
        import googlemaps
        
        gmaps = googlemaps.Client(key=api_key)
        result = gmaps.geocode('Hanoi, Vietnam')
        
        if result:
            print("Google Maps API is working!")
            print(f"Sample location: {result[0]['formatted_address']}")
            return True
        else:
            print("API working but no results returned")
            return True
            
    except ImportError:
        print("googlemaps not installed")
        print("Run: pip install googlemaps")
        return False
    except Exception as e:
        print(f"Google Maps API error: {str(e)}")
        if "REQUEST_DENIED" in str(e):
            print("Make sure billing is enabled in Google Cloud Console")
        return False


def test_tiktok():
    """Test TikTok API (Legacy - now using Apify)"""
    print("\nTesting TikTok API...")
    print("⚠️  TikTok API collector is disabled")
    print("Reason: Requires approval (2-4 weeks) and has limited access")
    print("Solution: Using Apify scraper instead (see test_apify below)")
    print("To re-enable: uncomment TikTok code in data_pipeline.py")
    return None  # Return None to indicate disabled


def _test_tiktok_backup():
    """Backup of original TikTok test function"""
    client_key = os.getenv('TIKTOK_CLIENT_KEY')
    client_secret = os.getenv('TIKTOK_CLIENT_SECRET')
    access_token = os.getenv('TIKTOK_ACCESS_TOKEN')
    
    if not client_key or not client_secret:
        print("TIKTOK_CLIENT_KEY or TIKTOK_CLIENT_SECRET not found in .env file")
        return False
    
    if not access_token:
        print("TIKTOK_ACCESS_TOKEN not found")
        print("TikTok API requires user authorization to get access token")
        return False
    
    try:
        import requests
        
        headers = {
            'Authorization': f'Bearer {access_token}',
            'Content-Type': 'application/json'
        }
        
        url = 'https://open.tiktokapis.com/v2/user/info/'
        params = {
            'fields': 'open_id,union_id,display_name'
        }
        response = requests.get(url, headers=headers, params=params)
        
        if response.status_code == 200:
            data = response.json()
            if data.get('data'):
                print("TikTok API is working!")
                print(f"Authenticated as: {data['data'].get('display_name', 'Unknown')}")
                return True
            else:
                print("API response received but no user data")
                return False
        else:
            print(f"TikTok API error: {response.status_code}")
            print(f"Response: {response.text[:200]}")
            return False
            
    except ImportError:
        print("requests library not installed")
        print("Run: pip install requests")
        return False
    except Exception as e:
        print(f"TikTok API error: {str(e)}")
        return False


# Facebook temporarily disabled due to API limitations
def test_facebook():
    print("\nTesting Facebook Graph API...")
    print("⚠️  Facebook Graph API collector is disabled")
    print("Reason: Graph API has strict limitations for public data access")
    print("Solution: Using Apify scraper instead (see test_apify below)")
    print("To re-enable: uncomment Facebook code in data_pipeline.py")
    return None  # Return None to indicate disabled


def _test_facebook_backup():
    """Backup of original Facebook test function"""
    print("\nTesting Facebook Graph API...")
    
    access_token = os.getenv('FACEBOOK_ACCESS_TOKEN')
    app_id = os.getenv('FACEBOOK_APP_ID')
    app_secret = os.getenv('FACEBOOK_APP_SECRET')
    
    if not access_token:
        print("FACEBOOK_ACCESS_TOKEN not found in .env file")
        return False
    
    if not app_id or not app_secret:
        print("FACEBOOK_APP_ID or FACEBOOK_APP_SECRET not found")
        print("These are optional but recommended")
    
    try:
        import requests
        
        url = f'https://graph.facebook.com/v18.0/me?access_token={access_token}'
        response = requests.get(url)
        
        if response.status_code == 200:
            data = response.json()
            print("Facebook API is working!")
            print(f"Authenticated as: {data.get('name', 'Unknown')}")
            print(f"User ID: {data.get('id', 'Unknown')}")
            
            debug_url = f'https://graph.facebook.com/v18.0/debug_token?input_token={access_token}&access_token={access_token}'
            debug_response = requests.get(debug_url)
            if debug_response.status_code == 200:
                debug_data = debug_response.json().get('data', {})
                if debug_data.get('expires_at'):
                    import datetime
                    expires = datetime.datetime.fromtimestamp(debug_data['expires_at'])
                    print(f"Token expires: {expires.strftime('%Y-%m-%d %H:%M:%S')}")
                else:
                    print("Token does not expire (Page Access Token)")
            
            return True
        elif response.status_code == 190:
            print("Access token is invalid or expired")
            print("Generate a new token from Graph API Explorer")
            return False
        else:
            print(f"Facebook API error: {response.json()}")
            return False
            
    except ImportError:
        print("requests library not installed")
        print("Run: pip install requests")
        return False
    except Exception as e:
        print(f"Facebook API error: {str(e)}")
        return False


def test_apify():
    """Test Apify API (for Facebook & TikTok scraping)"""
    print("\nTesting Apify API...")
    
    api_token = os.getenv('APIFY_API_TOKEN')
    if not api_token:
        print("❌ APIFY_API_TOKEN not found in .env file")
        print("Get your token from: https://console.apify.com/account/integrations")
        print("See docs/APIFY_SETUP.md for setup guide")
        return False
    
    try:
        from apify_client import ApifyClient
        
        client = ApifyClient(api_token)
        
        # Test authentication by getting user info
        user = client.user().get()
        
        if user:
            print("✅ Apify API is working!")
            print(f"Authenticated as: {user.get('username', 'Unknown')}")
            print(f"Plan: {user.get('plan', 'Unknown')}")
            
            # Check account usage
            usage = user.get('usage', {})
            if usage:
                print(f"Storage used: {usage.get('dataRetentionInDays', 'N/A')} days retention")
            
            return True
        else:
            print("❌ API response received but no user data")
            return False
            
    except ImportError:
        print("❌ apify-client not installed")
        print("Run: pip install apify-client")
        return False
    except Exception as e:
        error_msg = str(e)
        print(f"❌ Apify API error: {error_msg}")
        
        if "401" in error_msg or "Unauthorized" in error_msg:
            print("Token is invalid. Get a new one from Apify Console.")
        elif "403" in error_msg or "Forbidden" in error_msg:
            print("Token doesn't have necessary permissions.")
        
        return False


def test_all():
    """Run all credential tests"""
    print("=" * 60)
    print("Tourism Data Monitor - API Credentials Test")
    print("=" * 60)
    
    if not os.path.exists('.env'):
        print("\n❌ .env file not found!")
        print("Copy .env.example to .env and fill in your credentials:")
        print("cp .env.example .env  # Linux/Mac")
        print("copy .env.example .env  # Windows")
        return
    
    results = {
        'YouTube': test_youtube(),
        'Google Maps': test_google_maps(),
        'Apify (FB/TikTok)': test_apify(),
        'Facebook API': test_facebook(),
        'TikTok API': test_tiktok(),
    }
    
    print("\n" + "=" * 60)
    print("Test Summary")
    print("=" * 60)
    
    for platform, success in results.items():
        if success is None:
            status = "⚠️  DISABLED"
            icon = "⚠️ "
        elif success:
            status = "✅ WORKING"
            icon = "✅"
        else:
            status = "❌ FAILED"
            icon = "❌"
        print(f"{icon} {platform:20} {status}")
    
    active_results = {k: v for k, v in results.items() if v is not None}
    total = len(active_results)
    passed = sum(1 for v in active_results.values() if v)
    
    print(f"\n📊 Active platforms tested: {total}")
    print(f"✅ Platforms working: {passed}/{total}")
    
    if passed == total:
        print("\n🎉 All credentials are working! You're ready to start collecting data.")
        print("\n📚 Next steps:")
        print("   1. See docs/APIFY_SETUP.md for Apify setup")
        print("   2. Run examples/collection_usage_example.py to test collection")
        print("   3. Check docs/API_CREDENTIALS_SETUP.md for other APIs")
    elif passed > 0:
        print("\n⚠️  Some credentials are missing or invalid. Check the errors above.")
        print("\n📚 Setup guides:")
        print("   - YouTube: docs/API_CREDENTIALS_SETUP.md")
        print("   - Google Maps: docs/API_CREDENTIALS_SETUP.md")
        print("   - Apify: docs/APIFY_SETUP.md")
    else:
        print("\nNo valid credentials found. Please configure your .env file.")
    
    print("\nFor setup instructions, see: docs/API_CREDENTIALS_SETUP.md")
    print("=" * 60)


if __name__ == "__main__":
    test_all()
