import sys
import os
from pathlib import Path
import asyncio
from datetime import datetime

if sys.platform == 'win32':
    os.environ['PYTHONIOENCODING'] = 'utf-8'

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


def print_header():
    print("\n" + "="*80)
    print("TOURISM DATA COLLECTOR SERVICE")
    print("="*80)
    print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Python: {sys.version.split()[0]}")
    print(f"Working Directory: {os.getcwd()}")
    print("="*80 + "\n")


async def run_collection():
    try:
        from scripts.collect_data_comprehensive import ComprehensiveCollector
        
        print("[START] Initializing collector...")
        
        collector = ComprehensiveCollector(
            full_mode=False,
            use_gpu=False,
            batch_size=16
        )
        
        print("[OK] Collector initialized")
        print("\n" + "-"*80)
        print("COLLECTION PHASE")
        print("-"*80 + "\n")
        
        stats = await collector.collect_for_provinces(
            province_names=None,
            limit_per_province=None,
            all_attractions=True
        )
        
        print("\n" + "-"*80)
        print("COLLECTION SUMMARY")
        print("-"*80)
        print(f"Attractions processed: {stats.get('attractions', 0)}")
        print(f"Posts collected: {stats.get('posts_collected', 0)}")
        print(f"Comments collected: {stats.get('comments_collected', 0)}")
        print("-"*80 + "\n")
        
        return stats
        
    except Exception as e:
        print(f"\n[ERROR] Collection failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return None


def main():
    print_header()
    
    start_time = datetime.now()
    
    try:
        stats = asyncio.run(run_collection())
        
        if stats:
            print("\n" + "="*80)
            print("[SUCCESS] Collection completed successfully")
            print("="*80 + "\n")
            return 0
        else:
            print("\n" + "="*80)
            print("[FAILED] Collection failed - see errors above")
            print("="*80 + "\n")
            return 1
            
    except KeyboardInterrupt:
        print("\n\n[STOPPED] Collection interrupted by user")
        return 2
        
    except Exception as e:
        print(f"\n[FATAL] Unexpected error: {str(e)}")
        import traceback
        traceback.print_exc()
        return 1
        
    finally:
        end_time = datetime.now()
        duration = end_time - start_time
        
        print("\n" + "="*80)
        print(f"Total execution time: {duration}")
        print(f"Finished at: {end_time.strftime('%Y-%m-%d %H:%M:%S')}")
        print("="*80 + "\n")


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
