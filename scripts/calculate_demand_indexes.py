import asyncio
import sys
from pathlib import Path
from datetime import datetime, timedelta, timezone

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from app.database.connection import get_db
from app.services.demand_service import DemandIndexService


def calculate_weekly_demand():
    db = next(get_db())
    
    try:
        period_end = datetime.now(timezone.utc)
        period_start = period_end - timedelta(days=7)
        
        print(f"Calculating demand indexes for week: {period_start.date()} to {period_end.date()}")
        
        result = DemandIndexService.calculate_and_store_all_indexes(
            db, period_start, period_end, "week"
        )
        
        print(f"Success! Calculated {result['attractions']} attractions and {result['provinces']} provinces")
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        db.close()


def calculate_monthly_demand():
    db = next(get_db())
    
    try:
        period_end = datetime.now(timezone.utc)
        period_start = period_end - timedelta(days=30)
        
        print(f"Calculating demand indexes for month: {period_start.date()} to {period_end.date()}")
        
        result = DemandIndexService.calculate_and_store_all_indexes(
            db, period_start, period_end, "month"
        )
        
        print(f"Success! Calculated {result['attractions']} attractions and {result['provinces']} provinces")
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        db.close()


def calculate_quarterly_demand():
    db = next(get_db())
    
    try:
        period_end = datetime.now(timezone.utc)
        period_start = period_end - timedelta(days=90)
        
        print(f"Calculating demand indexes for quarter: {period_start.date()} to {period_end.date()}")
        
        result = DemandIndexService.calculate_and_store_all_indexes(
            db, period_start, period_end, "quarter"
        )
        
        print(f"Success! Calculated {result['attractions']} attractions and {result['provinces']} provinces")
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        db.close()


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Calculate tourism demand indexes")
    parser.add_argument(
        "--period",
        choices=["week", "month", "quarter", "all"],
        default="week",
        help="Period to calculate (default: week)"
    )
    
    args = parser.parse_args()
    
    print("="*80)
    print("TOURISM DEMAND INDEX CALCULATION")
    print("="*80)
    
    if args.period == "week":
        calculate_weekly_demand()
    elif args.period == "month":
        calculate_monthly_demand()
    elif args.period == "quarter":
        calculate_quarterly_demand()
    elif args.period == "all":
        print("\nCalculating all periods...\n")
        calculate_weekly_demand()
        print("\n" + "-"*80 + "\n")
        calculate_monthly_demand()
        print("\n" + "-"*80 + "\n")
        calculate_quarterly_demand()
    
    print("\n" + "="*80)
    print("COMPLETED")
    print("="*80)
