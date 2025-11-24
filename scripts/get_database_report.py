"""Generate database schema and statistics report"""
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from app.database.connection import SessionLocal
from sqlalchemy import text, inspect

def main():
    db = SessionLocal()
    try:
        inspector = inspect(db.bind)
        tables = inspector.get_table_names()
        
        print("\n" + "="*80)
        print("DATABASE SCHEMA REPORT")
        print("="*80)
        
        total_rows = 0
        total_columns = 0
        
        for table in tables:
            columns = inspector.get_columns(table)
            result = db.execute(text(f"SELECT COUNT(*) FROM {table}"))
            row_count = result.fetchone()[0]
            
            total_rows += row_count
            total_columns += len(columns)
            
            print(f"\n{table.upper()}")
            print("-" * 80)
            print(f"Columns: {len(columns)}")
            print(f"Rows: {row_count:,}")
            print("\nColumn Details:")
            
            for col in columns:
                nullable = "NULL" if col.get('nullable', True) else "NOT NULL"
                default = f" DEFAULT {col.get('default', '')}" if col.get('default') else ""
                print(f"  {col['name']:30s} {str(col['type']):20s} {nullable}{default}")
        
        print("\n" + "="*80)
        print("SUMMARY")
        print("="*80)
        print(f"Total Tables: {len(tables)}")
        print(f"Total Columns: {total_columns}")
        print(f"Total Rows: {total_rows:,}")
        print("="*80)
        
    finally:
        db.close()

if __name__ == "__main__":
    main()
