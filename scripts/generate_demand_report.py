"""
Generate Tourism Demand Analysis Report
Comprehensive analysis of tourism demand based on database statistics
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from app.database.connection import get_db
from app.models.tourist_attraction import TouristAttraction
from app.models.province import Province
from sqlalchemy import func

def analyze_tourism_types():
    """Analyze tourism type distribution"""
    
    db = next(get_db())
    
    try:
        print("="*70)
        print("1. TOURISM TYPE ANALYSIS (Loại Hình Du Lịch)")
        print("="*70 + "\n")
        
        # Get statistics by tourism type
        results = db.query(
            TouristAttraction.tourism_type,
            func.count(TouristAttraction.id).label('attraction_count'),
            func.sum(TouristAttraction.total_comments).label('total_comments'),
            func.sum(TouristAttraction.total_reviews).label('total_posts')
        ).filter(
            TouristAttraction.tourism_type.isnot(None)
        ).group_by(
            TouristAttraction.tourism_type
        ).order_by(
            func.sum(TouristAttraction.total_comments).desc()
        ).all()
        
        total_attractions = sum(r.attraction_count for r in results)
        total_comments = sum(r.total_comments or 0 for r in results)
        
        type_names = {
            'beach': 'Du lịch Biển',
            'mountain': 'Du lịch Núi/Cao nguyên',
            'historical': 'Di tích Lịch sử',
            'cultural': 'Văn hóa/Tâm linh',
            'nature': 'Sinh thái/Thiên nhiên',
            'urban': 'Du lịch Đô thị',
            'adventure': 'Thể thao/Mạo hiểm'
        }
        
        for i, result in enumerate(results, 1):
            type_name = type_names.get(result.tourism_type, result.tourism_type)
            attraction_pct = (result.attraction_count / total_attractions * 100) if total_attractions > 0 else 0
            comment_pct = (result.total_comments / total_comments * 100) if total_comments > 0 and result.total_comments else 0
            
            stars = "⭐" * min(5, int(comment_pct / 10))
            
            print(f"{i}. {type_name:25s}")
            print(f"   Attractions: {result.attraction_count:3d} ({attraction_pct:5.1f}%)")
            print(f"   Posts:       {result.total_posts:5d}")
            print(f"   Comments:    {result.total_comments:5d} ({comment_pct:5.1f}%) {stars}")
            print()
        
        print(f"Total: {total_attractions} attractions, {total_comments} comments\n")
        
    finally:
        db.close()

def analyze_provinces():
    """Analyze province popularity"""
    
    db = next(get_db())
    
    try:
        print("="*70)
        print("2. PROVINCE POPULARITY ANALYSIS (Xu Hướng Chọn Tỉnh)")
        print("="*70 + "\n")
        
        provinces = db.query(Province).order_by(Province.total_comments.desc()).all()
        
        total_comments = sum(p.total_comments for p in provinces)
        
        for i, province in enumerate(provinces, 1):
            comment_pct = (province.total_comments / total_comments * 100) if total_comments > 0 else 0
            avg_per_attraction = province.total_comments / province.total_attractions if province.total_attractions > 0 else 0
            
            # Popularity score (0-100)
            popularity_score = min(100, int(comment_pct * 3))  # Scale up for visibility
            
            medal = {1: "🥇", 2: "🥈", 3: "🥉"}.get(i, "  ")
            
            print(f"{medal} {province.name:20s} - Popularity Score: {popularity_score}/100")
            print(f"   Attractions:     {province.total_attractions:3d}")
            print(f"   Total Comments:  {province.total_comments:5d} ({comment_pct:5.1f}%)")
            print(f"   Avg/Attraction:  {avg_per_attraction:6.1f}")
            print()
        
    finally:
        db.close()

def analyze_province_tourism_types():
    """Analyze tourism types by province"""
    
    db = next(get_db())
    
    try:
        print("="*70)
        print("3. TOURISM TYPES BY PROVINCE")
        print("="*70 + "\n")
        
        provinces = db.query(Province).order_by(Province.name).all()
        
        type_names = {
            'beach': 'Biển',
            'mountain': 'Núi',
            'historical': 'Lịch sử',
            'cultural': 'Văn hóa',
            'nature': 'Sinh thái',
            'urban': 'Đô thị',
            'adventure': 'Mạo hiểm'
        }
        
        for province in provinces:
            print(f"📍 {province.name}")
            print(f"   Total: {province.total_attractions} attractions, {province.total_comments} comments\n")
            
            # Get tourism type breakdown
            results = db.query(
                TouristAttraction.tourism_type,
                func.count(TouristAttraction.id).label('count'),
                func.sum(TouristAttraction.total_comments).label('comments')
            ).filter(
                TouristAttraction.province_id == province.id,
                TouristAttraction.tourism_type.isnot(None)
            ).group_by(
                TouristAttraction.tourism_type
            ).order_by(
                func.count(TouristAttraction.id).desc()
            ).all()
            
            if results:
                for result in results:
                    type_name = type_names.get(result.tourism_type, result.tourism_type)
                    print(f"   {type_name:12s}: {result.count:2d} attractions, {result.comments:5d} comments")
            else:
                print("   (No data)")
            
            print()
        
    finally:
        db.close()

def analyze_top_attractions():
    """Analyze top attractions by comments"""
    
    db = next(get_db())
    
    try:
        print("="*70)
        print("4. TOP ATTRACTIONS BY POPULARITY (Comments)")
        print("="*70 + "\n")
        
        attractions = db.query(
            TouristAttraction,
            Province
        ).join(
            Province, TouristAttraction.province_id == Province.id
        ).filter(
            TouristAttraction.total_comments > 0
        ).order_by(
            TouristAttraction.total_comments.desc()
        ).limit(20).all()
        
        type_names = {
            'beach': '🏖️ Biển',
            'mountain': '⛰️ Núi',
            'historical': '🏛️ Lịch sử',
            'cultural': '🏯 Văn hóa',
            'nature': '🌳 Sinh thái',
            'urban': '🏙️ Đô thị',
            'adventure': '🚵 Mạo hiểm',
            None: '❓ Unknown'
        }
        
        for i, (attraction, province) in enumerate(attractions, 1):
            type_icon = type_names.get(attraction.tourism_type, '❓ Unknown')
            
            print(f"{i:2d}. {attraction.name[:45]:45s}")
            print(f"    Province: {province.name:20s} Type: {type_icon}")
            print(f"    Comments: {attraction.total_comments:5d}   Posts: {attraction.total_reviews:4d}")
            print()
        
    finally:
        db.close()

def generate_summary():
    """Generate executive summary"""
    
    db = next(get_db())
    
    try:
        print("="*70)
        print("📊 EXECUTIVE SUMMARY - NHU CẦU DU LỊCH")
        print("="*70 + "\n")
        
        # Total statistics
        total_attractions = db.query(func.count(TouristAttraction.id)).scalar()
        total_comments = db.query(func.sum(TouristAttraction.total_comments)).scalar() or 0
        total_posts = db.query(func.sum(TouristAttraction.total_reviews)).scalar() or 0
        
        print(f"📍 Database Coverage:")
        print(f"   - Total Attractions: {total_attractions}")
        print(f"   - Total Posts:       {total_posts}")
        print(f"   - Total Comments:    {total_comments}")
        print()
        
        # Top tourism type
        top_type = db.query(
            TouristAttraction.tourism_type,
            func.sum(TouristAttraction.total_comments).label('comments')
        ).filter(
            TouristAttraction.tourism_type.isnot(None)
        ).group_by(
            TouristAttraction.tourism_type
        ).order_by(
            func.sum(TouristAttraction.total_comments).desc()
        ).first()
        
        # Top province
        top_province = db.query(Province).order_by(Province.total_comments.desc()).first()
        
        # Top attraction
        top_attraction = db.query(
            TouristAttraction, Province
        ).join(
            Province
        ).order_by(
            TouristAttraction.total_comments.desc()
        ).first()
        
        print(f"🏆 Top Performers:")
        if top_type:
            print(f"   - Most Popular Type: {top_type.tourism_type} ({top_type.comments} comments)")
        if top_province:
            print(f"   - Most Popular Province: {top_province.name} ({top_province.total_comments} comments)")
        if top_attraction:
            print(f"   - Most Popular Attraction: {top_attraction[0].name} ({top_attraction[0].total_comments} comments)")
        print()
        
        print(f"💡 Key Insights:")
        print(f"   - Avg comments/attraction: {total_comments/total_attractions:.1f}")
        print(f"   - Avg comments/province: {total_comments/4:.1f}")
        
    finally:
        db.close()

if __name__ == "__main__":
    print("\n")
    print("="*70)
    print("TOURISM DEMAND ANALYSIS REPORT")
    print("Hệ Thống Phân Tích Nhu Cầu Du Lịch")
    print("="*70)
    print("\n")
    
    generate_summary()
    print("\n")
    
    analyze_tourism_types()
    print("\n")
    
    analyze_provinces()
    print("\n")
    
    analyze_province_tourism_types()
    print("\n")
    
    analyze_top_attractions()
    
    print("\n")
    print("="*70)
    print("✅ REPORT COMPLETED")
    print("="*70)
    print("\n")
