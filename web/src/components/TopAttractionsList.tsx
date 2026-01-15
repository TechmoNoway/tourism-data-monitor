import React from 'react';
import { Link } from 'react-router-dom';
import { TopAttraction, TopProvince } from '../services/api';

interface TopAttractionsListProps {
  type: 'attractions' | 'provinces';
  items: TopAttraction[] | TopProvince[];
  period: string;
  onPeriodChange?: (period: string) => void;
}

const TopAttractionsList: React.FC<TopAttractionsListProps> = ({
  type,
  items = [],
  period,
  onPeriodChange,
}) => {
  const getRankColor = (rank: number) => {
    if (rank === 1) return 'bg-yellow-100 text-yellow-800';
    if (rank === 2) return 'bg-gray-100 text-gray-800';
    if (rank === 3) return 'bg-orange-100 text-orange-800';
    return 'bg-blue-50 text-blue-800';
  };

  const getIndexColor = (score: number) => {
    if (score >= 70) return 'text-green-600';
    if (score >= 50) return 'text-yellow-600';
    if (score >= 30) return 'text-orange-600';
    return 'text-red-600';
  };

  return (
    <div className="bg-white rounded-lg shadow-md p-6">
      <div className="flex items-center justify-between mb-4">
        <h3 className="text-lg font-semibold text-gray-800">
          {type === 'attractions' ? 'Điểm Du Lịch Nổi Bật' : 'Tỉnh Thành Hàng Đầu'}
        </h3>
        
        {onPeriodChange && (
          <div className="flex gap-2">
            <button
              onClick={() => onPeriodChange('week')}
              className={`px-3 py-1 text-xs rounded ${
                period === 'week'
                  ? 'bg-blue-600 text-white'
                  : 'bg-gray-100 text-gray-600 hover:bg-gray-200'
              }`}
            >
              7 ngày
            </button>
            <button
              onClick={() => onPeriodChange('month')}
              className={`px-3 py-1 text-xs rounded ${
                period === 'month'
                  ? 'bg-blue-600 text-white'
                  : 'bg-gray-100 text-gray-600 hover:bg-gray-200'
              }`}
            >
              30 ngày
            </button>
            <button
              onClick={() => onPeriodChange('quarter')}
              className={`px-3 py-1 text-xs rounded ${
                period === 'quarter'
                  ? 'bg-blue-600 text-white'
                  : 'bg-gray-100 text-gray-600 hover:bg-gray-200'
              }`}
            >
              90 ngày
            </button>
          </div>
        )}
      </div>

      <div className="space-y-3">
        {!items || items.length === 0 ? (
          <div className="text-center py-8 text-gray-500">
            Chưa có dữ liệu cho khoảng thời gian này
          </div>
        ) : (
          items.map((item) => {
            if (type === 'attractions') {
              const attraction = item as TopAttraction;
              return (
                <Link
                  key={attraction.id}
                  to={`/attractions/${attraction.id}`}
                  className="block hover:bg-gray-50 p-3 rounded-lg transition-colors"
                >
                  <div className="flex items-center gap-3">
                    <div
                      className={`flex-shrink-0 w-8 h-8 rounded-full flex items-center justify-center font-bold text-sm ${getRankColor(
                        attraction.rank
                      )}`}
                    >
                      {attraction.rank}
                    </div>
                    
                    <div className="flex-1 min-w-0">
                      <h4 className="font-medium text-gray-900 truncate">
                        {attraction.name}
                      </h4>
                      <div className="flex items-center gap-2 text-sm text-gray-500">
                        <span>{attraction.province_name}</span>
                        <span>•</span>
                        <span className="text-xs">
                          {attraction.total_comments} bình luận
                        </span>
                      </div>
                    </div>

                    <div className="text-right">
                      <div className={`text-lg font-semibold ${getIndexColor(attraction.overall_index)}`}>
                        {attraction.overall_index.toFixed(1)}
                      </div>
                      <div className="text-xs text-gray-500">điểm</div>
                    </div>
                  </div>
                </Link>
              );
            } else {
              const province = item as TopProvince;
              return (
                <Link
                  key={province.id}
                  to={`/provinces/${province.id}`}
                  className="block hover:bg-gray-50 p-3 rounded-lg transition-colors"
                >
                  <div className="flex items-center gap-3">
                    <div
                      className={`flex-shrink-0 w-8 h-8 rounded-full flex items-center justify-center font-bold text-sm ${getRankColor(
                        province.rank
                      )}`}
                    >
                      {province.rank}
                    </div>
                    
                    <div className="flex-1 min-w-0">
                      <h4 className="font-medium text-gray-900">
                        {province.name}
                      </h4>
                      <div className="text-sm text-gray-500">
                        {province.active_attractions} điểm du lịch
                      </div>
                    </div>

                    <div className="text-right">
                      <div className={`text-lg font-semibold ${getIndexColor(province.overall_index)}`}>
                        {province.overall_index.toFixed(1)}
                      </div>
                      <div className="text-xs text-gray-500">điểm</div>
                    </div>
                  </div>
                </Link>
              );
            }
          })
        )}
      </div>
    </div>
  );
};

export default TopAttractionsList;
