import React from 'react';
import { DemandIndex, ProvinceDemandIndex } from '../services/api';

interface DemandIndexCardProps {
  demandIndex: DemandIndex | ProvinceDemandIndex;
  type: 'attraction' | 'province';
}

const DemandIndexCard: React.FC<DemandIndexCardProps> = ({ demandIndex, type }) => {
  const getIndexColor = (score: number) => {
    if (score >= 70) return 'text-green-600 bg-green-50';
    if (score >= 50) return 'text-yellow-600 bg-yellow-50';
    if (score >= 30) return 'text-orange-600 bg-orange-50';
    return 'text-red-600 bg-red-50';
  };

  const getScoreColor = (score: number) => {
    if (score >= 70) return 'text-green-600';
    if (score >= 50) return 'text-yellow-600';
    if (score >= 30) return 'text-orange-600';
    return 'text-red-600';
  };

  return (
    <div className="bg-white rounded-lg shadow-md p-6">
      <div className="flex items-center justify-between mb-4">
        <h3 className="text-lg font-semibold text-gray-800">
          Chỉ Số Nhu Cầu Du Lịch
        </h3>
        <span className="text-xs text-gray-500 bg-gray-100 px-2 py-1 rounded">
          {demandIndex.period_type === 'week' && '7 ngày'}
          {demandIndex.period_type === 'month' && '30 ngày'}
          {demandIndex.period_type === 'quarter' && '90 ngày'}
        </span>
      </div>

      <div className={`text-center p-6 rounded-lg mb-4 ${getIndexColor(demandIndex.overall_index)}`}>
        <div className="text-4xl font-bold mb-2">
          {demandIndex.overall_index.toFixed(1)}
        </div>
        <div className="text-sm font-medium">Điểm Tổng Thể</div>
        {demandIndex.rank_national && (
          <div className="mt-2 text-xs">
            #{demandIndex.rank_national} toàn quốc
            {type === 'attraction' && (demandIndex as DemandIndex).rank_in_province && (
              <> • #{(demandIndex as DemandIndex).rank_in_province} trong tỉnh</>
            )}
          </div>
        )}
      </div>

      <div className="grid grid-cols-2 gap-4">
        <div className="text-center">
          <div className={`text-2xl font-semibold ${getScoreColor(demandIndex.comment_volume_score)}`}>
            {demandIndex.comment_volume_score.toFixed(1)}
          </div>
          <div className="text-xs text-gray-600 mt-1">Khối lượng bình luận</div>
          <div className="text-xs text-gray-500">{demandIndex.total_comments} bình luận</div>
        </div>

        <div className="text-center">
          <div className={`text-2xl font-semibold ${getScoreColor(demandIndex.sentiment_score)}`}>
            {demandIndex.sentiment_score.toFixed(1)}
          </div>
          <div className="text-xs text-gray-600 mt-1">Cảm xúc tích cực</div>
          <div className="text-xs text-gray-500">{demandIndex.positive_rate.toFixed(0)}% tích cực</div>
        </div>

        <div className="text-center">
          <div className={`text-2xl font-semibold ${getScoreColor(demandIndex.engagement_score)}`}>
            {demandIndex.engagement_score.toFixed(1)}
          </div>
          <div className="text-xs text-gray-600 mt-1">Mức độ tương tác</div>
          <div className="text-xs text-gray-500">
            {'total_engagement' in demandIndex && 
              Math.round(demandIndex.total_engagement / Math.max(demandIndex.total_comments, 1))
            } {demandIndex.total_comments > 0 ? 'like/reply' : 'Chưa có dữ liệu'}
          </div>
        </div>

        <div className="text-center">
          <div className={`text-2xl font-semibold ${getScoreColor(demandIndex.growth_score)}`}>
            {demandIndex.growth_score.toFixed(1)}
          </div>
          <div className="text-xs text-gray-600 mt-1">Tăng trưởng</div>
          <div className={`text-xs ${demandIndex.growth_rate >= 0 ? 'text-green-600' : 'text-red-600'}`}>
            {demandIndex.growth_rate > 0 ? '+' : ''}{demandIndex.growth_rate.toFixed(0)}%
          </div>
        </div>
      </div>

      {'active_attractions' in demandIndex && (
        <div className="mt-4 pt-4 border-t border-gray-200">
          <div className="grid grid-cols-2 gap-2 text-sm">
            <div>
              <span className="text-gray-600">Tổng điểm du lịch:</span>
              <span className="ml-2 font-semibold">{demandIndex.total_attractions}</span>
            </div>
            <div>
              <span className="text-gray-600">Điểm hoạt động:</span>
              <span className="ml-2 font-semibold">{demandIndex.active_attractions}</span>
            </div>
          </div>
        </div>
      )}
    </div>
  );
};

export default DemandIndexCard;
