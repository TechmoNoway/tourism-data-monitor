import React from 'react';
import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
  ResponsiveContainer,
} from 'recharts';
import { DemandTrendData } from '../services/api';

interface DemandTrendChartProps {
  trendData: DemandTrendData[];
  period: string;
}

const DemandTrendChart: React.FC<DemandTrendChartProps> = ({ trendData, period }) => {
  const formatDate = (dateStr: string) => {
    const date = new Date(dateStr);
    return `${date.getMonth() + 1}/${date.getDate()}`;
  };

  const chartData = trendData.map((data) => ({
    date: formatDate(data.period_start),
    'Chỉ Số Tổng Thể': data.overall_index,
    'Khối Lượng': data.comment_volume_score,
    'Cảm Xúc': data.sentiment_score,
    'Tương Tác': data.engagement_score,
    'Tăng Trưởng': data.growth_score,
  }));

  const periodText = 
    period === 'week' ? '7 Ngày' : 
    period === 'month' ? '30 Ngày' : 
    '90 Ngày';

  return (
    <div className="bg-white rounded-lg shadow-md p-6">
      <h3 className="text-lg font-semibold text-gray-800 mb-4">
        Xu Hướng Nhu Cầu Du Lịch - {periodText}
      </h3>
      <div style={{ height: '400px' }}>
        {trendData.length === 0 ? (
          <div className="flex items-center justify-center h-full text-gray-500">
            Chưa có dữ liệu xu hướng
          </div>
        ) : (
          <ResponsiveContainer width="100%" height="100%">
            <LineChart data={chartData}>
              <CartesianGrid strokeDasharray="3 3" />
              <XAxis
                dataKey="date"
                label={{ value: 'Thời Gian', position: 'insideBottom', offset: -5 }}
              />
              <YAxis
                domain={[0, 100]}
                label={{ value: 'Điểm Số', angle: -90, position: 'insideLeft' }}
              />
              <Tooltip />
              <Legend />
              <Line
                type="monotone"
                dataKey="Chỉ Số Tổng Thể"
                stroke="#3b82f6"
                strokeWidth={2}
                dot={{ r: 4 }}
              />
              <Line
                type="monotone"
                dataKey="Khối Lượng"
                stroke="#10b981"
                strokeWidth={2}
                dot={{ r: 4 }}
              />
              <Line
                type="monotone"
                dataKey="Cảm Xúc"
                stroke="#f59e0b"
                strokeWidth={2}
                dot={{ r: 4 }}
              />
              <Line
                type="monotone"
                dataKey="Tương Tác"
                stroke="#8b5cf6"
                strokeWidth={2}
                dot={{ r: 4 }}
              />
              <Line
                type="monotone"
                dataKey="Tăng Trưởng"
                stroke="#ec4899"
                strokeWidth={2}
                dot={{ r: 4 }}
              />
            </LineChart>
          </ResponsiveContainer>
        )}
      </div>
    </div>
  );
};

export default DemandTrendChart;
