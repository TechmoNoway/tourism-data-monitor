import { useState, useEffect } from 'react';
import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
  ResponsiveContainer,
  Area,
  AreaChart,
} from 'recharts';
import { TrendingUp, Calendar, BarChart3 } from 'lucide-react';

interface CollectionTrendData {
  date: string;
  comment_count: number;
  attraction_count: number;
}

interface TrendStats {
  total_comments: number;
  total_periods: number;
  avg_comments_per_period: number;
  peak_period: {
    date: string;
    count: number;
  };
}

interface CollectionTrendsResponse {
  period: string;
  start_date: string;
  end_date: string;
  trends: CollectionTrendData[];
  statistics: TrendStats;
}

interface CollectionTrendsChartProps {
  attractionId?: number;
  provinceId?: number;
  period?: '7days' | '1week' | '2weeks' | '1month' | '3months' | '6months' | '1year';
  height?: number;
}

const CollectionTrendsChart = ({
  attractionId,
  provinceId,
  period = '6months',
  height = 400,
}: CollectionTrendsChartProps) => {
  const [data, setData] = useState<CollectionTrendsResponse | null>(null);
  const [loading, setLoading] = useState(true);
  const [selectedPeriod, setSelectedPeriod] = useState(period);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    fetchTrends();
    
    // Auto-refresh every 30 seconds to show new collected data
    const intervalId = setInterval(() => {
      fetchTrends();
    }, 30000); // 30 seconds
    
    return () => clearInterval(intervalId);
  }, [attractionId, provinceId, selectedPeriod]);

  const fetchTrends = async () => {
    setLoading(true);
    setError(null);
    
    try {
      const params = new URLSearchParams();
      if (attractionId) params.append('attraction_id', attractionId.toString());
      if (provinceId) params.append('province_id', provinceId.toString());
      params.append('period', selectedPeriod);

      const apiUrl = import.meta.env.VITE_API_URL || '/api';
      const response = await fetch(`${apiUrl}/analytics/collection-trends?${params}`);
      
      if (!response.ok) {
        throw new Error('Failed to fetch trends');
      }
      
      const result = await response.json();
      setData(result);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to load data');
      console.error('Error fetching collection trends:', err);
    } finally {
      setLoading(false);
    }
  };

  const formatDate = (dateString: string) => {
    const date = new Date(dateString);
    return date.toLocaleDateString('en-US', { month: 'short', day: 'numeric' });
  };

  const formatNumber = (num: number) => {
    return num.toLocaleString();
  };

  if (loading) {
    return (
      <div className="bg-white rounded-lg shadow-lg p-6">
        <div className="animate-pulse space-y-4">
          <div className="h-8 bg-gray-200 rounded w-1/3"></div>
          <div className="h-64 bg-gray-100 rounded"></div>
        </div>
      </div>
    );
  }

  if (error) {
    return (
      <div className="bg-white rounded-lg shadow-lg p-6">
        <div className="text-center py-8">
          <BarChart3 className="w-12 h-12 text-gray-400 mx-auto mb-3" />
          <p className="text-gray-600">{error}</p>
          <button
            onClick={fetchTrends}
            className="mt-4 px-4 py-2 bg-gray-900 text-white rounded-lg hover:bg-gray-800 transition"
          >
            Retry
          </button>
        </div>
      </div>
    );
  }

  if (!data || data.trends.length === 0) {
    return (
      <div className="bg-white rounded-lg shadow-lg p-6">
        <div className="text-center py-8">
          <TrendingUp className="w-12 h-12 text-gray-400 mx-auto mb-3" />
          <p className="text-gray-600">No collection data available for this period</p>
        </div>
      </div>
    );
  }

  const periods = [
    { value: '7days', label: '7 Days' },
    { value: '1week', label: '1 Week' },
    { value: '1month', label: '1 Month' },
    { value: '3months', label: '3 Months' },
    { value: '6months', label: '6 Months' },
    { value: '1year', label: '1 Year' },
  ];

  return (
    <div className="bg-white rounded-lg shadow-lg p-6 space-y-6">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <h3 className="text-2xl font-bold text-gray-900 flex items-center gap-2">
            <TrendingUp className="w-6 h-6" />
            Collection Trends
          </h3>
          <p className="text-sm text-gray-600 mt-1">
            Comment collection over time (grouped by week)
          </p>
        </div>
        
        {/* Period Selector */}
        <div className="flex items-center gap-2">
          <Calendar className="w-4 h-4 text-gray-500" />
          <select
            value={selectedPeriod}
            onChange={(e) => setSelectedPeriod(e.target.value as any)}
            className="px-3 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-gray-900 focus:border-transparent text-sm"
          >
            {periods.map((p) => (
              <option key={p.value} value={p.value}>
                {p.label}
              </option>
            ))}
          </select>
        </div>
      </div>

      {/* Statistics Cards */}
      <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
        <div className="bg-gradient-to-br from-blue-50 to-blue-100 rounded-lg p-4 border border-blue-200">
          <p className="text-xs font-semibold text-blue-700 uppercase tracking-wide">Total Comments</p>
          <p className="text-3xl font-black text-blue-900 mt-1">
            {formatNumber(data.statistics.total_comments)}
          </p>
        </div>
        
        <div className="bg-gradient-to-br from-green-50 to-green-100 rounded-lg p-4 border border-green-200">
          <p className="text-xs font-semibold text-green-700 uppercase tracking-wide">Avg per Week</p>
          <p className="text-3xl font-black text-green-900 mt-1">
            {formatNumber(Math.round(data.statistics.avg_comments_per_period))}
          </p>
        </div>
        
        <div className="bg-gradient-to-br from-purple-50 to-purple-100 rounded-lg p-4 border border-purple-200">
          <p className="text-xs font-semibold text-purple-700 uppercase tracking-wide">Collection Periods</p>
          <p className="text-3xl font-black text-purple-900 mt-1">
            {data.statistics.total_periods}
          </p>
        </div>
        
        <div className="bg-gradient-to-br from-orange-50 to-orange-100 rounded-lg p-4 border border-orange-200">
          <p className="text-xs font-semibold text-orange-700 uppercase tracking-wide">Peak Week</p>
          <p className="text-2xl font-black text-orange-900 mt-1">
            {formatNumber(data.statistics.peak_period.count)}
          </p>
          <p className="text-xs text-orange-600 mt-1">
            {data.statistics.peak_period.date && formatDate(data.statistics.peak_period.date)}
          </p>
        </div>
      </div>

      {/* Chart */}
      <div className="pt-4">
        <ResponsiveContainer width="100%" height={height}>
          <AreaChart
            data={data.trends}
            margin={{ top: 10, right: 30, left: 0, bottom: 0 }}
          >
            <defs>
              <linearGradient id="commentGradient" x1="0" y1="0" x2="0" y2="1">
                <stop offset="5%" stopColor="#1f2937" stopOpacity={0.8} />
                <stop offset="95%" stopColor="#1f2937" stopOpacity={0.1} />
              </linearGradient>
            </defs>
            <CartesianGrid strokeDasharray="3 3" stroke="#e5e7eb" />
            <XAxis
              dataKey="date"
              tickFormatter={formatDate}
              stroke="#6b7280"
              style={{ fontSize: '12px', fontWeight: 500 }}
            />
            <YAxis
              stroke="#6b7280"
              style={{ fontSize: '12px', fontWeight: 500 }}
              tickFormatter={formatNumber}
            />
            <Tooltip
              content={({ active, payload }) => {
                if (active && payload && payload.length) {
                  return (
                    <div className="bg-white border-2 border-gray-900 rounded-lg shadow-lg p-4">
                      <p className="font-bold text-gray-900 mb-2">
                        {formatDate(payload[0].payload.date)}
                      </p>
                      <div className="space-y-1">
                        <p className="text-sm font-semibold text-gray-700">
                          Comments: <span className="text-gray-900">{formatNumber(payload[0].payload.comment_count)}</span>
                        </p>
                        <p className="text-sm font-semibold text-gray-700">
                          Attractions: <span className="text-gray-900">{payload[0].payload.attraction_count}</span>
                        </p>
                      </div>
                    </div>
                  );
                }
                return null;
              }}
            />
            <Area
              type="monotone"
              dataKey="comment_count"
              stroke="#1f2937"
              strokeWidth={3}
              fillOpacity={1}
              fill="url(#commentGradient)"
              name="Comments"
            />
          </AreaChart>
        </ResponsiveContainer>
      </div>

      {/* Legend */}
      <div className="flex items-center justify-center gap-6 pt-2 border-t">
        <div className="flex items-center gap-2">
          <div className="w-4 h-4 bg-gray-900 rounded"></div>
          <span className="text-sm font-medium text-gray-700">Comments Collected</span>
        </div>
      </div>
    </div>
  );
};

export default CollectionTrendsChart;
