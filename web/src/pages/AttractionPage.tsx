import { useState, useEffect } from 'react';
import { useParams, useNavigate } from 'react-router-dom';
import {
  getAttractionStats,
  getCommentsByAspect,
  getAttractionDemandAnalytics,
  getAttractionDemandComparison,
  AttractionDetailStats,
  Comment,
  DemandAnalytics,
  ComparativeAnalysis,
} from '../services/api';
import DemandIndexCard from '../components/DemandIndexCard';
import DemandTrendChart from '../components/DemandTrendChart';
import {
  MapPin,
  MessageSquare,
  ThumbsUp,
  ThumbsDown,
  TrendingUp,
  ArrowLeft,
} from 'lucide-react';

const AttractionPage = () => {
  const { attractionId } = useParams<{ attractionId: string }>();
  const navigate = useNavigate();
  const [stats, setStats] = useState<AttractionDetailStats | null>(null);
  const [comments, setComments] = useState<Comment[]>([]);
  const [demandAnalytics, setDemandAnalytics] = useState<DemandAnalytics | null>(null);
  const [demandComparison, setDemandComparison] = useState<ComparativeAnalysis | null>(null);
  const [demandPeriod, setDemandPeriod] = useState<string>('month');
  const [activeAspect, setActiveAspect] = useState<string>('all');
  const [activeSentiment, setActiveSentiment] = useState<string>('all');
  const [isLoading, setIsLoading] = useState(true);

  useEffect(() => {
    const fetchData = async () => {
      if (!attractionId) return;

      try {
        const statsData = await getAttractionStats(Number(attractionId));
        setStats(statsData);
        
        // Combine top positive and negative comments
        const allComments = [
          ...(statsData.top_positive_comments || []),
          ...(statsData.top_negative_comments || []),
        ];
        setComments(allComments);
      } catch (error) {
        console.error('Error fetching data:', error);
      } finally {
        setIsLoading(false);
      }
    };

    fetchData();
  }, [attractionId]);

  useEffect(() => {
    const fetchDemandData = async () => {
      if (!attractionId) return;

      try {
        const [analytics, comparison] = await Promise.all([
          getAttractionDemandAnalytics(Number(attractionId), {
            period: demandPeriod,
            num_periods: 6,
          }),
          getAttractionDemandComparison(Number(attractionId), demandPeriod),
        ]);
        setDemandAnalytics(analytics);
        setDemandComparison(comparison);
      } catch (error) {
        console.error('Error fetching demand data:', error);
        setDemandAnalytics(null);
        setDemandComparison(null);
      }
    };

    fetchDemandData();
  }, [attractionId, demandPeriod]);

  // Load comments when aspect or sentiment changes
  useEffect(() => {
    const fetchCommentsByAspect = async () => {
      if (!attractionId || !stats) return;

      try {
        if (activeAspect === 'all' && activeSentiment === 'all') {
          // Show initial top comments when both filters are 'all'
          const allComments = [
            ...(stats.top_positive_comments || []),
            ...(stats.top_negative_comments || []),
          ];
          setComments(allComments);
        } else {
          // Fetch filtered comments from API
          const data = await getCommentsByAspect({
            attraction_id: Number(attractionId),
            aspect: activeAspect !== 'all' ? activeAspect : undefined,
            sentiment: activeSentiment !== 'all' ? activeSentiment : undefined,
            limit: 50,
          });
          setComments(data.comments || []);
        }
      } catch (error) {
        console.error('Error fetching comments:', error);
      }
    };

    fetchCommentsByAspect();
  }, [activeAspect, activeSentiment, attractionId, stats]);

  const aspects = stats?.aspects || [];

  if (isLoading) {
    return (
      <div className="flex items-center justify-center min-h-[60vh]">
        <div className="text-center">
          <div className="animate-spin rounded-full h-16 w-16 border-b-2 border-gray-900 mx-auto"></div>
          <p className="mt-4 text-gray-600 font-medium">Loading data...</p>
        </div>
      </div>
    );
  }

  if (!stats || !stats.attraction) {
    return (
      <div className="text-center py-12">
        <p className="text-gray-600 text-lg font-medium">Attraction not found</p>
      </div>
    );
  }

  const attraction = stats.attraction;
  
  const sentimentPercentage = {
    positive: Math.round(stats.overall_sentiment.positive_percentage) || 0,
    negative: stats.overall_sentiment.total 
      ? Math.round((stats.overall_sentiment.negative / stats.overall_sentiment.total) * 100)
      : 0,
    neutral: stats.overall_sentiment.total
      ? Math.round((stats.overall_sentiment.neutral / stats.overall_sentiment.total) * 100)
      : 0,
  };

  return (
    <div className="space-y-8">
      {/* Back Button */}
      <button
        onClick={() => navigate(-1)}
        className="flex items-center text-gray-600 hover:text-gray-900 transition font-bold group"
      >
        <ArrowLeft className="w-5 h-5 mr-2 group-hover:-translate-x-1 transition-transform" />
        Back
      </button>

      {/* Attraction Header */}
      <div 
        className="relative rounded-lg shadow-2xl overflow-hidden"
        style={attraction.image_url ? {
          backgroundImage: `url(${attraction.image_url})`,
          backgroundSize: 'cover',
          backgroundPosition: 'center'
        } : undefined}
      >
        {/* Dark overlay */}
        <div className="absolute inset-0 bg-gradient-to-t from-black/90 via-black/70 to-black/50" />
        
        {/* Fallback background if no image */}
        {!attraction.image_url && (
          <div className="absolute inset-0 bg-gradient-to-br from-gray-900 via-black to-gray-900" />
        )}
        
        {/* Content */}
        <div className="relative p-8 md:p-12 text-white min-h-[24rem] flex flex-col justify-end">
          <h1 className="text-5xl font-black mb-4 tracking-tight drop-shadow-lg">{attraction.name}</h1>
          {attraction.address && (
            <div className="flex items-start mb-3 text-gray-200">
              <MapPin className="w-5 h-5 mr-2 mt-1 flex-shrink-0 drop-shadow" />
              <span className="font-medium drop-shadow">{attraction.address}</span>
            </div>
          )}
          {attraction.tourism_type && (
            <span className="inline-block px-4 py-2 bg-white text-gray-900 rounded-md text-sm font-black uppercase tracking-wide shadow-lg w-fit">
              {attraction.tourism_type}
            </span>
          )}
          {attraction.description && (
            <p className="mt-4 text-gray-200 font-medium drop-shadow max-w-2xl">{attraction.description}</p>
          )}
        </div>
      </div>

      {/* Stats Cards */}
      <div className="grid grid-cols-1 md:grid-cols-4 gap-6">
        <div className="bg-white rounded-lg shadow-lg p-6 border-l-4 border-gray-900">
          <div className="flex items-center justify-between">
            <div>
              <p className="text-gray-700 text-sm font-bold uppercase tracking-wide">Total Comments</p>
              <p className="text-4xl font-black text-gray-900 mt-1">
                {stats.overall_sentiment.total || 0}
              </p>
            </div>
            <MessageSquare className="w-10 h-10 text-gray-300 opacity-50" />
          </div>
        </div>

        <div className="bg-white rounded-lg shadow-lg p-6 border-l-4 border-green-500">
          <div className="flex items-center justify-between">
            <div>
              <p className="text-gray-700 text-sm font-bold uppercase tracking-wide">Positive</p>
              <p className="text-4xl font-black text-green-600 mt-1">
                {sentimentPercentage.positive}%
              </p>
            </div>
            <ThumbsUp className="w-10 h-10 text-green-300 opacity-50" />
          </div>
          <p className="text-xs text-gray-600 mt-2 font-bold">
            {stats.overall_sentiment.positive || 0} comments
          </p>
        </div>

        <div className="bg-white rounded-lg shadow-lg p-6 border-l-4 border-red-500">
          <div className="flex items-center justify-between">
            <div>
              <p className="text-gray-700 text-sm font-bold uppercase tracking-wide">Negative</p>
              <p className="text-4xl font-black text-red-600 mt-1">
                {sentimentPercentage.negative}%
              </p>
            </div>
            <ThumbsDown className="w-10 h-10 text-red-300 opacity-50" />
          </div>
          <p className="text-xs text-gray-600 mt-2 font-bold">
            {stats.overall_sentiment.negative || 0} comments
          </p>
        </div>

        <div className="bg-white rounded-lg shadow-lg p-6 border-l-4 border-blue-500">
          <div className="flex items-center justify-between">
            <div>
              <p className="text-gray-700 text-sm font-bold uppercase tracking-wide">Neutral</p>
              <p className="text-4xl font-black text-blue-600 mt-1">
                {sentimentPercentage.neutral}%
              </p>
            </div>
            <TrendingUp className="w-10 h-10 text-blue-300 opacity-50" />
          </div>
          <p className="text-xs text-gray-600 mt-2 font-bold">
            {stats.overall_sentiment.neutral || 0} comments
          </p>
        </div>
      </div>

      {/* Demand Index and Trend Analysis */}
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        {demandAnalytics && (
          <>
            <DemandIndexCard demandIndex={demandAnalytics.demand_index} type="attraction" />
            <div className="lg:col-span-2">
              <DemandTrendChart trendData={demandAnalytics.trend} period={demandPeriod} />
            </div>
          </>
        )}
      </div>

      {/* Comparative Analysis */}
      {demandComparison && (
        <div className="bg-white rounded-lg shadow-md p-6">
          <h3 className="text-lg font-semibold text-gray-800 mb-4">So Sánh Nhu Cầu</h3>
          <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
            <div className="text-center p-4 bg-blue-50 rounded-lg">
              <div className="text-2xl font-bold text-blue-600">
                {demandComparison.demand_index.overall_index.toFixed(1)}
              </div>
              <div className="text-sm text-gray-600 mt-1">Điểm Hiện Tại</div>
            </div>
            <div className="text-center p-4 bg-green-50 rounded-lg">
              <div className="text-2xl font-bold text-green-600">
                {demandComparison.province_average.toFixed(1)}
              </div>
              <div className="text-sm text-gray-600 mt-1">Trung Bình Tỉnh</div>
              <div className={`text-xs mt-1 ${demandComparison.vs_province_avg >= 0 ? 'text-green-600' : 'text-red-600'}`}>
                {demandComparison.vs_province_avg > 0 ? '+' : ''}{demandComparison.vs_province_avg.toFixed(1)}%
              </div>
            </div>
            <div className="text-center p-4 bg-purple-50 rounded-lg">
              <div className="text-2xl font-bold text-purple-600">
                {demandComparison.national_average.toFixed(1)}
              </div>
              <div className="text-sm text-gray-600 mt-1">Trung Bình Quốc Gia</div>
              <div className={`text-xs mt-1 ${demandComparison.vs_national_avg >= 0 ? 'text-green-600' : 'text-red-600'}`}>
                {demandComparison.vs_national_avg > 0 ? '+' : ''}{demandComparison.vs_national_avg.toFixed(1)}%
              </div>
            </div>
          </div>
        </div>
      )}

      {/* Aspects Overview */}
      {aspects.length > 0 && (
        <div className="bg-white rounded-lg shadow-lg p-6">
          <h2 className="text-3xl font-black text-gray-900 mb-6 tracking-tight">Aspects Overview</h2>
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
            {aspects.map((aspect) => {
              const positivePercentage = Math.round(aspect.positive_percentage || 0);
              const hasData = aspect.total_mentions > 0;
              
              return (
                <div
                  key={aspect.aspect}
                  className="bg-gray-50 rounded-md p-4 border-2 border-gray-200 hover:border-gray-900 hover:shadow-md transition-all"
                >
                  <h3 className="font-black text-gray-900 capitalize mb-2 text-lg">
                    {aspect.aspect}
                  </h3>
                  <div className="space-y-2">
                    <div className="flex items-center justify-between text-sm">
                      <span className="text-gray-700 font-bold">Total Mentions</span>
                      <span className="font-black text-gray-900">{aspect.total_mentions}</span>
                    </div>
                    <div className="flex items-center justify-between text-sm">
                      <span className="text-green-700 font-bold">Positive</span>
                      <span className="font-black text-green-600">{positivePercentage}%</span>
                    </div>
                    <div className="h-2 bg-gray-200 rounded-full overflow-hidden">
                      <div
                        className="h-full bg-gradient-to-r from-green-500 to-green-400 transition-all duration-300"
                        style={{ width: `${hasData ? positivePercentage : 0}%` }}
                      />
                    </div>
                    {hasData && aspect.positive_count > 0 && (
                      <div className="text-xs text-gray-600 font-medium">
                        {aspect.positive_count} positive / {aspect.total_mentions} total
                      </div>
                    )}
                  </div>
                </div>
              );
            })}
          </div>
        </div>
      )}

      {/* Sentiment Filter */}
      <div className="bg-white rounded-lg shadow-lg p-6">
        <h2 className="text-2xl font-black text-gray-900 mb-4 tracking-tight">Filter by Sentiment</h2>
        <div className="flex flex-wrap gap-2">
          <button
            onClick={() => setActiveSentiment('all')}
            className={`px-4 py-2 rounded-md font-bold transition ${
              activeSentiment === 'all'
                ? 'bg-gray-900 text-white shadow-lg'
                : 'bg-gray-100 text-gray-700 hover:bg-gray-200'
            }`}
          >
            All
          </button>
          <button
            onClick={() => setActiveSentiment('positive')}
            className={`px-4 py-2 rounded-md font-bold transition ${
              activeSentiment === 'positive'
                ? 'bg-green-600 text-white shadow-lg'
                : 'bg-gray-100 text-gray-700 hover:bg-gray-200'
            }`}
          >
            Positive
          </button>
          <button
            onClick={() => setActiveSentiment('negative')}
            className={`px-4 py-2 rounded-md font-bold transition ${
              activeSentiment === 'negative'
                ? 'bg-red-600 text-white shadow-lg'
                : 'bg-gray-100 text-gray-700 hover:bg-gray-200'
            }`}
          >
            Negative
          </button>
          <button
            onClick={() => setActiveSentiment('neutral')}
            className={`px-4 py-2 rounded-md font-bold transition ${
              activeSentiment === 'neutral'
                ? 'bg-blue-600 text-white shadow-lg'
                : 'bg-gray-100 text-gray-700 hover:bg-gray-200'
            }`}
          >
            Neutral
          </button>
        </div>
      </div>

      {/* Aspects Tabs */}
      {aspects.length > 0 && (
        <div className="bg-white rounded-lg shadow-lg p-6">
          <h2 className="text-2xl font-black text-gray-900 mb-4 tracking-tight">Filter by Aspect</h2>
          <div className="flex flex-wrap gap-2 mb-6">
            <button
              onClick={() => setActiveAspect('all')}
              className={`px-4 py-2 rounded-md font-bold transition ${
                activeAspect === 'all'
                  ? 'bg-gray-900 text-white shadow-lg'
                  : 'bg-gray-100 text-gray-700 hover:bg-gray-200'
              }`}
            >
              All
            </button>
            {aspects.map((aspect) => (
              <button
                key={aspect.aspect}
                onClick={() => setActiveAspect(aspect.aspect)}
                className={`px-4 py-2 rounded-md font-bold transition capitalize ${
                  activeAspect === aspect.aspect
                    ? 'bg-gray-900 text-white shadow-lg'
                    : 'bg-gray-100 text-gray-700 hover:bg-gray-200'
                }`}
              >
                {aspect.aspect} ({aspect.total_mentions})
              </button>
            ))}
          </div>

          {/* Comments List */}
          <div className="space-y-4">
            <h3 className="text-xl font-black text-gray-900 tracking-tight">
              Comments ({comments.length})
            </h3>
            {comments.length > 0 ? (
              <div className="space-y-4 max-h-[600px] overflow-y-auto">
                {comments.map((comment) => (
                  <div
                    key={comment.id}
                    className="bg-gray-50 rounded-md p-4 border-2 border-gray-200 hover:border-gray-900 hover:shadow-md transition-all"
                  >
                    <div className="flex items-start justify-between mb-2">
                      <div className="flex items-center space-x-2">
                        <span className="font-black text-gray-900">
                          {comment.author_name || 'Anonymous'}
                        </span>
                        {comment.sentiment && (
                          <span
                            className={`px-2 py-1 text-xs rounded font-bold uppercase tracking-wide ${
                              comment.sentiment === 'positive'
                                ? 'bg-green-100 text-green-700'
                                : comment.sentiment === 'negative'
                                ? 'bg-red-100 text-red-700'
                                : 'bg-blue-100 text-blue-700'
                            }`}
                          >
                            {comment.sentiment === 'positive'
                              ? 'Positive'
                              : comment.sentiment === 'negative'
                              ? 'Negative'
                              : 'Neutral'}
                          </span>
                        )}
                        {comment.platform && (
                          <span className="px-2 py-1 text-xs bg-gray-900 text-white rounded font-bold uppercase tracking-wide">
                            {comment.platform}
                          </span>
                        )}
                      </div>
                      <span className="text-sm text-gray-500 font-medium">
                        {comment.comment_date
                          ? new Date(comment.comment_date).toLocaleDateString('en-US')
                          : new Date(comment.scraped_at).toLocaleDateString('en-US')}
                      </span>
                    </div>
                    <p className="text-gray-700 mb-2 font-medium">{comment.text}</p>
                    {comment.topics && comment.topics.length > 0 && (
                      <div className="flex flex-wrap gap-1">
                        {comment.topics.map((topic, idx) => (
                          <span
                            key={idx}
                            className="px-2 py-1 text-xs bg-gray-900 text-white rounded font-bold uppercase tracking-wide"
                          >
                            {topic}
                          </span>
                        ))}
                      </div>
                    )}
                  </div>
                ))}
              </div>
            ) : (
              <p className="text-gray-500 text-center py-8 font-medium">
                No comments match the selected filters
              </p>
            )}
          </div>
        </div>
      )}
    </div>
  );
};

export default AttractionPage;
