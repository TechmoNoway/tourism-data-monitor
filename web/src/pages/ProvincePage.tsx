import { useState, useEffect } from 'react';
import { useParams, useNavigate } from 'react-router-dom';
import {
  getProvinceById,
  getAttractions,
  Province,
  TouristAttraction,
} from '../services/api';
import AttractionCard from '../components/AttractionCard';
import CollectionTrendsChart from '../components/CollectionTrendsChart';
import { Search, Filter, ThumbsUp, ThumbsDown, TrendingUp, MapPin, ArrowLeft } from 'lucide-react';

type FilterType = 'all' | 'positive' | 'negative' | 'trending';

const ProvincePage = () => {
  const { provinceId } = useParams<{ provinceId: string }>();
  const navigate = useNavigate();
  const [province, setProvince] = useState<Province | null>(null);
  const [attractions, setAttractions] = useState<TouristAttraction[]>([]);
  const [filteredAttractions, setFilteredAttractions] = useState<TouristAttraction[]>([]);
  const [searchQuery, setSearchQuery] = useState('');
  const [activeFilter, setActiveFilter] = useState<FilterType>('all');
  const [isLoading, setIsLoading] = useState(true);

  useEffect(() => {
    const fetchData = async () => {
      if (!provinceId) return;

      try {
        const [provinceData, attractionsData] = await Promise.all([
          getProvinceById(Number(provinceId)),
          getAttractions({ province_id: Number(provinceId), limit: 100 }),
        ]);
        setProvince(provinceData);
        setAttractions(attractionsData);
        setFilteredAttractions(attractionsData);
      } catch (error) {
        console.error('Error fetching data:', error);
      } finally {
        setIsLoading(false);
      }
    };

    fetchData();
  }, [provinceId]);

  useEffect(() => {
    let filtered = [...attractions];

    // Apply search filter
    if (searchQuery.trim()) {
      filtered = filtered.filter((attraction) =>
        attraction.name.toLowerCase().includes(searchQuery.toLowerCase())
      );
    }

    // Apply sentiment filter
    if (activeFilter === 'positive') {
      filtered = filtered.sort((a, b) => {
        const aPositive = (a.positive_count || 0) / (a.total_comments || 1);
        const bPositive = (b.positive_count || 0) / (b.total_comments || 1);
        return bPositive - aPositive;
      });
    } else if (activeFilter === 'negative') {
      filtered = filtered.sort((a, b) => {
        const aNegative = (a.negative_count || 0) / (a.total_comments || 1);
        const bNegative = (b.negative_count || 0) / (b.total_comments || 1);
        return bNegative - aNegative;
      });
    } else if (activeFilter === 'trending') {
      filtered = filtered.sort((a, b) => (b.total_comments || 0) - (a.total_comments || 0));
    }

    setFilteredAttractions(filtered);
  }, [searchQuery, activeFilter, attractions]);

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

  if (!province) {
    return (
      <div className="text-center py-12">
        <p className="text-gray-600 text-lg font-medium">Province not found</p>
      </div>
    );
  }

  return (
    <div className="space-y-8">
      {/* Back Button */}
      <button
        onClick={() => navigate('/')}
        className="flex items-center text-gray-600 hover:text-gray-900 transition font-bold group"
      >
        <ArrowLeft className="w-5 h-5 mr-2 group-hover:-translate-x-1 transition-transform" />
        Back to Home
      </button>

      {/* Province Header */}
      <div className="bg-gradient-to-br from-gray-900 via-black to-gray-900 rounded-lg shadow-2xl p-8 text-white">
        <div className="flex items-center space-x-6">
          <div className="w-24 h-24 bg-white/10 backdrop-blur-sm rounded-md flex items-center justify-center text-4xl font-black">
            {province.image_url ? (
              <img
                src={province.image_url}
                alt={province.name}
                className="w-full h-full object-cover"
              />
            ) : (
              province.name.charAt(0)
            )}
          </div>
          <div className="flex-1">
            <h1 className="text-5xl font-black mb-2 tracking-tight">{province.name}</h1>
            <div className="flex items-center space-x-4 text-gray-300">
              <div className="flex items-center">
                <MapPin className="w-5 h-5 mr-2" />
                <span className="font-bold">{province.region}</span>
              </div>
              <div className="flex items-center">
                <TrendingUp className="w-5 h-5 mr-2" />
                <span className="font-bold">{attractions.length} attractions</span>
              </div>
            </div>
            {province.description && (
              <p className="mt-3 text-gray-400 font-medium">{province.description}</p>
            )}
          </div>
        </div>
      </div>

      {/* Collection Trends Chart */}
      <CollectionTrendsChart provinceId={Number(provinceId)} period="6months" height={350} />

      {/* Search and Filter Section */}
      <div className="bg-white rounded-lg shadow-md p-6">
        <div className="flex flex-col lg:flex-row gap-4">
          {/* Search Bar */}
          <div className="flex-1 relative">
            <Search className="absolute left-3 top-1/2 transform -translate-y-1/2 text-gray-500 w-5 h-5" />
            <input
              type="text"
              value={searchQuery}
              onChange={(e) => setSearchQuery(e.target.value)}
              placeholder="Search attractions..."
              className="w-full pl-10 pr-4 py-3 text-gray-900 placeholder-gray-500 border-2 border-gray-300 rounded-md focus:outline-none focus:border-gray-900 focus:ring-2 focus:ring-gray-900/20 font-medium bg-white"
            />
          </div>

          {/* Filters */}
          <div className="flex items-center space-x-2">
            <Filter className="w-5 h-5 text-gray-500" />
            <button
              onClick={() => setActiveFilter('all')}
              className={`px-4 py-2 rounded-md font-bold transition ${
                activeFilter === 'all'
                  ? 'bg-gray-900 text-white shadow-lg'
                  : 'bg-gray-100 text-gray-700 hover:bg-gray-200'
              }`}
            >
              All
            </button>
            <button
              onClick={() => setActiveFilter('trending')}
              className={`px-4 py-2 rounded-md font-bold transition flex items-center ${
                activeFilter === 'trending'
                  ? 'bg-gray-900 text-white shadow-lg'
                  : 'bg-gray-100 text-gray-700 hover:bg-gray-200'
              }`}
            >
              <TrendingUp className="w-4 h-4 mr-1" />
              Trending
            </button>
            <button
              onClick={() => setActiveFilter('positive')}
              className={`px-4 py-2 rounded-md font-bold transition flex items-center ${
                activeFilter === 'positive'
                  ? 'bg-green-600 text-white shadow-lg'
                  : 'bg-gray-100 text-gray-700 hover:bg-gray-200'
              }`}
            >
              <ThumbsUp className="w-4 h-4 mr-1" />
              Positive
            </button>
            <button
              onClick={() => setActiveFilter('negative')}
              className={`px-4 py-2 rounded-md font-bold transition flex items-center ${
                activeFilter === 'negative'
                  ? 'bg-red-600 text-white shadow-lg'
                  : 'bg-gray-100 text-gray-700 hover:bg-gray-200'
              }`}
            >
              <ThumbsDown className="w-4 h-4 mr-1" />
              Negative
            </button>
          </div>
        </div>

        <div className="mt-4 text-sm text-gray-600 font-bold">
          Showing {filteredAttractions.length} / {attractions.length} attractions
        </div>
      </div>

      {/* Attractions Grid */}
      {filteredAttractions.length > 0 ? (
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 xl:grid-cols-4 gap-6">
          {filteredAttractions.map((attraction) => (
            <AttractionCard key={attraction.id} attraction={attraction} />
          ))}
        </div>
      ) : (
        <div className="text-center py-12 bg-white rounded-lg shadow-md">
          <p className="text-gray-500 text-lg font-medium">No attractions found</p>
        </div>
      )}
    </div>
  );
};

export default ProvincePage;
