import { useState, useEffect } from 'react';
import SearchBar from '../components/SearchBar';
import { getProvinces, getAttractions, Province, TouristAttraction } from '../services/api';
import { useNavigate } from 'react-router-dom';
import { MapPin, TrendingUp, Star } from 'lucide-react';

const HomePage = () => {
  const [provinces, setProvinces] = useState<Province[]>([]);
  const [topAttractions, setTopAttractions] = useState<TouristAttraction[]>([]);
  const [isLoading, setIsLoading] = useState(true);
  const navigate = useNavigate();

  useEffect(() => {
    const fetchData = async () => {
      try {
        const [provincesData, attractionsData] = await Promise.all([
          getProvinces(),
          getAttractions({ limit: 8 }),
        ]);
        setProvinces(provincesData.slice(0, 12));
        setTopAttractions(attractionsData);
      } catch (error) {
        console.error('Error fetching data:', error);
      } finally {
        setIsLoading(false);
      }
    };

    fetchData();
  }, []);

  if (isLoading) {
    return (
      <div className="flex items-center justify-center min-h-[60vh]">
        <div className="text-center">
          <div className="animate-spin rounded-full h-16 w-16 border-b-2 border-gray-900 mx-auto"></div>
          <p className="mt-4 text-gray-700 font-medium">Loading data...</p>
        </div>
      </div>
    );
  }

  return (
    <div className="space-y-12">
      {/* Hero Section */}
      <div className="relative bg-gradient-to-br from-gray-900 via-black to-gray-900 rounded-lg shadow-2xl p-12 text-white">
        <div className="absolute inset-0 bg-[url('data:image/svg+xml;base64,PHN2ZyB3aWR0aD0iNjAiIGhlaWdodD0iNjAiIHZpZXdCb3g9IjAgMCA2MCA2MCIgeG1sbnM9Imh0dHA6Ly93d3cudzMub3JnLzIwMDAvc3ZnIj48ZyBmaWxsPSJub25lIiBmaWxsLXJ1bGU9ImV2ZW5vZGQiPjxnIGZpbGw9IiNmZmYiIGZpbGwtb3BhY2l0eT0iMC4wNSI+PHBhdGggZD0iTTM2IDE0YzMuMzEgMCA2LTIuNjkgNi02cy0yLjY5LTYtNi02LTYgMi42OS02IDYgMi42OSA2IDYgNnptMCAxMmMzLjMxIDAgNi0yLjY5IDYtNnMtMi42OS02LTYtNi02IDIuNjktNiA2IDIuNjkgNiA2IDZ6bTAgMTJjMy4zMSAwIDYtMi42OSA2LTZzLTIuNjktNi02LTYtNiAyLjY5LTYgNiAyLjY5IDYgNiA2ek0xMiA0NGMzLjMxIDAgNi0yLjY5IDYtNnMtMi42OS02LTYtNi02IDIuNjktNiA2IDIuNjkgNiA2IDZ6bTAgMTJjMy4zMSAwIDYtMi42OSA2LTZzLTIuNjktNi02LTYtNiAyLjY5LTYgNiAyLjY5IDYgNiA2ek0xMiAxNGMzLjMxIDAgNi0yLjY5IDYtNnMtMi42OS02LTYtNi02IDIuNjktNiA2IDIuNjkgNiA2IDZ6Ii8+PC9nPjwvZz48L3N2Zz4=')] opacity-20 rounded-lg"></div>
        <div className="max-w-4xl mx-auto text-center space-y-6 relative z-10">
          <div className="inline-block px-4 py-2 bg-white/10 backdrop-blur-sm rounded-md mb-4">
            <span className="text-sm font-bold tracking-wider text-green-400">LIVE DATA ANALYTICS</span>
          </div>
          <h1 className="text-6xl font-black mb-4 tracking-tight">
            Discover Vietnam
          </h1>
          <p className="text-xl text-gray-300 mb-8 font-medium">
            Real-time insights from top tourist destinations
          </p>
          <SearchBar />
        </div>
      </div>

      {/* Stats Section */}
      <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
        <div className="bg-white rounded-lg shadow-lg p-6 border-l-4 border-gray-900 hover:shadow-xl transition-shadow">
          <div className="flex items-center justify-between">
            <div>
              <p className="text-gray-700 text-sm font-bold uppercase tracking-wide">Provinces</p>
              <p className="text-4xl font-black text-gray-900 mt-2">{provinces.length}</p>
            </div>
            <MapPin className="w-12 h-12 text-gray-300 opacity-50" />
          </div>
        </div>

        <div className="bg-white rounded-lg shadow-lg p-6 border-l-4 border-green-500 hover:shadow-xl transition-shadow">
          <div className="flex items-center justify-between">
            <div>
              <p className="text-gray-700 text-sm font-bold uppercase tracking-wide">Attractions</p>
              <p className="text-4xl font-black text-gray-900 mt-2">{topAttractions.length}+</p>
            </div>
            <Star className="w-12 h-12 text-green-300 opacity-50" />
          </div>
        </div>

        <div className="bg-white rounded-lg shadow-lg p-6 border-l-4 border-blue-500 hover:shadow-xl transition-shadow">
          <div className="flex items-center justify-between">
            <div>
              <p className="text-gray-700 text-sm font-bold uppercase tracking-wide">Live Updates</p>
              <p className="text-4xl font-black text-gray-900 mt-2">24/7</p>
            </div>
            <TrendingUp className="w-12 h-12 text-blue-300 opacity-50" />
          </div>
        </div>
      </div>

      {/* Provinces Section */}
      <div>
        <div className="flex items-center justify-between mb-6">
          <h2 className="text-3xl font-black text-gray-900 tracking-tight">Explore Provinces</h2>
          <button className="text-gray-900 hover:text-gray-700 font-bold flex items-center space-x-2 group">
            <span>View All</span>
            <span className="group-hover:translate-x-1 transition-transform">→</span>
          </button>
        </div>
        <div className="grid grid-cols-2 md:grid-cols-3 lg:grid-cols-4 xl:grid-cols-6 gap-4">
          {provinces.map((province) => (
            <button
              key={province.id}
              onClick={() => navigate(`/province/${province.id}`)}
              className="relative rounded-lg shadow-md hover:shadow-2xl transition-all duration-300 overflow-hidden h-48 group hover:-translate-y-1"
              style={province.image_url ? {
                backgroundImage: `url(${province.image_url})`,
                backgroundSize: 'cover',
                backgroundPosition: 'center'
              } : undefined}
            >
              {/* Dark overlay */}
              <div className="absolute inset-0 bg-gradient-to-t from-black/80 via-black/40 to-black/20 group-hover:from-black/70 transition-colors" />
              
              {/* Fallback background if no image */}
              {!province.image_url && (
                <div className="absolute inset-0 bg-gradient-to-br from-gray-900 to-gray-700" />
              )}
              
              {/* Content */}
              <div className="relative h-full flex flex-col items-center justify-center p-6 text-center">
                <h3 className="font-black text-white text-2xl mb-2 drop-shadow-lg">{province.name}</h3>
                <p className="text-sm text-white/90 font-medium drop-shadow">{province.region}</p>
              </div>
            </button>
          ))}
        </div>
      </div>

      {/* Top Attractions Section */}
      <div>
        <div className="flex items-center justify-between mb-6">
          <h2 className="text-3xl font-black text-gray-900 tracking-tight">Top Attractions</h2>
        </div>
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
          {topAttractions.map((attraction) => (
            <button
              key={attraction.id}
              onClick={() => navigate(`/attraction/${attraction.id}`)}
              className="bg-white rounded-lg shadow-md hover:shadow-2xl transition-all duration-300 overflow-hidden text-left group hover:-translate-y-1"
            >
              <div className="relative h-40 bg-gradient-to-br from-gray-800 to-gray-900 overflow-hidden">
                {attraction.image_url ? (
                  <img
                    src={attraction.image_url}
                    alt={attraction.name}
                    className="w-full h-full object-cover group-hover:scale-110 transition-transform duration-300"
                  />
                ) : (
                  <div className="w-full h-full flex items-center justify-center">
                    <MapPin className="w-16 h-16 text-white opacity-50" />
                  </div>
                )}
              </div>
              <div className="p-4">
                <h3 className="font-bold text-gray-900 mb-1 line-clamp-1">
                  {attraction.name}
                </h3>
                <p className="text-sm text-gray-500 line-clamp-1">
                  {attraction.province_name || attraction.address}
                </p>
                {attraction.tourism_type && (
                  <span className="inline-block mt-2 px-3 py-1 text-xs bg-gray-900 text-white font-bold rounded uppercase tracking-wide">
                    {attraction.tourism_type}
                  </span>
                )}
              </div>
            </button>
          ))}
        </div>
      </div>
    </div>
  );
};

export default HomePage;
