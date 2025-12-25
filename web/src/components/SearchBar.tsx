import { useState, useEffect, useRef } from 'react';
import { Search, X } from 'lucide-react';
import { search, Province, TouristAttraction } from '../services/api';
import { useNavigate } from 'react-router-dom';

const SearchBar = () => {
  const [query, setQuery] = useState('');
  const [isOpen, setIsOpen] = useState(false);
  const [results, setResults] = useState<{
    provinces: Province[];
    attractions: TouristAttraction[];
  }>({ provinces: [], attractions: [] });
  const [isLoading, setIsLoading] = useState(false);
  const searchRef = useRef<HTMLDivElement>(null);
  const navigate = useNavigate();

  useEffect(() => {
    const handleClickOutside = (event: MouseEvent) => {
      if (searchRef.current && !searchRef.current.contains(event.target as Node)) {
        setIsOpen(false);
      }
    };

    document.addEventListener('mousedown', handleClickOutside);
    return () => document.removeEventListener('mousedown', handleClickOutside);
  }, []);

  useEffect(() => {
    const delayDebounceFn = setTimeout(async () => {
      if (query.trim().length > 0) {
        setIsLoading(true);
        setIsOpen(true);
        try {
          const data = await search(query);
          setResults(data);
        } catch (error) {
          console.error('Search error:', error);
        } finally {
          setIsLoading(false);
        }
      } else {
        setResults({ provinces: [], attractions: [] });
        setIsOpen(false);
      }
    }, 300);

    return () => clearTimeout(delayDebounceFn);
  }, [query]);

  const handleProvinceClick = (provinceId: number) => {
    navigate(`/province/${provinceId}`);
    setQuery('');
    setIsOpen(false);
  };

  const handleAttractionClick = (attractionId: number) => {
    navigate(`/attraction/${attractionId}`);
    setQuery('');
    setIsOpen(false);
  };

  const clearSearch = () => {
    setQuery('');
    setIsOpen(false);
  };

  return (
    <div ref={searchRef} className="relative w-full max-w-3xl mx-auto">
      <div className="relative">
        <Search className="absolute left-4 top-1/2 transform -translate-y-1/2 text-gray-500 w-5 h-5" />
        <input
          type="text"
          value={query}
          onChange={(e) => setQuery(e.target.value)}
          placeholder="Search provinces or attractions..."
          className="w-full pl-12 pr-12 py-4 text-lg text-gray-900 placeholder-gray-500 border-2 border-gray-300 rounded-md focus:outline-none focus:border-gray-900 focus:ring-2 focus:ring-gray-900/20 transition shadow-lg font-medium bg-white"
        />
        {query && (
          <button
            onClick={clearSearch}
            className="absolute right-4 top-1/2 transform -translate-y-1/2 text-gray-400 hover:text-gray-600"
          >
            <X className="w-5 h-5" />
          </button>
        )}
      </div>

      {isOpen && (
        <div className="absolute z-[100] w-full mt-2 bg-white rounded-lg shadow-2xl border border-gray-200 max-h-96 overflow-y-auto">
          {isLoading ? (
            <div className="p-4 text-center text-gray-500">
              <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-gray-900 mx-auto"></div>
              <p className="mt-2 font-medium">Searching...</p>
            </div>
          ) : (
            <>
              {results.provinces.length > 0 && (
                <div className="border-b border-gray-100">
                  <div className="px-4 py-2 bg-gray-900 text-white text-xs font-bold uppercase tracking-wider sticky top-0">
                    Provinces
                  </div>
                  {results.provinces.map((province) => (
                    <button
                      key={province.id}
                      onClick={() => handleProvinceClick(province.id)}
                      className="w-full px-4 py-3 text-left hover:bg-gray-100 transition flex items-center space-x-3 group"
                    >
                      <div className="w-12 h-12 bg-gray-900 rounded-md flex items-center justify-center text-white font-black group-hover:scale-110 transition-transform">
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
                      <div>
                        <div className="font-bold text-gray-900">{province.name}</div>
                        <div className="text-sm text-gray-500 font-medium">{province.region}</div>
                      </div>
                    </button>
                  ))}
                </div>
              )}

              {results.attractions.length > 0 && (
                <div>
                  <div className="px-4 py-2 bg-gray-900 text-white text-xs font-bold uppercase tracking-wider sticky top-0">
                    Attractions
                  </div>
                  {results.attractions.map((attraction) => (
                    <button
                      key={attraction.id}
                      onClick={() => handleAttractionClick(attraction.id)}
                      className="w-full px-4 py-3 text-left hover:bg-gray-100 transition group"
                    >
                      <div className="font-bold text-gray-900 group-hover:text-gray-700">{attraction.name}</div>
                      <div className="text-sm text-gray-500 font-medium">
                        {attraction.province_name || attraction.address}
                      </div>
                      {attraction.tourism_type && (
                        <span className="inline-block mt-1 px-2 py-1 text-xs bg-gray-900 text-white font-bold rounded uppercase tracking-wide">
                          {attraction.tourism_type}
                        </span>
                      )}
                    </button>
                  ))}
                </div>
              )}

              {results.provinces.length === 0 && results.attractions.length === 0 && (
                <div className="p-4 text-center text-gray-500 font-medium">
                  No results found
                </div>
              )}
            </>
          )}
        </div>
      )}
    </div>
  );
};

export default SearchBar;
