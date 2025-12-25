import { Link } from 'react-router-dom';
import { MapPin, TrendingUp } from 'lucide-react';
import { useEffect, useState } from 'react';

const Header = () => {
  const [scrolled, setScrolled] = useState(false);

  useEffect(() => {
    const handleScroll = () => {
      setScrolled(window.scrollY > 20);
    };

    window.addEventListener('scroll', handleScroll);
    return () => window.removeEventListener('scroll', handleScroll);
  }, []);

  return (
    <header
      className={`fixed top-0 left-0 right-0 z-50 transition-all duration-300 ${
        scrolled
          ? 'bg-black/80 backdrop-blur-lg shadow-xl'
          : 'bg-gradient-to-r from-gray-900 to-black shadow-lg'
      }`}
    >
      <div className="container mx-auto px-4 py-4">
        <div className="flex items-center justify-between">
          <Link to="/" className="flex items-center space-x-3 hover:opacity-70 transition group">
            <div className="bg-white p-2 rounded-md group-hover:scale-110 transition-transform">
              <MapPin className="w-6 h-6 text-gray-900" />
            </div>
            <div>
              <h1 className="text-2xl font-bold tracking-tight text-white">Tourism Monitor</h1>
              <p className="text-gray-400 text-sm font-medium">Vietnam Travel Insights</p>
            </div>
          </Link>
          <div className="flex items-center space-x-2 bg-white/10 px-4 py-2 rounded-md backdrop-blur-sm">
            <TrendingUp className="w-5 h-5 text-green-400" />
            <span className="text-sm font-semibold text-white">Live Analytics</span>
          </div>
        </div>
      </div>
    </header>
  );
};

export default Header;
