import { TouristAttraction } from '../services/api';
import { MapPin, MessageSquare, ThumbsUp, ThumbsDown } from 'lucide-react';
import { useNavigate } from 'react-router-dom';

interface AttractionCardProps {
  attraction: TouristAttraction;
}

const AttractionCard = ({ attraction }: AttractionCardProps) => {
  const navigate = useNavigate();

  const totalComments = attraction.total_comments || 0;
  const positiveCount = attraction.positive_count || 0;
  const negativeCount = attraction.negative_count || 0;
  const neutralCount = totalComments - positiveCount - negativeCount;

  const sentimentPercentage = {
    positive: totalComments > 0 ? (positiveCount / totalComments) * 100 : 0,
    negative: totalComments > 0 ? (negativeCount / totalComments) * 100 : 0,
    neutral: totalComments > 0 ? (neutralCount / totalComments) * 100 : 0,
  };

  return (
    <div
      onClick={() => navigate(`/attraction/${attraction.id}`)}
      className="bg-white rounded-xl shadow-md hover:shadow-xl transition-all duration-300 cursor-pointer overflow-hidden group"
    >
      <div className="relative h-48 bg-gradient-to-br from-primary-400 to-primary-600 overflow-hidden">
        {attraction.image_url ? (
          <img
            src={attraction.image_url}
            alt={attraction.name}
            className="w-full h-full object-cover group-hover:scale-110 transition-transform duration-300"
          />
        ) : (
          <div className="w-full h-full flex items-center justify-center">
            <MapPin className="w-20 h-20 text-white opacity-50" />
          </div>
        )}
        {attraction.tourism_type && (
          <div className="absolute top-3 right-3 px-3 py-1 bg-white/90 backdrop-blur-sm rounded-full text-sm font-medium text-primary-700">
            {attraction.tourism_type}
          </div>
        )}
      </div>

      <div className="p-5">
        <h3 className="text-xl font-bold text-gray-900 mb-2 line-clamp-1">
          {attraction.name}
        </h3>
        
        {attraction.address && (
          <p className="text-gray-600 text-sm mb-3 line-clamp-2 flex items-start">
            <MapPin className="w-4 h-4 mr-1 mt-0.5 flex-shrink-0" />
            {attraction.address}
          </p>
        )}

        <div className="flex items-center justify-between pt-3 border-t border-gray-100">
          <div className="flex items-center space-x-3">
            <div className="flex items-center text-gray-600">
              <MessageSquare className="w-4 h-4 mr-1" />
              <span className="text-sm font-medium">{totalComments}</span>
            </div>
            <div className="flex items-center text-green-600">
              <ThumbsUp className="w-4 h-4 mr-1" />
              <span className="text-sm font-medium">{Math.round(sentimentPercentage.positive)}%</span>
            </div>
            <div className="flex items-center text-red-600">
              <ThumbsDown className="w-4 h-4 mr-1" />
              <span className="text-sm font-medium">{Math.round(sentimentPercentage.negative)}%</span>
            </div>
          </div>
        </div>

        {/* Sentiment Progress Bar */}
        {totalComments > 0 && (
          <div className="mt-3">
            <div className="flex h-2 bg-gray-200 rounded-full overflow-hidden">
              {sentimentPercentage.positive > 0 && (
                <div
                  className="bg-green-500 transition-all duration-300"
                  style={{ width: `${sentimentPercentage.positive}%` }}
                  title={`Positive: ${Math.round(sentimentPercentage.positive)}%`}
                />
              )}
              {sentimentPercentage.neutral > 0 && (
                <div
                  className="bg-gray-400 transition-all duration-300"
                  style={{ width: `${sentimentPercentage.neutral}%` }}
                  title={`Neutral: ${Math.round(sentimentPercentage.neutral)}%`}
                />
              )}
              {sentimentPercentage.negative > 0 && (
                <div
                  className="bg-red-500 transition-all duration-300"
                  style={{ width: `${sentimentPercentage.negative}%` }}
                  title={`Negative: ${Math.round(sentimentPercentage.negative)}%`}
                />
              )}
            </div>
            <div className="flex justify-between mt-1 text-xs font-medium text-gray-500">
              <span className="text-green-600">{positiveCount} positive</span>
              {neutralCount > 0 && <span className="text-gray-600">{neutralCount} neutral</span>}
              <span className="text-red-600">{negativeCount} negative</span>
            </div>
          </div>
        )}
        
        {totalComments === 0 && (
          <div className="mt-3 text-center text-gray-500 text-sm font-medium py-2">
            No comments yet
          </div>
        )}
      </div>
    </div>
  );
};

export default AttractionCard;
