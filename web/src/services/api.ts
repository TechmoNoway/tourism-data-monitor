import axios from 'axios';

const api = axios.create({
  baseURL: import.meta.env.VITE_API_URL || '/api',
  headers: {
    'Content-Type': 'application/json',
  },
});

export interface Province {
  id: number;
  name: string;
  region: string;
  description?: string;
  image_url?: string;
}

export interface TouristAttraction {
  id: number;
  name: string;
  province_id: number;
  province_name?: string;
  address?: string;
  description?: string;
  tourism_type?: string;
  image_url?: string;
  total_posts?: number;
  total_comments?: number;
  positive_count?: number;
  negative_count?: number;
  neutral_count?: number;
}

export interface Comment {
  id: number;
  text: string;
  author_name?: string;
  comment_date?: string;
  scraped_at: string;
  sentiment?: string;
  topics?: string[];
  aspect_sentiments?: Record<string, string>;
  platform?: string;
  quality_score?: number;
  quality_tier?: string;
}

export interface AspectSentiment {
  aspect: string;
  total_mentions: number;
  positive_count: number;
  negative_count: number;
  neutral_count: number;
  positive_percentage: number;
  sentiment_score: number;
}

export interface AttractionDetailStats {
  attraction: TouristAttraction;
  total_comments_6months: number;
  meaningful_comments_6months: number;
  overall_sentiment: {
    positive: number;
    negative: number;
    neutral: number;
    total: number;
    avg_score: number;
    positive_percentage: number;
  };
  aspects: AspectSentiment[];
  top_positive_comments: Comment[];
  top_negative_comments: Comment[];
}

// Province APIs
export const getProvinces = async (search?: string) => {
  const response = await api.get<Province[]>('/provinces/', {
    params: { search },
  });
  return response.data;
};

export const getProvinceById = async (id: number) => {
  const response = await api.get<Province>(`/provinces/${id}`);
  return response.data;
};

// Attraction APIs
export const getAttractions = async (params?: {
  province_id?: number;
  search?: string;
  tourism_type?: string;
  skip?: number;
  limit?: number;
}) => {
  const response = await api.get<TouristAttraction[]>('/attractions/', {
    params,
  });
  return response.data;
};

export const getAttractionById = async (id: number) => {
  const response = await api.get<TouristAttraction>(`/attractions/${id}`);
  return response.data;
};

export const getAttractionStats = async (id: number, months: number = 6) => {
  const response = await api.get<AttractionDetailStats>(`/attractions/${id}/detail-stats`, {
    params: { months },
  });
  return response.data;
};

// Comments by aspect
export const getCommentsByAspect = async (params: {
  attraction_id: number;
  aspect?: string;
  sentiment?: string;
  months?: number;
  limit?: number;
  offset?: number;
}) => {
  const response = await api.get<{ comments: Comment[]; total: number }>(
    `/attractions/${params.attraction_id}/comments-by-aspect`,
    { params }
  );
  return response.data;
};

// Search API for both provinces and attractions
export const search = async (query: string) => {
  const [provinces, attractions] = await Promise.all([
    getProvinces(query),
    getAttractions({ search: query }),
  ]);
  return { provinces, attractions };
};

export default api;
