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

export interface DemandIndex {
  id: number;
  attraction_id: number;
  attraction_name?: string;
  province_id: number;
  province_name?: string;
  period_start: string;
  period_end: string;
  period_type: string;
  overall_index: number;
  comment_volume_score: number;
  sentiment_score: number;
  engagement_score: number;
  growth_score: number;
  total_comments: number;
  positive_rate: number;
  negative_rate: number;
  neutral_rate: number;
  avg_sentiment: number;
  total_engagement: number;
  growth_rate: number;
  rank_national?: number;
  rank_in_province?: number;
}

export interface ProvinceDemandIndex {
  id: number;
  province_id: number;
  province_name?: string;
  period_start: string;
  period_end: string;
  period_type: string;
  overall_index: number;
  comment_volume_score: number;
  sentiment_score: number;
  engagement_score: number;
  growth_score: number;
  total_comments: number;
  total_attractions: number;
  active_attractions: number;
  positive_rate: number;
  avg_sentiment: number;
  growth_rate: number;
  rank_national?: number;
}

export interface TopAttraction {
  id: number;  // attraction_id from backend
  name: string;  // attraction name
  province_name: string;
  overall_index: number;
  total_comments: number;
  positive_rate: number;
  growth_rate: number;
  rank: number;  // rank_national or rank_in_province
}

export interface TopProvince {
  id: number;  // province_id from backend
  name: string;  // province name
  overall_index: number;
  total_comments: number;
  active_attractions: number;
  positive_rate: number;
  growth_rate: number;
  rank: number;  // rank_national
}

export interface DemandTrendData {
  period_start: string;
  period_end: string;
  overall_index: number;
  comment_volume_score: number;
  sentiment_score: number;
  engagement_score: number;
  growth_score: number;
}

export interface DemandAnalytics {
  demand_index: DemandIndex;
  trend: DemandTrendData[];
}

export interface ComparativeAnalysis {
  demand_index: DemandIndex;
  province_average: number;
  national_average: number;
  vs_province_avg: number;
  vs_national_avg: number;
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

// Demand Index APIs
export const getTopAttractionsByDemand = async (params?: {
  period?: string;
  province_id?: number;
  limit?: number;
}) => {
  const response = await api.get<TopAttraction[]>(
    '/analytics/demand/attractions/top',
    { params }
  );
  return response.data;
};

export const getTopProvincesByDemand = async (params?: {
  period?: string;
  limit?: number;
}) => {
  const response = await api.get<TopProvince[]>(
    '/analytics/demand/provinces/top',
    { params }
  );
  return response.data;
};

export const getAttractionDemandIndex = async (
  attractionId: number,
  period?: string
) => {
  const response = await api.get<DemandIndex>(
    `/analytics/demand/attractions/${attractionId}`,
    { params: { period } }
  );
  return response.data;
};

export const getProvinceDemandIndex = async (
  provinceId: number,
  period?: string
) => {
  const response = await api.get<ProvinceDemandIndex>(
    `/analytics/demand/provinces/${provinceId}`,
    { params: { period } }
  );
  return response.data;
};

export const getAttractionDemandAnalytics = async (
  attractionId: number,
  params?: { period?: string; num_periods?: number }
) => {
  const response = await api.get<DemandAnalytics>(
    `/analytics/demand/attractions/${attractionId}/analytics`,
    { params }
  );
  return response.data;
};

export const getAttractionDemandComparison = async (
  attractionId: number,
  period?: string
) => {
  const response = await api.get<ComparativeAnalysis>(
    `/analytics/demand/attractions/${attractionId}/compare`,
    { params: { period } }
  );
  return response.data;
};

export const calculateDemandIndexes = async (period?: string) => {
  const response = await api.post<{ message: string; results: any }>(
    '/analytics/demand/calculate',
    {},
    { params: { period } }
  );
  return response.data;
};

export default api;
