-- Migration: Add Demand Index Tables
-- Created: 2026-01-11

-- Table 1: Demand Indexes (Per Attraction)
CREATE TABLE demand_indexes (
    id SERIAL PRIMARY KEY,
    attraction_id INTEGER NOT NULL REFERENCES tourist_attractions(id) ON DELETE CASCADE,
    province_id INTEGER NOT NULL REFERENCES provinces(id) ON DELETE CASCADE,
    
    period_start TIMESTAMP WITH TIME ZONE NOT NULL,
    period_end TIMESTAMP WITH TIME ZONE NOT NULL,
    period_type VARCHAR(20) NOT NULL,
    
    overall_index DOUBLE PRECISION DEFAULT 0.0,
    comment_volume_score DOUBLE PRECISION DEFAULT 0.0,
    sentiment_score DOUBLE PRECISION DEFAULT 0.0,
    engagement_score DOUBLE PRECISION DEFAULT 0.0,
    growth_score DOUBLE PRECISION DEFAULT 0.0,
    
    total_comments INTEGER DEFAULT 0,
    positive_rate DOUBLE PRECISION DEFAULT 0.0,
    negative_rate DOUBLE PRECISION DEFAULT 0.0,
    neutral_rate DOUBLE PRECISION DEFAULT 0.0,
    
    avg_sentiment DOUBLE PRECISION DEFAULT 0.0,
    total_engagement INTEGER DEFAULT 0,
    
    growth_rate DOUBLE PRECISION DEFAULT 0.0,
    
    rank_in_province INTEGER,
    rank_national INTEGER,
    
    calculated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    
    CONSTRAINT demand_index_period_check CHECK (period_end > period_start),
    CONSTRAINT demand_index_scores_check CHECK (
        overall_index >= 0 AND overall_index <= 100 AND
        comment_volume_score >= 0 AND comment_volume_score <= 100 AND
        sentiment_score >= 0 AND sentiment_score <= 100 AND
        engagement_score >= 0 AND engagement_score <= 100 AND
        growth_score >= 0 AND growth_score <= 100
    ),
    CONSTRAINT demand_index_rates_check CHECK (
        positive_rate >= 0 AND positive_rate <= 100 AND
        negative_rate >= 0 AND negative_rate <= 100 AND
        neutral_rate >= 0 AND neutral_rate <= 100
    )
);

-- Indexes for demand_indexes
CREATE INDEX idx_demand_attraction_period ON demand_indexes(attraction_id, period_start, period_end);
CREATE INDEX idx_demand_province_period ON demand_indexes(province_id, period_start, period_end);
CREATE INDEX idx_demand_overall ON demand_indexes(overall_index DESC);
CREATE INDEX idx_demand_calculated ON demand_indexes(calculated_at DESC);
CREATE INDEX idx_demand_period_type ON demand_indexes(period_type);
CREATE INDEX idx_demand_rank_province ON demand_indexes(rank_in_province);
CREATE INDEX idx_demand_rank_national ON demand_indexes(rank_national);

-- Table 2: Province Demand Indexes (Per Province)
CREATE TABLE province_demand_indexes (
    id SERIAL PRIMARY KEY,
    province_id INTEGER NOT NULL REFERENCES provinces(id) ON DELETE CASCADE,
    
    period_start TIMESTAMP WITH TIME ZONE NOT NULL,
    period_end TIMESTAMP WITH TIME ZONE NOT NULL,
    period_type VARCHAR(20) NOT NULL,
    
    overall_index DOUBLE PRECISION DEFAULT 0.0,
    comment_volume_score DOUBLE PRECISION DEFAULT 0.0,
    sentiment_score DOUBLE PRECISION DEFAULT 0.0,
    engagement_score DOUBLE PRECISION DEFAULT 0.0,
    growth_score DOUBLE PRECISION DEFAULT 0.0,
    
    total_comments INTEGER DEFAULT 0,
    total_attractions INTEGER DEFAULT 0,
    active_attractions INTEGER DEFAULT 0,
    
    positive_rate DOUBLE PRECISION DEFAULT 0.0,
    avg_sentiment DOUBLE PRECISION DEFAULT 0.0,
    
    growth_rate DOUBLE PRECISION DEFAULT 0.0,
    rank_national INTEGER,
    
    calculated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    
    CONSTRAINT province_demand_period_check CHECK (period_end > period_start),
    CONSTRAINT province_demand_scores_check CHECK (
        overall_index >= 0 AND overall_index <= 100 AND
        comment_volume_score >= 0 AND comment_volume_score <= 100 AND
        sentiment_score >= 0 AND sentiment_score <= 100 AND
        engagement_score >= 0 AND engagement_score <= 100 AND
        growth_score >= 0 AND growth_score <= 100
    )
);

-- Indexes for province_demand_indexes
CREATE INDEX idx_prov_demand_period ON province_demand_indexes(province_id, period_start, period_end);
CREATE INDEX idx_prov_demand_overall ON province_demand_indexes(overall_index DESC);
CREATE INDEX idx_prov_demand_calculated ON province_demand_indexes(calculated_at DESC);
CREATE INDEX idx_prov_demand_period_type ON province_demand_indexes(period_type);
CREATE INDEX idx_prov_demand_rank ON province_demand_indexes(rank_national);

-- Comments
COMMENT ON TABLE demand_indexes IS 'Tourism demand indexes calculated per attraction and time period';
COMMENT ON TABLE province_demand_indexes IS 'Tourism demand indexes calculated per province and time period';

COMMENT ON COLUMN demand_indexes.overall_index IS 'Overall demand index score (0-100)';
COMMENT ON COLUMN demand_indexes.comment_volume_score IS 'Score based on comment volume (0-100)';
COMMENT ON COLUMN demand_indexes.sentiment_score IS 'Score based on sentiment analysis (0-100)';
COMMENT ON COLUMN demand_indexes.engagement_score IS 'Score based on likes and replies (0-100)';
COMMENT ON COLUMN demand_indexes.growth_score IS 'Score based on growth rate (0-100)';
