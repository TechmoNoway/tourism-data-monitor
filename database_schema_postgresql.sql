-- CREATE DATABASE tourism_db;
-- \c tourism_db;

-- 1. TABLE: provinces
CREATE TABLE provinces (
    id              SERIAL PRIMARY KEY,
    name            VARCHAR(100) NOT NULL,
    code            VARCHAR(10) UNIQUE NOT NULL,
    main_city       VARCHAR(100),
    created_at      TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- 2. TABLE: tourist_attractions
CREATE TABLE tourist_attractions (
    id                SERIAL PRIMARY KEY,
    name              VARCHAR(200) NOT NULL,
    category          VARCHAR(100),
    address           VARCHAR(500),
    province_id       INTEGER NOT NULL,
    description       TEXT,
    keywords          JSONB,                          -- JSON for better performance in PostgreSQL
    google_place_id   VARCHAR(200),
    rating            DECIMAL(3,2) DEFAULT 0.0,
    total_reviews     INTEGER DEFAULT 0,
    is_active         BOOLEAN DEFAULT TRUE,
    created_at        TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_at        TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    
    FOREIGN KEY (province_id) REFERENCES provinces(id) ON DELETE CASCADE,
    
    -- Constraints
    CONSTRAINT rating_range CHECK (rating >= 0.0 AND rating <= 5.0),
    CONSTRAINT reviews_positive CHECK (total_reviews >= 0)
);

-- 3. TABLE: social_posts
CREATE TABLE social_posts (
    id                  SERIAL PRIMARY KEY,
    platform            VARCHAR(50) NOT NULL,
    platform_post_id    VARCHAR(200) NOT NULL,
    post_url            VARCHAR(1000),
    title               VARCHAR(300),
    content             TEXT,
    author              VARCHAR(200),
    author_id           VARCHAR(200),
    attraction_id       INTEGER NOT NULL,
    view_count          INTEGER DEFAULT 0,
    like_count          INTEGER DEFAULT 0,
    comment_count       INTEGER DEFAULT 0,
    share_count         INTEGER DEFAULT 0,
    post_date           TIMESTAMP WITH TIME ZONE,
    scraped_at          TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    last_updated        TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,

    FOREIGN KEY (attraction_id) REFERENCES tourist_attractions(id) ON DELETE CASCADE,
    
    -- Constraints
    CONSTRAINT platform_values CHECK (platform IN ('youtube', 'facebook', 'tiktok', 'google_review')),
    CONSTRAINT positive_counts CHECK (
        view_count >= 0 AND 
        like_count >= 0 AND 
        comment_count >= 0 AND 
        share_count >= 0
    ),
    
    -- Unique constraint for platform + post_id
    UNIQUE(platform, platform_post_id)
);

-- 4. TABLE: comments
CREATE TABLE comments (
    id                      SERIAL PRIMARY KEY,
    platform                VARCHAR(50) NOT NULL,
    platform_comment_id     VARCHAR(200) NOT NULL,
    post_id                 INTEGER,
    attraction_id           INTEGER NOT NULL,
    content                 TEXT NOT NULL,
    author                  VARCHAR(200),
    author_id               VARCHAR(100),
    like_count              INTEGER DEFAULT 0,
    reply_count             INTEGER DEFAULT 0,
    comment_date            TIMESTAMP WITH TIME ZONE,
    scraped_at              TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    
    -- Data cleaning fields
    cleaned_content         TEXT,                                   -- Content after preprocessing
    is_valid                BOOLEAN DEFAULT TRUE,                   -- Quality filter result
    language                VARCHAR(10),                            -- 'vi', 'en', 'zh-cn', 'ko', 'ja', 'th', 'unknown'
    word_count              INTEGER,                                -- Word count in comment
    
    -- Sentiment analysis fields (Multi-language support: PhoBERT + XLM-RoBERTa)
    sentiment               VARCHAR(20),                            -- 'positive', 'neutral', 'negative'
    sentiment_score         DECIMAL(4,3),                          -- Confidence score 0.000 to 1.000
    analysis_model          VARCHAR(50),                            -- 'phobert', 'xlm-roberta', 'rule-based'
    analyzed_at             TIMESTAMP WITH TIME ZONE,              -- Analysis timestamp
    
    -- Topic classification fields
    topics                  JSONB,                                  -- ['scenery', 'food', 'service', 'pricing', 'accessibility']
    aspect_sentiments       JSONB,                                  -- {'scenery': 'positive', 'food': 'neutral'}
    
    -- Spam/quality detection fields
    is_spam                 BOOLEAN DEFAULT FALSE,                  -- Spam/bot detection result
    spam_score              DECIMAL(4,3),                          -- Spam confidence 0.000 to 1.000

    FOREIGN KEY (post_id) REFERENCES social_posts(id) ON DELETE CASCADE,
    FOREIGN KEY (attraction_id) REFERENCES tourist_attractions(id) ON DELETE CASCADE,
    
    -- Constraints
    CONSTRAINT sentiment_values CHECK (
        sentiment IN ('positive', 'negative', 'neutral') OR sentiment IS NULL
    ),
    CONSTRAINT sentiment_score_range CHECK (
        sentiment_score >= 0.0 AND sentiment_score <= 1.0 OR sentiment_score IS NULL
    ),
    CONSTRAINT spam_score_range CHECK (
        spam_score >= 0.0 AND spam_score <= 1.0 OR spam_score IS NULL
    ),
    CONSTRAINT positive_engagement CHECK (
        like_count >= 0 AND reply_count >= 0
    ),
    CONSTRAINT word_count_positive CHECK (
        word_count >= 0 OR word_count IS NULL
    ),
    
    -- Unique constraint for platform + comment_id
    UNIQUE(platform, platform_comment_id)
);

-- 5. TABLE: analysis_logs
CREATE TABLE analysis_logs (
    id                  SERIAL PRIMARY KEY,
    attraction_id       INTEGER NOT NULL,
    analysis_date       TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    total_comments      INTEGER DEFAULT 0,
    positive_comments   INTEGER DEFAULT 0,
    negative_comments   INTEGER DEFAULT 0,
    neutral_comments    INTEGER DEFAULT 0,
    bot_comments        INTEGER DEFAULT 0,
    avg_sentiment       DECIMAL(4,3) DEFAULT 0.0,
    trending_aspects    JSONB,                          -- JSON for trending aspects
    activity_score      DECIMAL(4,2) DEFAULT 0.0,      -- 0.00 đến 10.00
    
    FOREIGN KEY (attraction_id) REFERENCES tourist_attractions(id) ON DELETE CASCADE,
    
    -- Constraints
    CONSTRAINT comment_counts_positive CHECK (
        total_comments >= 0 AND
        positive_comments >= 0 AND
        negative_comments >= 0 AND
        neutral_comments >= 0 AND
        bot_comments >= 0
    ),
    CONSTRAINT sentiment_range CHECK (
        avg_sentiment >= -1.0 AND avg_sentiment <= 1.0
    ),
    CONSTRAINT activity_score_range CHECK (
        activity_score >= 0.0 AND activity_score <= 10.0
    ),
    CONSTRAINT comment_breakdown_logic CHECK (
        positive_comments + negative_comments + neutral_comments <= total_comments
    )
);

-- INDEXES FOR PERFORMANCE OPTIMIZATION

-- Indexes for provinces table
CREATE INDEX idx_provinces_code ON provinces(code);

-- Indexes for tourist_attractions table
CREATE INDEX idx_attractions_province ON tourist_attractions(province_id);
CREATE INDEX idx_attractions_active ON tourist_attractions(is_active);
CREATE INDEX idx_attractions_category ON tourist_attractions(category);
CREATE INDEX idx_attractions_rating ON tourist_attractions(rating DESC);
CREATE INDEX idx_attractions_keywords ON tourist_attractions USING GIN (keywords);

-- Indexes for social_posts table
CREATE INDEX idx_posts_attraction ON social_posts(attraction_id);
CREATE INDEX idx_posts_platform ON social_posts(platform);
CREATE INDEX idx_posts_author ON social_posts(author_id);
CREATE INDEX idx_posts_date ON social_posts(post_date DESC);
CREATE INDEX idx_posts_scraped ON social_posts(scraped_at DESC);

-- Indexes for comments table
CREATE INDEX idx_comments_post ON comments(post_id);
CREATE INDEX idx_comments_attraction ON comments(attraction_id);
CREATE INDEX idx_comments_author ON comments(author_id);
CREATE INDEX idx_comments_platform ON comments(platform);
CREATE INDEX idx_comments_date ON comments(comment_date DESC);
CREATE INDEX idx_comments_scraped ON comments(scraped_at DESC);

-- Indexes for sentiment analysis
CREATE INDEX idx_comments_sentiment ON comments(sentiment);
CREATE INDEX idx_comments_language ON comments(language);
CREATE INDEX idx_comments_analyzed ON comments(analyzed_at DESC);
CREATE INDEX idx_comments_sentiment_score ON comments(sentiment_score DESC);
CREATE INDEX idx_comments_valid ON comments(is_valid);
CREATE INDEX idx_comments_spam ON comments(is_spam);

-- GIN indexes for JSONB fields
CREATE INDEX idx_comments_topics ON comments USING GIN (topics);
CREATE INDEX idx_comments_aspects ON comments USING GIN (aspect_sentiments);

-- Indexes for analysis_logs table
CREATE INDEX idx_analysis_attraction ON analysis_logs(attraction_id);
CREATE INDEX idx_analysis_date ON analysis_logs(analysis_date DESC);
CREATE INDEX idx_analysis_activity ON analysis_logs(activity_score DESC);
CREATE INDEX idx_analysis_sentiment ON analysis_logs(avg_sentiment DESC);
CREATE INDEX idx_analysis_trending ON analysis_logs USING GIN (trending_aspects);

-- COMPOSITE INDEXES FOR COMPLEX QUERIES

-- Comments by attraction and sentiment
CREATE INDEX idx_comments_attraction_sentiment ON comments(attraction_id, sentiment_label);

-- Posts by attraction and platform
CREATE INDEX idx_posts_attraction_platform ON social_posts(attraction_id, platform);

-- Time-based analysis queries
CREATE INDEX idx_comments_attraction_date ON comments(attraction_id, comment_date DESC);
CREATE INDEX idx_analysis_attraction_date ON analysis_logs(attraction_id, analysis_date DESC);

-- Bot analysis
CREATE INDEX idx_comments_bot_sentiment ON comments(is_bot_suspected, sentiment_label);

-- FULL-TEXT SEARCH INDEXES (PostgreSQL specific)

-- Full-text search for comment content
CREATE INDEX idx_comments_content_fts ON comments USING GIN (to_tsvector('english', content));

-- Full-text search for post content
CREATE INDEX idx_posts_content_fts ON social_posts USING GIN (
    to_tsvector('english', COALESCE(title, '') || ' ' || COALESCE(content, ''))
);

-- FUNCTIONS AND TRIGGERS

-- Function: Update timestamp
CREATE OR REPLACE FUNCTION update_updated_at()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = CURRENT_TIMESTAMP;
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

-- Trigger: Auto update tourist_attractions.updated_at
CREATE TRIGGER update_attraction_timestamp
    BEFORE UPDATE ON tourist_attractions
    FOR EACH ROW
    EXECUTE FUNCTION update_updated_at();

-- Trigger: Auto update social_posts.last_updated
CREATE TRIGGER update_post_timestamp
    BEFORE UPDATE ON social_posts
    FOR EACH ROW
    EXECUTE FUNCTION update_updated_at();

-- MATERIALIZED VIEWS FOR ANALYTICS

-- Materialized view: Province statistics
CREATE MATERIALIZED VIEW province_stats AS
SELECT 
    p.id,
    p.name,
    p.code,
    p.main_city,
    COUNT(ta.id) as total_attractions,
    COUNT(CASE WHEN ta.is_active = true THEN 1 END) as active_attractions,
    COALESCE(SUM(ta.total_reviews), 0) as total_reviews,
    COALESCE(AVG(ta.rating), 0) as avg_rating,
    COUNT(c.id) as total_comments,
    COUNT(CASE WHEN c.sentiment_label = 'positive' THEN 1 END) as positive_comments,
    COUNT(CASE WHEN c.sentiment_label = 'negative' THEN 1 END) as negative_comments,
    COALESCE(AVG(c.sentiment_score), 0) as avg_sentiment
FROM provinces p
LEFT JOIN tourist_attractions ta ON p.id = ta.province_id
LEFT JOIN comments c ON ta.id = c.attraction_id
GROUP BY p.id, p.name, p.code, p.main_city;

-- Create index on materialized view
CREATE INDEX idx_province_stats_code ON province_stats(code);
