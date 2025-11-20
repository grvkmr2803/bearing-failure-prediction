-- ==============================================================================
-- IMS Bearing Condition Monitoring — PostgreSQL Schema
-- Production-grade schema for raw data → features → predictions → model registry
-- ==============================================================================

-- Drop existing tables (dev only)
DROP TABLE IF EXISTS predictions CASCADE;
DROP TABLE IF EXISTS features CASCADE;
DROP TABLE IF EXISTS windows CASCADE;
DROP TABLE IF EXISTS raw_data CASCADE;
DROP TABLE IF EXISTS model_registry CASCADE;
DROP TABLE IF EXISTS experiments CASCADE;

-- ------------------------------------------------------------------------------
-- Table: raw_data
-- Stores raw vibration samples (optional—can skip if processing on-the-fly)
-- Use for debugging or re-processing; otherwise load directly to windows
-- ------------------------------------------------------------------------------
CREATE TABLE raw_data (
    id BIGSERIAL PRIMARY KEY,
    file_name VARCHAR(255) NOT NULL,
    timestamp TIMESTAMPTZ NOT NULL,
    bearing_id SMALLINT NOT NULL CHECK (bearing_id BETWEEN 1 AND 4),
    axis CHAR(1) NOT NULL CHECK (axis IN ('x', 'y')),
    channel_idx SMALLINT NOT NULL CHECK (channel_idx BETWEEN 0 AND 7),
    sample_idx INTEGER NOT NULL,
    amplitude REAL NOT NULL,
    fs INTEGER DEFAULT 20000,
    created_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX idx_raw_file_bearing ON raw_data(file_name, bearing_id, axis);
CREATE INDEX idx_raw_timestamp ON raw_data(timestamp);

-- ------------------------------------------------------------------------------
-- Table: windows
-- Stores metadata for each windowed segment (no raw samples—saves space)
-- Links to features table via window_id
-- ------------------------------------------------------------------------------
CREATE TABLE windows (
    window_id BIGSERIAL PRIMARY KEY,
    file_name VARCHAR(255) NOT NULL,
    timestamp TIMESTAMPTZ NOT NULL,
    bearing_id SMALLINT NOT NULL,
    axis CHAR(1) NOT NULL,
    window_idx INTEGER NOT NULL,
    window_size INTEGER NOT NULL DEFAULT 2048,
    overlap_pct REAL NOT NULL DEFAULT 0.5,
    fs INTEGER NOT NULL DEFAULT 20000,
    created_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX idx_win_file ON windows(file_name);
CREATE INDEX idx_win_timestamp ON windows(timestamp);
CREATE INDEX idx_win_bearing_axis ON windows(bearing_id, axis);

-- ------------------------------------------------------------------------------
-- Table: features
-- Stores aggregated features per file-bearing-axis (one row per channel)
-- This matches your current features_sample.csv schema
-- ------------------------------------------------------------------------------
CREATE TABLE features (
    feature_id BIGSERIAL PRIMARY KEY,
    file_name VARCHAR(255) NOT NULL,
    timestamp TIMESTAMPTZ NOT NULL,
    channel SMALLINT NOT NULL,
    bearing_id SMALLINT NOT NULL,
    axis CHAR(1) NOT NULL,
    n_windows INTEGER,

    -- Time-domain features (mean + std across windows)
    rms_mean REAL,
    rms_std REAL,
    kurtosis_mean REAL,
    kurtosis_std REAL,
    skew_mean REAL,
    skew_std REAL,
    crest_factor_mean REAL,
    crest_factor_std REAL,
    peak_to_peak_mean REAL,
    peak_to_peak_std REAL,
    std_mean REAL,
    std_std REAL,

    -- Frequency-domain features
    spec_centroid_mean REAL,
    spec_centroid_std REAL,
    bp_0_1k_mean REAL,
    bp_0_1k_std REAL,
    bp_1k_5k_mean REAL,
    bp_1k_5k_std REAL,
    bp_5k_10k_mean REAL,
    bp_5k_10k_std REAL,

    -- Temporal/engineered features (from your temporal_features.py)
    -- Rolling features
    rms_mean_roll_mean_5 REAL,
    rms_mean_roll_std_5 REAL,
    kurtosis_mean_roll_mean_5 REAL,

    -- Delta features
    rms_mean_diff REAL,
    rms_mean_pctchg REAL,

    -- EMA features
    rms_mean_ema_10 REAL,
    rms_mean_ema_30 REAL,

    -- Slope features
    rms_mean_slope_5 REAL,
    rms_mean_slope_10 REAL,

    -- Z-score
    rms_mean_zscore REAL,

    -- Labels
    rul_hours REAL,
    rul_seconds REAL,
    failed BOOLEAN DEFAULT FALSE,
    censored BOOLEAN DEFAULT TRUE,

    created_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX idx_feat_timestamp ON features(timestamp);
CREATE INDEX idx_feat_bearing ON features(bearing_id, axis);
CREATE INDEX idx_feat_file ON features(file_name);
CREATE INDEX idx_feat_failed ON features(failed) WHERE failed = TRUE;

-- ------------------------------------------------------------------------------
-- Table: predictions
-- Stores model predictions (classification + RUL regression)
-- ------------------------------------------------------------------------------
CREATE TABLE predictions (
    prediction_id BIGSERIAL PRIMARY KEY,
    feature_id BIGINT REFERENCES features(feature_id) ON DELETE CASCADE,
    model_id INTEGER NOT NULL,
    timestamp TIMESTAMPTZ NOT NULL,
    bearing_id SMALLINT NOT NULL,
    axis CHAR(1) NOT NULL,

    -- Classification outputs
    pred_class VARCHAR(50),  -- 'Healthy', 'Warning', 'Critical'
    pred_proba_healthy REAL,
    pred_proba_warning REAL,
    pred_proba_critical REAL,

    -- Regression outputs
    pred_rul_hours REAL,
    pred_rul_confidence REAL,

    -- Ground truth (if available)
    true_rul_hours REAL,
    true_class VARCHAR(50),

    -- Errors
    mae REAL,

    created_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX idx_pred_model ON predictions(model_id);
CREATE INDEX idx_pred_bearing ON predictions(bearing_id, axis);
CREATE INDEX idx_pred_timestamp ON predictions(timestamp);

-- ------------------------------------------------------------------------------
-- Table: model_registry
-- Tracks trained models, hyperparameters, and performance metrics
-- ------------------------------------------------------------------------------
CREATE TABLE model_registry (
    model_id SERIAL PRIMARY KEY,
    model_name VARCHAR(255) NOT NULL,
    model_type VARCHAR(100) NOT NULL,  -- 'LightGBM', 'XGBoost', 'RandomForest'
    task_type VARCHAR(50) NOT NULL,    -- 'classification', 'regression'
    version VARCHAR(50) NOT NULL,

    -- Paths
    model_path TEXT NOT NULL,
    feature_list TEXT[],  -- Array of feature names used

    -- Hyperparameters (JSON)
    hyperparams JSONB,

    -- Metrics
    train_mae REAL,
    train_rmse REAL,
    train_r2 REAL,
    test_mae REAL,
    test_rmse REAL,
    test_r2 REAL,
    pr_auc REAL,
    roc_auc REAL,

    -- Metadata
    trained_on TIMESTAMPTZ DEFAULT NOW(),
    trained_by VARCHAR(100),
    notes TEXT,
    is_active BOOLEAN DEFAULT TRUE,

    created_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX idx_model_active ON model_registry(is_active) WHERE is_active = TRUE;
CREATE INDEX idx_model_type ON model_registry(model_type, task_type);

-- ------------------------------------------------------------------------------
-- Table: experiments
-- Tracks experiments for reproducibility and comparison
-- ------------------------------------------------------------------------------
CREATE TABLE experiments (
    experiment_id SERIAL PRIMARY KEY,
    experiment_name VARCHAR(255) NOT NULL,
    description TEXT,
    model_id INTEGER REFERENCES model_registry(model_id),

    -- Config
    feature_set VARCHAR(100),  -- 'baseline', 'temporal', 'selected_top50'
    train_split_date TIMESTAMPTZ,
    test_split_date TIMESTAMPTZ,

    -- Results
    metrics JSONB,

    -- Metadata
    started_at TIMESTAMPTZ DEFAULT NOW(),
    completed_at TIMESTAMPTZ,
    status VARCHAR(50) DEFAULT 'running'  -- 'running', 'completed', 'failed'
);

CREATE INDEX idx_exp_name ON experiments(experiment_name);
CREATE INDEX idx_exp_status ON experiments(status);

-- ------------------------------------------------------------------------------
-- Grant permissions (adjust for your user)
-- ------------------------------------------------------------------------------
-- GRANT ALL ON ALL TABLES IN SCHEMA public TO your_db_user;
-- GRANT USAGE, SELECT ON ALL SEQUENCES IN SCHEMA public TO your_db_user;
