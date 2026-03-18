-- ============================================================
-- A/B Test Experiment Logging Schema
-- Tracks all experiments, variants, and results over time
-- ============================================================

-- Experiments registry
CREATE TABLE IF NOT EXISTS experiments (
    experiment_id     SERIAL PRIMARY KEY,
    name              VARCHAR(200) NOT NULL,
    description       TEXT,
    hypothesis        TEXT,
    metric_primary    VARCHAR(100) NOT NULL,  -- e.g. 'conversion_rate', 'ctr', 'revenue_per_user'
    metric_secondary  VARCHAR(100),
    min_detectable_effect DECIMAL(5,4),       -- MDE as proportion (e.g. 0.05 = 5%)
    statistical_power DECIMAL(4,3) DEFAULT 0.80,
    significance_level DECIMAL(4,3) DEFAULT 0.05,
    required_sample_size INTEGER,
    started_at        TIMESTAMP,
    ended_at          TIMESTAMP,
    status            VARCHAR(20) DEFAULT 'draft',  -- draft, running, paused, completed, cancelled
    winner_variant    VARCHAR(50),
    created_by        VARCHAR(100),
    created_at        TIMESTAMP DEFAULT NOW()
);

-- Variants per experiment
CREATE TABLE IF NOT EXISTS experiment_variants (
    variant_id        SERIAL PRIMARY KEY,
    experiment_id     INTEGER REFERENCES experiments(experiment_id),
    variant_name      VARCHAR(50) NOT NULL,  -- 'control', 'treatment_a', 'treatment_b'
    is_control        BOOLEAN DEFAULT FALSE,
    traffic_split     DECIMAL(4,3),          -- proportion e.g. 0.50 for 50%
    description       TEXT
);

-- Daily results snapshot per variant
CREATE TABLE IF NOT EXISTS experiment_results (
    result_id         SERIAL PRIMARY KEY,
    experiment_id     INTEGER REFERENCES experiments(experiment_id),
    variant_id        INTEGER REFERENCES experiment_variants(variant_id),
    snapshot_date     DATE NOT NULL,
    users_exposed     INTEGER,
    conversions       INTEGER,
    total_revenue     DECIMAL(12,2),
    conversion_rate   DECIMAL(8,6) GENERATED ALWAYS AS
                        (CASE WHEN users_exposed > 0
                         THEN conversions::DECIMAL / users_exposed
                         ELSE 0 END) STORED,
    revenue_per_user  DECIMAL(10,4) GENERATED ALWAYS AS
                        (CASE WHEN users_exposed > 0
                         THEN total_revenue / users_exposed
                         ELSE 0 END) STORED,
    p_value           DECIMAL(8,6),
    uplift_vs_control DECIMAL(8,4),
    is_significant    BOOLEAN
);

-- ============================================================
-- Analysis Queries
-- ============================================================

-- Latest results summary for all running experiments
CREATE OR REPLACE VIEW v_experiment_summary AS
SELECT
    e.experiment_id,
    e.name,
    e.status,
    e.metric_primary,
    v.variant_name,
    v.is_control,
    r.snapshot_date,
    r.users_exposed,
    r.conversions,
    r.conversion_rate,
    r.revenue_per_user,
    r.p_value,
    r.uplift_vs_control,
    r.is_significant,
    DATEDIFF('day', e.started_at, NOW()) AS days_running,
    e.required_sample_size
FROM experiments e
JOIN experiment_variants v ON e.experiment_id = v.experiment_id
JOIN experiment_results r ON v.variant_id = r.variant_id
WHERE r.snapshot_date = (
    SELECT MAX(snapshot_date) FROM experiment_results
    WHERE experiment_id = e.experiment_id
)
AND e.status = 'running'
ORDER BY e.experiment_id, v.is_control DESC;

-- Sample size progress tracker
CREATE OR REPLACE VIEW v_experiment_progress AS
SELECT
    e.experiment_id,
    e.name,
    e.required_sample_size,
    SUM(r.users_exposed) AS total_exposed,
    ROUND(100.0 * SUM(r.users_exposed) / NULLIF(e.required_sample_size, 0), 1) AS pct_complete
FROM experiments e
JOIN experiment_variants v ON e.experiment_id = v.experiment_id
JOIN experiment_results r ON v.variant_id = r.variant_id
WHERE e.status = 'running'
GROUP BY e.experiment_id, e.name, e.required_sample_size;
