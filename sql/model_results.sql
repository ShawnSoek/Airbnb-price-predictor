-- Look at the model's results with SQL.
-- Run after training:
--     sqlite3 -header -column data/airbnb.db < sql/model_results.sql

-- 1. All training runs, newest first, best model on top within each run
SELECT
    run_at,
    model,
    ROUND(mae_eur, 1)          AS avg_miss_eur,
    ROUND(median_pct_error, 1) AS typical_miss_pct,
    ROUND(r2, 2)               AS r2
FROM model_runs
ORDER BY run_at DESC, mae_eur;

-- 2. Where is the model good or bad? Error per neighbourhood
--    bias > 0: model guesses too high there; bias < 0: too low
SELECT
    f.neighbourhood,
    COUNT(*)                        AS listings,
    ROUND(AVG(p.actual_eur))        AS avg_price,
    ROUND(AVG(ABS(p.error_eur)), 1) AS avg_miss_eur,
    ROUND(AVG(p.error_eur), 1)      AS bias_eur
FROM predictions p
JOIN model_features f ON f.id = p.id
GROUP BY f.neighbourhood
HAVING COUNT(*) >= 20
ORDER BY avg_miss_eur DESC;

-- 3. Error per room type
SELECT
    f.room_type,
    COUNT(*)                        AS listings,
    ROUND(AVG(p.actual_eur))        AS avg_price,
    ROUND(AVG(ABS(p.error_eur)), 1) AS avg_miss_eur
FROM predictions p
JOIN model_features f ON f.id = p.id
GROUP BY f.room_type
ORDER BY listings DESC;

-- 4. Possible bargains: listings much cheaper than the model thinks they should be
SELECT
    p.id,
    f.neighbourhood,
    f.room_type,
    f.accommodates,
    p.actual_eur,
    p.predicted_eur,
    p.error_eur AS cheaper_than_expected_by
FROM predictions p
JOIN model_features f ON f.id = p.id
ORDER BY p.error_eur DESC
LIMIT 10;