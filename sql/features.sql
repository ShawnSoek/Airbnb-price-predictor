-- Feature view
-- Everything here is a simple row-by-row rule and does not include
-- learning from the data.
--
-- Run from the project root:
--     sqlite3 -header -column data/airbnb.db < sql/features.sql

DROP VIEW IF EXISTS model_features;

CREATE VIEW model_features AS
SELECT
    id,
    price_eur,

    -- categories
    room_type,
    property_type,
    neighbourhood_cleansed AS neighbourhood,

    -- size of the place
    accommodates,
    bedrooms,
    beds,

    -- bathrooms_text looks like "1.5 baths", "2 shared baths" or "Half-bath".
    -- CAST keeps the leading number; "half-bath" has no number, so it gets 0.5.
    CASE
        WHEN bathrooms_text LIKE '%half-bath%' THEN 0.5
        ELSE CAST(bathrooms_text AS REAL)
    END AS bathrooms,
    CASE WHEN bathrooms_text LIKE '%shared%' THEN 1 ELSE 0 END AS shared_bath,

    -- location
    latitude,
    longitude,

    -- booking rules and reputation
    minimum_nights,
    availability_365,
    number_of_reviews,
    review_scores_rating,
    CASE WHEN number_of_reviews = 0 THEN 1 ELSE 0 END AS no_reviews,

    -- amenities: the raw column is one long text, so we count the items (= commas + 1)
    -- and pull out a few yes/no flags
    LENGTH(amenities) - LENGTH(REPLACE(amenities, ',', '')) + 1 AS amenity_count,
    CASE WHEN amenities LIKE '%Wifi%'             THEN 1 ELSE 0 END AS has_wifi,
    CASE WHEN amenities LIKE '%Air conditioning%' THEN 1 ELSE 0 END AS has_ac,
    CASE WHEN amenities LIKE '%Kitchen%'          THEN 1 ELSE 0 END AS has_kitchen,
    CASE WHEN amenities LIKE '%Washer%'           THEN 1 ELSE 0 END AS has_washer,
    CASE WHEN amenities LIKE '%Elevator%'         THEN 1 ELSE 0 END AS has_elevator,
    CASE WHEN amenities LIKE '%Pool%'             THEN 1 ELSE 0 END AS has_pool

FROM listings_clean;


-- ============================================================
-- Sanity checks
-- ============================================================

-- 1. The view should have exactly as many rows as the clean table
SELECT
    (SELECT COUNT(*) FROM listings_clean) AS rows_clean,
    (SELECT COUNT(*) FROM model_features) AS rows_features;

-- 2. Missing values per column (Python fills these later, but they shouldn't be huge;
--    missing ratings are normal for listings with no reviews)
SELECT
    SUM(price_eur IS NULL)            AS price_missing,      -- must be 0
    SUM(room_type IS NULL)            AS room_type_missing,
    SUM(property_type IS NULL)        AS property_type_missing,
    SUM(neighbourhood IS NULL)        AS neighbourhood_missing,
    SUM(bedrooms IS NULL)             AS bedrooms_missing,   -- must be 0 (dropped in cleaning)
    SUM(beds IS NULL)                 AS beds_missing,
    SUM(bathrooms IS NULL)            AS bathrooms_missing,
    SUM(review_scores_rating IS NULL) AS rating_missing,
    SUM(no_reviews)                   AS listings_without_reviews
FROM model_features;

-- 3. Value ranges should look plausible
SELECT
    MIN(price_eur)        AS price_min,              -- should be >= 10
    MAX(price_eur)        AS price_max,              -- should be <= 500
    ROUND(AVG(price_eur)) AS price_avg,
    MAX(accommodates)     AS max_guests,
    MAX(bedrooms)         AS max_bedrooms,
    MAX(bathrooms)        AS max_bathrooms,
    MAX(minimum_nights)   AS max_min_nights,
    MIN(availability_365) AS availability_min,       -- should be >= 0
    MAX(availability_365) AS availability_max,       -- should be <= 365
    MAX(amenity_count)    AS max_amenities
FROM model_features;

-- 4. Bathroom text -> number: spot-check the translation
SELECT
    l.bathrooms_text,
    f.bathrooms,
    f.shared_bath,
    COUNT(*) AS listings
FROM listings_clean l
JOIN model_features f ON f.id = l.id
GROUP BY l.bathrooms_text
ORDER BY listings DESC
LIMIT 10;

-- 5. Share of listings per yes/no flag
--    (exactly 0.00 or 1.00 usually means the LIKE pattern doesn't match the data)
SELECT
    ROUND(AVG(shared_bath), 2)  AS shared_bath,
    ROUND(AVG(has_wifi), 2)     AS wifi,
    ROUND(AVG(has_ac), 2)       AS ac,
    ROUND(AVG(has_kitchen), 2)  AS kitchen,
    ROUND(AVG(has_washer), 2)   AS washer,
    ROUND(AVG(has_elevator), 2) AS elevator,
    ROUND(AVG(has_pool), 2)     AS pool
FROM model_features;

-- 6. Most common property types (rare ones get grouped together in Python)
SELECT property_type, COUNT(*) AS listings
FROM model_features
GROUP BY property_type
ORDER BY listings DESC
LIMIT 10;

-- 7. Room types: how many and how expensive (entire homes should cost the most)
SELECT
    room_type,
    COUNT(*)              AS listings,
    ROUND(AVG(price_eur)) AS avg_price
FROM model_features
GROUP BY room_type
ORDER BY listings DESC;