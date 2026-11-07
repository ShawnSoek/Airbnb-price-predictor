-- Build the cleaned table listings_clean from listings_raw.
--
-- Run from the project root:
--     sqlite3 data/airbnb.db < sql/clean_listings.sql
--
-- Rules:
--   1. price text like "$1,200.00" becomes a real number (price_eur)
--   2. listings without a price are dropped
--   3. only prices between 10 and 500 EUR per night are kept
--   4. listings with an unknown number of bedrooms are dropped
--
-- listings_raw stays untouched, so we can always rebuild with other limits.

DROP TABLE IF EXISTS listings_clean;

CREATE TABLE listings_clean AS
SELECT *
FROM (
    SELECT
        id,
        neighbourhood_cleansed,
        latitude,
        longitude,
        property_type,
        room_type,
        accommodates,
        bathrooms_text,
        bedrooms,
        beds,
        amenities,
        minimum_nights,
        availability_365,
        number_of_reviews,
        review_scores_rating,
        CAST(REPLACE(REPLACE(price, '$', ''), ',', '') AS REAL) AS price_eur
    FROM listings_raw
    WHERE price IS NOT NULL
)
WHERE price_eur BETWEEN 10 AND 500
  AND bedrooms IS NOT NULL;

-- Quick check: how many rows are left, and what do the prices look like?
SELECT COUNT(*)                  AS rows_left,
       MIN(price_eur)            AS min_price,
       MAX(price_eur)            AS max_price,
       ROUND(AVG(price_eur), 1)  AS avg_price
FROM listings_clean;