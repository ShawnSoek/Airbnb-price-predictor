"""
Load the raw Barcelona listings into a SQLite database.

Run from the project root:
    uv run python src/load_db.py
"""

import sqlite3
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CSV_PATH = PROJECT_ROOT / "data" / "raw" / "listings.csv.gz"
DB_PATH = PROJECT_ROOT / "data" / "airbnb.db"

# Only the columns we expect to need. The other ~75 are ignored.
COLUMNS = [
    "id",
    "neighbourhood_cleansed",
    "latitude",
    "longitude",
    "property_type",
    "room_type",
    "accommodates",
    "bathrooms_text",
    "bedrooms",
    "beds",
    "amenities",
    "price",                
    "minimum_nights",
    "availability_365",
    "number_of_reviews",
    "review_scores_rating",
]


def main() -> None:
    df = pd.read_csv(CSV_PATH, usecols=COLUMNS)
    print(f"Read {len(df)} rows from {CSV_PATH.name}")

    with sqlite3.connect(DB_PATH) as conn:
        # if_exists="replace": running the script again rebuilds the table
        df.to_sql("listings_raw", conn, if_exists="replace", index=False)

        count = conn.execute("SELECT COUNT(*) FROM listings_raw").fetchone()[0]
        print(f"Table listings_raw in {DB_PATH.name} now has {count} rows")


if __name__ == "__main__":
    main()