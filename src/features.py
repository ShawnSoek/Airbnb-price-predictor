"""Turn the clean SQL table into model inputs (X, y).

Run from the project root:
    uv run python -m src.features
"""
import sqlite3
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

DB_PATH = Path("data/airbnb.db")
SQL_PATH = Path("sql/features.sql")

NUMERIC = [
    "accommodates", "bedrooms", "beds", "bathrooms",
    "minimum_nights", "number_of_reviews", "review_scores_rating",
    "no_reviews", "shared_bath",
    "has_wifi", "has_ac", "has_kitchen", "has_washer", "has_elevator", "has_pool",
    "dist_center_km", "people_per_bedroom",
]
CATEGORICAL = ["room_type", "neighbourhood"]


# Get the data out of SQL

def load_features(conn: sqlite3.Connection) -> pd.DataFrame:
    """(Re)create the SQL view and read it into a DataFrame."""
    conn.executescript(SQL_PATH.read_text())
    return pd.read_sql("SELECT * FROM model_features", conn)


# Additional features

def haversine_km(lat1, lon1, lat2, lon2):
    """Distance in km between two points on Earth (works on whole columns)."""
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    a = (np.sin((lat2 - lat1) / 2) ** 2
         + np.cos(lat1) * np.cos(lat2) * np.sin((lon2 - lon1) / 2) ** 2)
    return 6371 * 2 * np.arcsin(np.sqrt(a))


def add_python_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    # "Centre" = middle of all listings -> works for any city without hard-coding
    center_lat, center_lon = df["latitude"].median(), df["longitude"].median()
    df["dist_center_km"] = haversine_km(df["latitude"], df["longitude"],
                                        center_lat, center_lon)
    df["people_per_bedroom"] = df["accommodates"] / df["bedrooms"].clip(lower=1)
    return df


# "Translator" to transform columns

def build_preprocessor() -> ColumnTransformer:
    numeric = Pipeline([
        ("fill_missing", SimpleImputer(strategy="median")),
        ("scale", StandardScaler()),
    ])
    categorical = Pipeline([
        ("fill_missing", SimpleImputer(strategy="most_frequent")),
        # one yes/no column per category; tiny neighbourhoods are grouped together
        ("one_hot", OneHotEncoder(handle_unknown="infrequent_if_exist",
                                  min_frequency=20, sparse_output=False)),
    ])
    return ColumnTransformer([
        ("num", numeric, NUMERIC),
        ("cat", categorical, CATEGORICAL),
    ])


# Split into training and test data

def get_train_test(test_size: float = 0.2, seed: int = 42):
    conn = sqlite3.connect(DB_PATH)
    try:
        df = add_python_features(load_features(conn))
        df = df.set_index("id")   # row labels = listing ids, so predictions can be traced back

        X = df[NUMERIC + CATEGORICAL]
        y = np.log1p(df["price_eur"])          # log price because of long tail in distribution
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=seed)

        # Remember which listing went where, so later phases (and the app) can query it
        split = pd.DataFrame({
            "id": df.index,
            "split": np.where(df.index.isin(X_train.index), "train", "test"),
        })
        split.to_sql("data_split", conn, if_exists="replace", index=False)
    finally:
        conn.close()

    return X_train, X_test, y_train, y_test


# ---------- Sanity check ----------

if __name__ == "__main__":
    X_train, X_test, y_train, y_test = get_train_test()
    print(f"Train: {X_train.shape}, Test: {X_test.shape}")

    pre = build_preprocessor()
    Xt_train = pre.fit_transform(X_train)   # learn ONLY from training data
    Xt_test = pre.transform(X_test)         # just apply to test data

    print(f"Columns after translation: {Xt_train.shape[1]}")
    print(f"Missing values left: {np.isnan(Xt_train).sum()} / {np.isnan(Xt_test).sum()}")
    print(f"Typical price (median, EUR): {np.expm1(y_train.median()):.0f}")
    print("\nFirst 10 model columns:")
    for name in pre.get_feature_names_out()[:10]:
        print("  ", name)