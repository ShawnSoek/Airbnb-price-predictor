"""Phase 5: the Barcelona Airbnb price app.

Run from the project root:
    uv run streamlit run app.py
"""
import sqlite3

import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

from src.features import (BEACH_POINTS, CATEGORICAL, DB_PATH, LANDMARKS, NUMERIC,
                          add_python_features, city_center)
from src.train import MODEL_PATH

st.set_page_config(page_title="Barcelona Airbnb Prices", layout="wide")


# ---------- Data access: every number in the app comes from SQL ----------

@st.cache_data
def query(sql: str, params: tuple = ()) -> pd.DataFrame:
    # read-only connection: the app can never change the database
    with sqlite3.connect(f"file:{DB_PATH}?mode=ro", uri=True) as conn:
        return pd.read_sql(sql, conn, params=params)


@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)


@st.cache_data
def get_center():
    # must be the same centre the model was trained with
    return city_center(query("SELECT latitude, longitude FROM model_features"))


def latest_scores() -> pd.DataFrame:
    return query("""
        SELECT model, mae_eur, median_pct_error, r2
        FROM model_runs
        WHERE run_at = (SELECT MAX(run_at) FROM model_runs)
        ORDER BY mae_eur
    """)


model = load_model()
best = latest_scores().iloc[0]

st.title("What should an Airbnb in Barcelona cost?")
st.caption(f"A {best['model']} model trained on real Inside Airbnb listings. "
           f"Its estimates are typically within ±{best['median_pct_error']:.0f}% of the real price.")

tab_estimate, tab_map, tab_cheap, tab_model = st.tabs(
    ["Estimate a price", "Price map", "Cheap for what they offer", "How good is the model?"])


# ---------- Tab 1: estimate a price ----------

with tab_estimate:
    hoods = query("""
        SELECT neighbourhood, AVG(latitude) AS lat, AVG(longitude) AS lon, COUNT(*) AS n
        FROM model_features
        GROUP BY neighbourhood
        HAVING COUNT(*) >= 20
        ORDER BY neighbourhood
    """)
    typical = query("""
        SELECT AVG(minimum_nights)   AS min_nights,
               AVG(availability_365) AS availability,
               AVG(amenity_count)    AS amenities,
               AVG(number_of_reviews) AS reviews,
               AVG(review_scores_rating) AS rating
        FROM model_features
    """).iloc[0]

    left, right = st.columns([1, 1], gap="large")

    with left:
        st.subheader("Your place")
        c1, c2 = st.columns(2)
        hood = c1.selectbox("Neighbourhood", hoods["neighbourhood"])
        room_types = query("SELECT room_type FROM model_features GROUP BY room_type "
                           "ORDER BY COUNT(*) DESC")["room_type"]
        room_type = c2.selectbox("Room type", room_types)
        property_types = query("""
            SELECT property_type FROM model_features WHERE room_type = ?
            GROUP BY property_type ORDER BY COUNT(*) DESC LIMIT 15
        """, (room_type,))["property_type"]
        property_type = st.selectbox("Property type", property_types)

        c1, c2, c3 = st.columns(3)
        guests = c1.number_input("Guests", 1, 16, 2)
        bedrooms = c2.number_input("Bedrooms", 1, 10, 1)
        beds = c3.number_input("Beds", 1, 16, max(1, guests // 2))
        c1, c2 = st.columns(2)
        bathrooms = c1.number_input("Bathrooms", 0.5, 8.0, 1.0, step=0.5)
        shared_bath = c2.checkbox("Shared bathroom")

        st.write("Amenities")
        a1, a2, a3 = st.columns(3)
        wifi = a1.checkbox("Wifi", True)
        kitchen = a1.checkbox("Kitchen", True)
        ac = a2.checkbox("Air conditioning", True)
        washer = a2.checkbox("Washer", True)
        elevator = a3.checkbox("Elevator")
        pool = a3.checkbox("Pool")

        with st.expander("Reviews and booking rules (preset to typical values)"):
            reviews = st.number_input("Number of reviews", 0, 2000, int(typical["reviews"]))
            rating = st.slider("Rating", 1.0, 5.0, round(float(typical["rating"]), 1), 0.1)
            min_nights = st.number_input("Minimum nights", 1, 365, max(1, int(typical["min_nights"])))
            availability = st.slider("Days bookable per year", 0, 365, int(typical["availability"]))
            amenities = st.number_input("Total number of amenities", 1, 150, int(typical["amenities"]))

    # one row with exactly the columns the model was trained on
    spot = hoods.set_index("neighbourhood").loc[hood]
    row = pd.DataFrame([{
        "room_type": room_type, "property_type": property_type, "neighbourhood": hood,
        "accommodates": guests, "bedrooms": bedrooms, "beds": beds, "bathrooms": bathrooms,
        "shared_bath": int(shared_bath),
        "latitude": spot["lat"], "longitude": spot["lon"],
        "minimum_nights": min_nights, "availability_365": availability,
        "number_of_reviews": reviews, "review_scores_rating": rating if reviews > 0 else np.nan,
        "no_reviews": int(reviews == 0), "amenity_count": amenities,
        "has_wifi": int(wifi), "has_ac": int(ac), "has_kitchen": int(kitchen),
        "has_washer": int(washer), "has_elevator": int(elevator), "has_pool": int(pool),
    }])
    row = add_python_features(row, center=get_center())
    price = float(np.expm1(model.predict(row[NUMERIC + CATEGORICAL]))[0])
    margin = best["median_pct_error"] / 100

    with right:
        st.subheader("Estimated price")
        st.metric("per night", f"{price:,.0f} €")
        st.caption(f"Usual range: {price * (1 - margin):,.0f} – {price * (1 + margin):,.0f} €. "
                   "The location is taken as the middle of the neighbourhood.")

        similar = query("""
            SELECT price_eur FROM model_features
            WHERE neighbourhood = ? AND room_type = ? AND accommodates BETWEEN ? AND ?
        """, (hood, room_type, guests - 1, guests + 1))
        st.subheader("Similar listings nearby")
        if len(similar) >= 5:
            st.caption(f"{len(similar)} listings in {hood}, same room type, "
                       f"{guests - 1}–{guests + 1} guests. The line is your estimate.")
            fig = px.histogram(similar, x="price_eur", nbins=25,
                               labels={"price_eur": "Price per night (€)"})
            fig.add_vline(x=price, line_width=3, line_color="#e4572e")
            fig.update_layout(height=320, margin=dict(l=0, r=0, t=10, b=0),
                              yaxis_title="Listings", bargap=0.05)
            st.plotly_chart(fig, width="stretch")
        else:
            st.info("Fewer than 5 similar listings here, so there is nothing useful to compare with.")


# ---------- Tab 2: price map ----------

with tab_map:
    mode = st.radio("Colour by", ["Price per night", "Real price vs. model estimate"],
                    horizontal=True)

    if mode == "Price per night":
        df = query("SELECT latitude, longitude, price_eur, neighbourhood, room_type "
                   "FROM model_features")
        fig = px.scatter_map(
            df, lat="latitude", lon="longitude", color="price_eur",
            color_continuous_scale="Viridis",
            range_color=(df["price_eur"].quantile(0.05), df["price_eur"].quantile(0.95)),
            hover_data={"neighbourhood": True, "room_type": True, "price_eur": ":.0f",
                        "latitude": False, "longitude": False},
            labels={"price_eur": "€ / night"})
    else:
        st.caption("Only listings the model never saw while learning, so the comparison is fair. "
                   "Blue = cheaper than the model expects, red = more expensive.")
        df = query("""
            SELECT f.latitude, f.longitude, f.neighbourhood, f.room_type,
                   p.actual_eur, p.predicted_eur,
                   ROUND((p.actual_eur / p.predicted_eur - 1) * 100) AS diff_pct
            FROM predictions p
            JOIN model_features f ON f.id = p.id
        """)
        fig = px.scatter_map(
            df, lat="latitude", lon="longitude", color="diff_pct",
            color_continuous_scale="RdBu_r", range_color=(-50, 50),
            hover_data={"neighbourhood": True, "room_type": True, "actual_eur": ":.0f",
                        "predicted_eur": ":.0f", "latitude": False, "longitude": False},
            labels={"diff_pct": "% vs. estimate", "actual_eur": "real €",
                    "predicted_eur": "estimated €"})

    fig.update_traces(marker=dict(size=6, opacity=0.75))
    marks = pd.DataFrame(
        [(name.replace("_", " ").title(), lat, lon) for name, (lat, lon) in LANDMARKS.items()]
        + [("Beach", lat, lon) for lat, lon in BEACH_POINTS],
        columns=["name", "lat", "lon"])
    fig.add_scattermap(lat=marks["lat"], lon=marks["lon"], text=marks["name"],
                       mode="markers+text", textposition="top right",
                       marker=dict(size=11, color="black"), name="Landmarks",
                       hoverinfo="text", showlegend=False)
    fig.update_layout(map_style="carto-positron", map_zoom=11.8,
                      map_center=dict(lat=41.395, lon=2.165),
                      height=650, margin=dict(l=0, r=0, t=0, b=0))
    st.plotly_chart(fig, width="stretch")


# ---------- Tab 3: cheap for what they offer ----------

with tab_cheap:
    st.write("Listings whose real price is far below what the model expects for their size, "
             "location and features. Some are genuine deals, others are cheap for reasons the "
             "data can't show (photos, condition, an outdated price), so have a look before booking.")
    c1, c2, c3 = st.columns(3)
    rt = c1.selectbox("Room type", ["Any"] + list(room_types), key="cheap_rt")
    min_guests = c2.number_input("At least this many guests", 1, 16, 2)
    max_price = c3.number_input("Max price per night (€)", 10, 500, 150, step=10)

    cheap = query("""
        SELECT 'https://www.airbnb.com/rooms/' || p.id      AS link,
               f.neighbourhood, f.room_type, f.accommodates AS guests,
               ROUND(p.actual_eur)                          AS price,
               ROUND(p.predicted_eur)                       AS model_estimate,
               ROUND((1 - p.actual_eur / p.predicted_eur) * 100) AS pct_below
        FROM predictions p
        JOIN model_features f ON f.id = p.id
        WHERE (? = 'Any' OR f.room_type = ?)
          AND f.accommodates >= ?
          AND p.actual_eur <= ?
        ORDER BY pct_below DESC
        LIMIT 25
    """, (rt, rt, min_guests, max_price))

    st.dataframe(
        cheap, hide_index=True, width="stretch",
        column_config={
            "link": st.column_config.LinkColumn("Listing", display_text="Open on Airbnb"),
            "neighbourhood": "Neighbourhood",
            "room_type": "Room type",
            "guests": "Guests",
            "price": st.column_config.NumberColumn("Price €", format="%d"),
            "model_estimate": st.column_config.NumberColumn("Model estimate €", format="%d"),
            "pct_below": st.column_config.ProgressColumn(
                "Below estimate", format="%d%%", min_value=0, max_value=100),
        })
    st.caption("The data is a snapshot, so some links may no longer work.")


# ---------- Tab 4: how good is the model? ----------

with tab_model:
    scores = latest_scores()
    c1, c2, c3 = st.columns(3)
    c1.metric("Average miss", f"{best['mae_eur']:.0f} €")
    c2.metric("Typical miss", f"{best['median_pct_error']:.0f} %")
    c3.metric("Price differences explained (R²)", f"{best['r2']:.2f}")

    left, right = st.columns(2, gap="large")
    with left:
        st.subheader("All models compared")
        fig = px.bar(scores.sort_values("mae_eur", ascending=False),
                     x="mae_eur", y="model", orientation="h", text_auto=".0f",
                     labels={"mae_eur": "Average miss (€) – lower is better", "model": ""})
        fig.update_layout(height=300, margin=dict(l=0, r=0, t=10, b=0))
        st.plotly_chart(fig, width="stretch")

    with right:
        st.subheader("Miss per neighbourhood")
        hood_err = query("""
            SELECT f.neighbourhood,
                   COUNT(*) AS listings,
                   ROUND(AVG(ABS(p.error_eur) / p.actual_eur) * 100, 1) AS avg_miss_pct
            FROM predictions p
            JOIN model_features f ON f.id = p.id
            GROUP BY f.neighbourhood
            HAVING COUNT(*) >= 20
            ORDER BY avg_miss_pct
        """)
        fig = px.bar(hood_err, x="avg_miss_pct", y="neighbourhood", orientation="h",
                     hover_data=["listings"],
                     labels={"avg_miss_pct": "Average miss (%)", "neighbourhood": ""})
        fig.update_layout(height=max(300, 22 * len(hood_err)), margin=dict(l=0, r=0, t=10, b=0))
        st.plotly_chart(fig, width="stretch")

    st.subheader("Real vs. estimated price")
    preds = query("SELECT actual_eur, predicted_eur FROM predictions")
    fig = px.scatter(preds, x="actual_eur", y="predicted_eur", opacity=0.35,
                     labels={"actual_eur": "Real price (€)", "predicted_eur": "Estimated price (€)"})
    top = float(max(preds["actual_eur"].max(), preds["predicted_eur"].max()))
    fig.add_shape(type="line", x0=0, y0=0, x1=top, y1=top, line=dict(dash="dash", color="gray"))
    fig.update_layout(height=450, margin=dict(l=0, r=0, t=10, b=0))
    st.plotly_chart(fig, width="stretch")
    st.caption("Each dot is a listing the model never saw while learning. "
               "Dots on the dashed line were estimated perfectly.")