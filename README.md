# Airbnb-Price-Predictor

Predicts a fair nightly price for Airbnb listings in Barcelona.

**Status:** work in progress

## Planned features

- **Price Estimator:** enter apartment details, get an estimated nightly price
- **Price Map:** interactive map of Barcelona colored by price
- **Deals & Rip-offs:** listings priced far below or above the model's estimate

## Tech stack

- Python, pandas, scikit-learn
- SQLite for storing and cleaning the data
- Streamlit and Plotly for the app and the charts
- uv for package management

## Setup

Requires [uv](https://docs.astral.sh/uv/).

```bash
git clone https://github.com/YOUR-USERNAME/Airbnb-price-predictor.git
cd Airbnb-price-predictor
uv sync
```

## Data

Listing data comes from [Inside Airbnb](https://insideairbnb.com/get-the-data/).