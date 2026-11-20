# Barcelona Airbnb Price Predictor

How much should a night in a Barcelona Airbnb cost? I trained a few classic machine-learning models on real listings to find out, and built a small web app around the best one. You describe a place and get a price estimate. You can also explore prices across the city on a map, or look for listings that seem cheap for what they offer.

<img width="959" height="471" alt="image" src="https://github.com/user-attachments/assets/3714e1fc-9225-404a-b563-317f250a3baf" />
<img width="959" height="431" alt="image" src="https://github.com/user-attachments/assets/1462d4d6-978b-4ddf-904e-2158645b4b23" />


## The app

The app has four tabs. The first one is the actual price estimator: pick a neighbourhood, room type, size and a few amenities, and it tells you what the place should cost per night. Next to the estimate you see the real prices of similar listings nearby, so you can judge for yourself whether the number makes sense.

The map shows every listing in the city, coloured by price. You can switch it to show how far each listing's real price is from what the model expected, which makes it easy to spot areas where the model tends to guess too high or too low.

The third tab lists places that are priced well below the model's estimate. Some of them are probably good deals; others are cheap for reasons the data can't capture, so treat it as a starting point, not a recommendation.

The last tab shows how well the model actually performs, broken down by neighbourhood.

## Results

I compared four models on 20% of the listings, which I kept aside and never used for training:

| Model | Average miss | Typical miss | R² |
|---|---|---|---|
| Baseline (always guesses the typical price) | 93.5 € | 43.7 % | -0.01 |
| Linear regression | 46.5 € | 21.3 % | 0.65 |
| Random forest | 37.4 € | 15.4 % | 0.75 |
| Gradient boosting | 36.9 € | 15.8 % | 0.77 |

"Average miss" is how many euros the estimate is off on average. "Typical miss" is the median error in percent, which is less affected by a few extreme listings. R² is the share of price differences the model can explain.

The baseline is there as a sanity check: a model that can't beat "always guess the typical price" hasn't learned anything. All three real models beat it comfortably, and the two tree-based ones clearly beat linear regression. That makes sense, since prices don't work like a simple sum. A fifth guest or a central location is worth more in some places than in others.

Two decisions made the biggest difference. First, the models predict the logarithm of the price instead of the price itself, so being 20 € off on a 50 € room counts as a bigger mistake than on a 400 € flat. Second, location matters a lot, so besides the neighbourhood I gave the model distances to the city centre, a few landmarks like Sagrada Família and Park Güell, and the nearest beach.

## How it's built

The project uses SQLite for the data and Python (scikit-learn, Streamlit) for the rest:

```
Inside Airbnb CSV
   → listings_raw          raw data loaded into SQLite
   → clean_listings.sql    prices converted to euros, outliers and incomplete rows removed
   → features.sql          amenity flags, bathroom counts, review info
   → features.py           distances to landmarks, train/test split
   → train.py              trains four models, saves the best one and all scores
   → app.py                the Streamlit app
```

I tried to keep a clear line between the two languages. SQL handles anything that's a fixed rule per row, like cleaning or turning the amenities text into yes/no columns, and it's also where I analyse the results afterwards. Python handles everything that has to learn from the data. That split also makes it easier to be sure the test data never influences training.

```
├── app.py
├── sql/
│   ├── clean_listings.sql
│   ├── features.sql        (includes sanity checks at the end)
│   └── model_results.sql
├── src/
│   ├── features.py
│   └── train.py
├── data/                   database, not in Git
└── models/                 trained model, not in Git
```

## Running it yourself

You'll need [uv](https://docs.astral.sh/uv/) and `sqlite3`. Clone the repository and install the dependencies:

```bash
git clone https://github.com/ShawnSoek/Airbnb-price-predictor
cd Airbnb-price-predictor
uv sync
```

Then download the data. On [Inside Airbnb](https://insideairbnb.com/get-the-data/), find Barcelona and get the detailed `listings.csv.gz`.
I used the snapshot from 17 September 2026.

From there, it's one command per step:

```bash
# load the CSV into SQLite
uv run python src/load_db.py

# clean the data and build the features
sqlite3 data/airbnb.db < sql/clean_listings.sql
sqlite3 -header -column data/airbnb.db < sql/features.sql

# train the models
uv run python -m src.train

# start the app and open http://localhost:8501
uv run streamlit run app.py
```

If you want to dig into the results with SQL, `sql/model_results.sql` has a few queries for that, including the error per neighbourhood.

## Limitations

The model can only work with what's in the data. Photos, the exact size of the flat, how recently it was renovated or whether it has a view all affect the price, but none of that is available, so some listings will always be far off.

The data is also a snapshot. Prices are the ones from the day the listings were collected, and some of the places may not exist anymore.

In the app, your place is assumed to be in the middle of the neighbourhood you pick, not at an exact address.

Finally, the data mixes short holiday stays with monthly rentals. Listings with a long minimum stay charge much less per night, so the "minimum nights" setting has a big influence on the estimate.

## Data

The listing data comes from [Inside Airbnb](https://insideairbnb.com/)
