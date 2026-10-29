"""
Download the Barcelona Airbnb data from Inside Airbnb.

Run from the project root:
    uv run python src/download_data.py
"""

import argparse
import urllib.request
from pathlib import Path

# Latest Barcelona snapshot listed on insideairbnb.com/get-the-data
DEFAULT_DATE = "2026-09-17"
BASE_URL = "https://data.insideairbnb.com/spain/catalonia/barcelona/{date}"

# Only the files we actually need:
#   listings.csv.gz        -> all details per listing (the training data for the model)
#   neighbourhoods.geojson -> neighbourhood shapes (for the map later on)
FILES = {
    "listings.csv.gz": "data/listings.csv.gz",
    "neighbourhoods.geojson": "visualisations/neighbourhoods.geojson",
}

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = PROJECT_ROOT / "data" / "raw"


def download(url: str, target: Path) -> None:
    """Download one file, skipping it if it already exists."""
    if target.exists():
        print(f"  ✓ {target.name} already there, skipping")
        return

    print(f"  ↓ {url}")
    tmp = target.with_suffix(target.suffix + ".part")
    # A browser-like User-Agent avoids occasional 403 errors
    request = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(request) as response, open(tmp, "wb") as f:
        f.write(response.read())
    tmp.rename(target)  # only rename once the download finished completely
    print(f"  ✓ saved {target.relative_to(PROJECT_ROOT)} ({target.stat().st_size / 1e6:.1f} MB)")


def main() -> None:
    parser = argparse.ArgumentParser(description="Download Barcelona Airbnb data")
    parser.add_argument("--date", default=DEFAULT_DATE, help="snapshot date, YYYY-MM-DD")
    args = parser.parse_args()

    RAW_DIR.mkdir(parents=True, exist_ok=True)
    base = BASE_URL.format(date=args.date)

    print(f"Downloading Barcelona snapshot {args.date} into {RAW_DIR.relative_to(PROJECT_ROOT)}/")
    for filename, remote_path in FILES.items():
        download(f"{base}/{remote_path}", RAW_DIR / filename)

    # Remember which snapshot we used (useful for the README later)
    (RAW_DIR / "SNAPSHOT_DATE.txt").write_text(args.date + "\n")
    print("Done.")


if __name__ == "__main__":
    main()