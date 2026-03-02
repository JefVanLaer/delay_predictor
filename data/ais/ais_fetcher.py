"""
This script sources AIS data from the NOAA HTTPS server. It works as follows:
1. For each date in the requested range, construct the download URL.
2. Download the daily .csv.zst file (Zstandard-compressed CSV).
3. Stream-decompress and filter for cargo vessels (vessel_type 70-79).
4. Save the filtered data as a Parquet file for further use.

Usage:
    python ais_fetcher.py --start 2025-01-01 --end 2025-01-31
    python ais_fetcher.py --start 2025-01-01 --end 2025-12-31 --output data/ais
"""

import sys
sys.path.insert(0, '..')
import argparse
from datetime import date, timedelta
from pathlib import Path

import pandas as pd
import requests
import zstandard

BASE_URL = "https://coast.noaa.gov/htdata/CMSP/AISDataHandler/{year}/"

CARGO_TYPE_MIN = 70
CARGO_TYPE_MAX = 79

# Raw NOAA column names → project column names (no-op if already lowercase)
_COLUMN_RENAME = {
    'MMSI':             'mmsi',
    'BaseDateTime':     'base_date_time',
    'LAT':              'latitude',
    'LON':              'longitude',
    'SOG':              'sog',
    'COG':              'cog',
    'Heading':          'heading',
    'VesselName':       'vessel_name',
    'IMO':              'imo',
    'CallSign':         'call_sign',
    'VesselType':       'vessel_type',
    'Status':           'status',
    'Length':           'length',
    'Width':            'width',
    'Draft':            'draft',
    'Cargo':            'cargo',
    'TransceiverClass': 'transceiver',
}


def _parse_args():
    parser = argparse.ArgumentParser(
        description="Download and filter NOAA AIS data for cargo vessels."
    )
    parser.add_argument(
        "--start", required=True, metavar="YYYY-MM-DD",
        help="First date to fetch (inclusive)",
    )
    parser.add_argument(
        "--end", required=True, metavar="YYYY-MM-DD",
        help="Last date to fetch (inclusive)",
    )
    parser.add_argument(
        "--output", default="data/ais",
        help="Output directory for filtered CSV files (default: data/ais)",
    )
    return parser.parse_args()


def _date_range(start: date, end: date):
    current = start
    while current <= end:
        yield current
        current += timedelta(days=1)


def _filter_cargo(df: pd.DataFrame) -> pd.DataFrame:
    df = df.rename(columns=_COLUMN_RENAME)
    df.columns = [c.lower() for c in df.columns]

    if 'vessel_type' not in df.columns:
        raise ValueError(
            f"'vessel_type' column not found. Available columns: {list(df.columns)}"
        )

    df['vessel_type'] = pd.to_numeric(df['vessel_type'], errors='coerce')
    return df[
        (df['vessel_type'] >= CARGO_TYPE_MIN) & (df['vessel_type'] <= CARGO_TYPE_MAX)
    ].copy()


def main():
    args = _parse_args()
    start = date.fromisoformat(args.start)
    end   = date.fromisoformat(args.end)

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    dctx = zstandard.ZstdDecompressor()

    for current_date in _date_range(start, end):
        filename    = f"ais-{current_date.isoformat()}.csv.zst"
        output_path = output_dir / f"ais-{current_date.isoformat()}.parquet"
        url         = BASE_URL.format(year=current_date.year) + filename

        if output_path.exists():
            print(f"[skip]     {filename} — output already exists")
            continue

        print(f"[download] {filename} ...", end=" ", flush=True)
        response = requests.get(url, stream=True)

        if response.status_code == 404:
            print("not found, skipping")
            continue
        response.raise_for_status()

        response.raw.decode_content = True
        with dctx.stream_reader(response.raw) as reader:
            df = pd.read_csv(reader, low_memory=False)

        compressed_mb = int(response.headers.get('Content-Length', 0)) / 1_048_576
        print(f"{compressed_mb:.0f} MB compressed, {len(df):,} total rows")

        df = _filter_cargo(df)

        if df.empty:
            print(f"[skip]     No cargo vessels found in {filename}\n")
            continue

        df.to_parquet(output_path, index=False)
        print(f"[saved]    {output_path}  ({len(df):,} cargo vessel rows)\n")

    print("Done.")


if __name__ == "__main__":
    main()
