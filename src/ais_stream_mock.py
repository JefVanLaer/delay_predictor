"""AIS Stream Mock — replays AIS data from a Parquet file as timed JSON messages.

Each row in the Parquet file is emitted as a JSON object to stdout at a pace
controlled by the timestamps in the data and a configurable speed_factor.

Programmatic usage::

    from src.ais_stream_mock import AISStreamMock

    mock = AISStreamMock("data/ais/ais-2025-01-01.parquet", speed_factor=60)
    for message in mock.stream():
        process(message)          # dict with AIS fields

Script usage (newline-delimited JSON on stdout)::

    python src/ais_stream_mock.py \\
        --parquet data/ais/ais-2025-01-01.parquet \\
        --speed-factor 60
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

import pandas as pd


class _AISEncoder(json.JSONEncoder):
    """JSON encoder that coerces pandas/numpy scalars to Python natives."""

    def default(self, obj):
        if hasattr(obj, "item"):   # numpy int64, float64, etc.
            return obj.item()
        if pd.isna(obj):           # pandas NA, NaT, float NaN
            return None
        return super().default(obj)


class AISStreamMock:
    """
    Replays AIS records from a Parquet file as a timed stream of JSON messages.

    Records are sorted by *timestamp_col* and emitted in order. Between
    consecutive records the streamer sleeps for the real-time equivalent of
    the gap between their timestamps, scaled by *speed_factor*.

    Parameters
    ----------
    parquet_path : str or Path
        Path to the Parquet file containing AIS records.
    speed_factor : float
        Playback speed multiplier.  ``1.0`` = real-time (one AIS-second per
        real second); ``60.0`` = one AIS-minute per real second, etc.
    timestamp_col : str
        Name of the datetime column used for timing.  Defaults to
        ``'base_date_time'`` to match the NOAA AIS Parquet schema produced
        by ``data/ais/ais_fetcher.py``.
    """

    def __init__(
        self,
        parquet_path,
        speed_factor: float = 1.0,
        timestamp_col: str = "base_date_time",
    ):
        self.parquet_path = Path(parquet_path)
        self.speed_factor = float(speed_factor)
        self.timestamp_col = timestamp_col

    def stream(self):
        """
        Yield AIS records as dicts in timestamp order, sleeping between them.

        Timestamps are converted to ISO-8601 strings so every yielded dict
        is directly JSON-serialisable.

        Yields
        ------
        dict
            One AIS record per yield.
        """
        df = pd.read_parquet(self.parquet_path)
        df[self.timestamp_col] = pd.to_datetime(df[self.timestamp_col])
        df = df.sort_values(self.timestamp_col).reset_index(drop=True)

        prev_ts = None
        for _, row in df.iterrows():
            ts = row[self.timestamp_col]

            if prev_ts is not None:
                delta_s = (ts - prev_ts).total_seconds()
                sleep_s = delta_s / self.speed_factor
                if sleep_s > 0:
                    time.sleep(sleep_s)

            msg = row.to_dict()
            msg[self.timestamp_col] = ts.isoformat()
            yield msg

            prev_ts = ts

    def run(self, output=None):
        """
        Write all records as newline-delimited JSON to *output*.

        Parameters
        ----------
        output : file-like, optional
            Destination stream.  Defaults to ``sys.stdout``.
        """
        if output is None:
            output = sys.stdout
        for msg in self.stream():
            output.write(json.dumps(msg, cls=_AISEncoder) + "\n")
            output.flush()


# ---------------------------------------------------------------------------
# CLI entry-point
# ---------------------------------------------------------------------------

def _parse_args():
    parser = argparse.ArgumentParser(
        description="Replay AIS data from a Parquet file as JSON messages.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--parquet",
        required=True,
        metavar="PATH",
        help="Path to the Parquet file containing AIS records.",
    )
    parser.add_argument(
        "--speed-factor",
        type=float,
        default=1.0,
        metavar="N",
        help=(
            "Playback speed multiplier. "
            "E.g. 60 means one AIS-minute is replayed in one real second."
        ),
    )
    parser.add_argument(
        "--timestamp-col",
        default="base_date_time",
        metavar="COL",
        help="Timestamp column name in the Parquet file.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    os.chdir(Path(__file__).resolve().parent.parent)
    args = _parse_args()
    mock = AISStreamMock(
        parquet_path=args.parquet,
        speed_factor=args.speed_factor,
        timestamp_col=args.timestamp_col,
    )
    mock.run()
