"""AIS Kafka Predictor — consumes AIS messages from Kafka and applies the delay predictor.

Programmatic usage::

    import geopandas as gpd
    from shapely.geometry import Point
    import pandas as pd

    from src.methods import dms_to_dd

    df_ports = pd.read_csv("data/ports/ports.csv")
    df_ports["lat_dd"] = df_ports["latitude"].apply(dms_to_dd)
    df_ports["lon_dd"] = df_ports["longitude"].apply(dms_to_dd)
    df_ports["geometry"] = df_ports.apply(
        lambda r: Point(r["lon_dd"], r["lat_dd"]), axis=1
    )
    gdf_ports = gpd.GeoDataFrame(df_ports, geometry="geometry", crs="EPSG:4326")

    predictor = AISKafkaPredictor(
        bootstrap_servers="localhost:9092",
        topic="ais-stream",
        ports_gdf=gdf_ports,
    )
    predictor.run()
"""

import json
import logging
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from kafka import KafkaConsumer

from src.voyage_creator import VoyageCreator

logger = logging.getLogger(__name__)

# Feature columns must match the training pipeline in notebooks/training.ipynb
_NUM_FEATURES = [
    "sog",
    "sog_roll_6h",
    "sog_roll_24h",
    "sog_deviation",
    "remaining_dist_nm",
    "total_dist_nm",
    "fraction_completed",
]
_CAT_FEATURES = ["origin_port", "destination_port"]


@dataclass
class VesselPrediction:
    """Latest delay prediction for a single vessel."""

    mmsi: int
    last_prediction_time: datetime
    predicted_remaining_hours: float
    origin_port: str
    destination_port: str
    latitude: float
    longitude: float


class AISKafkaPredictor:
    """
    Consumes AIS messages from a Kafka topic, applies the delay predictor, and
    maintains a registry of all known vessels with their latest prediction times.

    Messages are expected to be JSON-encoded AIS records (as produced by
    ``AISStreamMock``) with at least the fields::

        mmsi, base_date_time, latitude, longitude, sog

    The predictor accumulates a rolling buffer of recent pings per vessel and,
    every *batch_size* messages, runs the full VoyageCreator pipeline to detect
    voyages and compute features. Predictions are generated for each vessel that
    is currently at sea with a known origin and destination port.

    Parameters
    ----------
    bootstrap_servers : str or list of str
        Kafka bootstrap server address(es), e.g. ``'localhost:9092'``.
    topic : str
        Kafka topic name containing AIS messages.
    ports_gdf : GeoDataFrame
        Port reference data with columns ``portName``, ``lat_dd``, ``lon_dd``
        and geometry in EPSG:4326.  Used by VoyageCreator for port matching
        and for computing distance-based features.
    model_path : str or Path
        Path to the serialised scikit-learn pipeline produced by
        ``notebooks/training.ipynb``.  Defaults to
        ``'models/delay_predictor.joblib'``.
    group_id : str
        Kafka consumer group ID.
    buffer_hours : float
        How many hours of AIS history to retain per vessel.  Must be at least
        24 h so that rolling-SOG windows and port-visit gap detection work
        correctly.
    batch_size : int
        Number of newly received messages that triggers a prediction pass.
    voyage_radius_nm : float
        Port buffer radius in nautical miles forwarded to VoyageCreator.
    voyage_max_speed : float
        Maximum SOG (knots) for a ping to be considered in-port.
    voyage_gap_h : float
        Hours gap between consecutive in-port pings that starts a new visit.
    timestamp_col : str
        Name of the timestamp field in Kafka messages.
    """

    def __init__(
        self,
        bootstrap_servers,
        topic: str,
        ports_gdf,
        model_path="models/delay_predictor.joblib",
        group_id: str = "ais-predictor",
        buffer_hours: float = 48.0,
        batch_size: int = 500,
        voyage_radius_nm: float = 10,
        voyage_max_speed: float = 1.5,
        voyage_gap_h: float = 24,
        timestamp_col: str = "base_date_time",
    ):
        self.topic = topic
        self.buffer_hours = buffer_hours
        self.batch_size = batch_size
        self.timestamp_col = timestamp_col

        servers = (
            bootstrap_servers
            if isinstance(bootstrap_servers, list)
            else [bootstrap_servers]
        )
        self._consumer = KafkaConsumer(
            topic,
            bootstrap_servers=servers,
            group_id=group_id,
            value_deserializer=lambda m: json.loads(m.decode("utf-8")),
            auto_offset_reset="earliest",
            enable_auto_commit=True,
        )

        self._model = joblib.load(model_path)

        self._port_loc = ports_gdf.set_index("portName")[["lat_dd", "lon_dd"]]
        self._voyage_creator = VoyageCreator(
            ports_gdf,
            radius_nm=voyage_radius_nm,
            max_speed_knots=voyage_max_speed,
            gap_threshold_h=voyage_gap_h,
        )

        # mmsi -> list of raw AIS dicts (the rolling buffer)
        self._ping_buffer: dict[int, list] = defaultdict(list)

        # Public vessel registry: mmsi -> VesselPrediction
        self.vessel_predictions: dict[int, VesselPrediction] = {}

    # ------------------------------------------------------------------
    # Feature engineering (mirrors notebooks/training.ipynb)
    # ------------------------------------------------------------------

    @staticmethod
    def _haversine_nm(lat1, lon1, lat2, lon2):
        """Vectorised great-circle distance in nautical miles."""
        radius = 3440.065  # Earth radius in nautical miles
        phi1, phi2 = np.radians(lat1), np.radians(lat2)
        delta_phi = np.radians(lat2 - lat1)
        delta_lam = np.radians(lon2 - lon1)
        a = (
            np.sin(delta_phi / 2) ** 2
            + np.cos(phi1) * np.cos(phi2) * np.sin(delta_lam / 2) ** 2
        )
        return 2 * radius * np.arcsin(np.sqrt(np.clip(a, 0, 1)))

    def _add_distance_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add remaining_dist_nm, total_dist_nm, and fraction_completed."""
        df = df.join(
            self._port_loc.rename(columns={"lat_dd": "dest_lat", "lon_dd": "dest_lon"}),
            on="destination_port",
        )
        df = df.join(
            self._port_loc.rename(
                columns={"lat_dd": "origin_lat", "lon_dd": "origin_lon"}
            ),
            on="origin_port",
        )
        df["remaining_dist_nm"] = self._haversine_nm(
            df["latitude"], df["longitude"],
            df["dest_lat"], df["dest_lon"],
        )
        df["total_dist_nm"] = self._haversine_nm(
            df["origin_lat"], df["origin_lon"],
            df["dest_lat"], df["dest_lon"],
        )
        df["fraction_completed"] = (
            1 - df["remaining_dist_nm"] / df["total_dist_nm"].replace(0, np.nan)
        ).clip(0, 1)
        return df.drop(
            columns=["dest_lat", "dest_lon", "origin_lat", "origin_lon"],
            errors="ignore",
        )

    def _add_rolling_sog(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add sog_roll_6h and sog_roll_24h using a time-based rolling window."""
        df = (
            df.sort_values(["mmsi", self.timestamp_col])
            .set_index(self.timestamp_col)
        )
        sog_grp = df.groupby("mmsi")["sog"]
        df["sog_roll_6h"] = sog_grp.rolling("6h").mean().reset_index(level=0, drop=True)
        df["sog_roll_24h"] = sog_grp.rolling("24h").mean().reset_index(level=0, drop=True)
        return df.reset_index()

    @staticmethod
    def _add_sog_deviation(df: pd.DataFrame) -> pd.DataFrame:
        """Add hist_avg_sog and sog_deviation from buffer-level lane averages."""
        lane_cols = ["origin_port", "destination_port"]
        df["_vessel_lane_avg"] = df.groupby(["mmsi"] + lane_cols)["sog"].transform("mean")
        df["_lane_avg"] = df.groupby(lane_cols)["sog"].transform("mean")
        df["hist_avg_sog"] = df["_vessel_lane_avg"].fillna(df["_lane_avg"])
        df["sog_deviation"] = df["sog"] - df["hist_avg_sog"]
        return df.drop(columns=["_vessel_lane_avg", "_lane_avg"])

    def _build_features(self, df_sea: pd.DataFrame) -> pd.DataFrame:
        """Orchestrate all feature engineering on labeled sea pings."""
        df_sea = self._add_distance_features(df_sea)
        df_sea = self._add_rolling_sog(df_sea)
        df_sea = self._add_sog_deviation(df_sea)
        return df_sea

    # ------------------------------------------------------------------
    # Buffer management
    # ------------------------------------------------------------------

    def _expire_old_pings(self, cutoff: pd.Timestamp) -> None:
        """Drop pings older than *cutoff* from every vessel buffer."""
        ts_col = self.timestamp_col
        for mmsi in list(self._ping_buffer):
            self._ping_buffer[mmsi] = [
                p
                for p in self._ping_buffer[mmsi]
                if pd.Timestamp(p[ts_col]) >= cutoff
            ]
            if not self._ping_buffer[mmsi]:
                del self._ping_buffer[mmsi]

    def _buffer_to_df(self) -> pd.DataFrame:
        """Flatten all vessel buffers into a single DataFrame."""
        all_pings = [p for pings in self._ping_buffer.values() for p in pings]
        if not all_pings:
            return pd.DataFrame()
        df = pd.DataFrame(all_pings)
        df[self.timestamp_col] = pd.to_datetime(df[self.timestamp_col])
        df["mmsi"] = df["mmsi"].astype(int)
        return df

    # ------------------------------------------------------------------
    # Prediction pass
    # ------------------------------------------------------------------

    def _run_prediction_pass(self) -> None:
        """
        Run the full voyage pipeline on buffered pings and update
        ``vessel_predictions`` with the latest prediction for each vessel
        that is currently at sea with a known destination.
        """
        df = self._buffer_to_df()
        if df.empty:
            return

        # Trim the buffer window
        latest_ts = df[self.timestamp_col].max()
        cutoff = latest_ts - pd.Timedelta(hours=self.buffer_hours)
        self._expire_old_pings(cutoff)
        df = df[df[self.timestamp_col] >= cutoff].copy()

        # Run voyage pipeline
        try:
            port_visits = self._voyage_creator.find_port_visits(
                df, timestamp_col=self.timestamp_col
            )
        except Exception:
            logger.warning("Port visit detection failed", exc_info=True)
            return

        if port_visits.empty:
            logger.debug("No port visits detected yet — buffering more pings")
            return

        df_labeled = VoyageCreator.label_pings(
            df, port_visits, timestamp_col=self.timestamp_col
        )
        df_labeled, _ = VoyageCreator.build_voyages(
            df_labeled, port_visits, timestamp_col=self.timestamp_col
        )

        # Keep only sea pings that belong to a voyage with a known destination
        df_sea = df_labeled[
            df_labeled["voyage_id"].notna()
            & df_labeled["destination_port"].notna()
        ].copy()

        if df_sea.empty:
            logger.debug("No active sea pings with voyage context")
            return

        # Feature engineering
        try:
            df_features = self._build_features(df_sea)
        except Exception:
            logger.warning("Feature engineering failed", exc_info=True)
            return

        # For each vessel take only the latest ping (most up-to-date position)
        df_latest = (
            df_features
            .sort_values(self.timestamp_col)
            .groupby("mmsi", as_index=False)
            .last()
        )

        X = df_latest[_NUM_FEATURES + _CAT_FEATURES]
        predictions = self._model.predict(X)

        # Update vessel registry
        ts_col = self.timestamp_col
        for row, pred in zip(df_latest.itertuples(index=False), predictions):
            mmsi = int(row.mmsi)
            ping_time = getattr(row, ts_col)
            if not isinstance(ping_time, datetime):
                ping_time = ping_time.to_pydatetime()

            self.vessel_predictions[mmsi] = VesselPrediction(
                mmsi=mmsi,
                last_prediction_time=ping_time,
                predicted_remaining_hours=float(pred),
                origin_port=str(row.origin_port),
                destination_port=str(row.destination_port),
                latitude=float(row.latitude),
                longitude=float(row.longitude),
            )
            logger.info(
                "MMSI %s | %s → %s | %.1f h remaining (predicted)",
                mmsi,
                row.origin_port,
                row.destination_port,
                pred,
            )

        logger.info(
            "Prediction pass complete — %d vessel(s) updated, %d total tracked",
            len(df_latest),
            len(self.vessel_predictions),
        )

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def run(self) -> None:
        """
        Consume AIS messages from Kafka indefinitely.

        A prediction pass is triggered after every *batch_size* messages.
        Stop the loop with ``KeyboardInterrupt`` (Ctrl-C).
        """
        logger.info("Starting Kafka consumer on topic '%s'", self.topic)
        msg_count = 0

        try:
            for kafka_msg in self._consumer:
                msg: dict = kafka_msg.value
                mmsi = msg.get("mmsi")
                if not mmsi:
                    continue

                self._ping_buffer[int(mmsi)].append(msg)
                msg_count += 1

                if msg_count % self.batch_size == 0:
                    logger.debug("Triggering prediction pass at message %d", msg_count)
                    self._run_prediction_pass()

        except KeyboardInterrupt:
            logger.info("Stopped by user after %d messages.", msg_count)
        finally:
            self._consumer.close()

    def get_vessel_summary(self) -> pd.DataFrame:
        """
        Return a DataFrame summarising the latest prediction for every tracked vessel.

        Columns: mmsi, last_prediction_time, predicted_remaining_hours,
                 origin_port, destination_port, latitude, longitude.

        Returns an empty DataFrame when no predictions have been made yet.
        """
        if not self.vessel_predictions:
            return pd.DataFrame(
                columns=[
                    "mmsi",
                    "last_prediction_time",
                    "predicted_remaining_hours",
                    "origin_port",
                    "destination_port",
                    "latitude",
                    "longitude",
                ]
            )
        rows = [
            {
                "mmsi": v.mmsi,
                "last_prediction_time": v.last_prediction_time,
                "predicted_remaining_hours": v.predicted_remaining_hours,
                "origin_port": v.origin_port,
                "destination_port": v.destination_port,
                "latitude": v.latitude,
                "longitude": v.longitude,
            }
            for v in self.vessel_predictions.values()
        ]
        return pd.DataFrame(rows).sort_values("mmsi").reset_index(drop=True)
