"""Tests for PortMatcher."""
import pytest
import pandas as pd
import geopandas as gpd
from datetime import datetime, timedelta
from shapely.geometry import Point

from src.port_matcher import PortMatcher

BASE_TIME = datetime(2025, 1, 1, 0, 0, 0)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def ports_gdf():
    """Two ports: one near New York, one near Los Angeles."""
    return gpd.GeoDataFrame(
        {
            "portName": ["NewYorkPort", "LosAngelesPort"],
            "geometry": [Point(-74.0, 40.7), Point(-118.2, 33.7)],
        },
        crs="EPSG:4326",
    )


@pytest.fixture
def matcher(ports_gdf):
    return PortMatcher(ports_gdf, radius_nm=10, max_speed_knots=1.5, min_time_in_port=1)


def _ais_df(mmsi, lon, lat, sog, n_hours=3):
    """Create an AIS DataFrame with n_hours hourly records at the given position."""
    timestamps = [BASE_TIME + timedelta(hours=i) for i in range(n_hours)]
    return pd.DataFrame(
        {
            "mmsi": mmsi,
            "longitude": lon,
            "latitude": lat,
            "sog": sog,
            "base_date_time": timestamps,
        }
    )


# ---------------------------------------------------------------------------
# match()
# ---------------------------------------------------------------------------

class TestMatch:
    def test_vessel_at_port_is_matched(self, matcher):
        ais = _ais_df(mmsi=111, lon=-74.0, lat=40.7, sog=0.5)
        result = matcher.match(ais)

        assert list(result.columns) == ["mmsi", "portName"]
        assert len(result) == 1
        assert result.iloc[0]["mmsi"] == 111
        assert result.iloc[0]["portName"] == "NewYorkPort"

    def test_fast_vessel_is_excluded(self, matcher):
        # sog=5.0 exceeds max_speed_knots=1.5
        ais = _ais_df(mmsi=222, lon=-74.0, lat=40.7, sog=5.0)
        assert len(matcher.match(ais)) == 0

    def test_vessel_outside_radius_is_excluded(self, matcher):
        # ~170 km west of NewYorkPort — well outside 10 nm radius
        ais = _ais_df(mmsi=333, lon=-76.0, lat=40.7, sog=0.5)
        assert len(matcher.match(ais)) == 0

    def test_short_stay_is_excluded(self, ports_gdf):
        matcher = PortMatcher(ports_gdf, radius_nm=10, max_speed_knots=1.5, min_time_in_port=5)
        # 2 records 1 hour apart → duration = 1 h < min_time_in_port = 5 h
        ais = _ais_df(mmsi=444, lon=-74.0, lat=40.7, sog=0.5, n_hours=2)
        assert len(matcher.match(ais)) == 0

    def test_result_is_deduplicated(self, matcher):
        ais = _ais_df(mmsi=555, lon=-74.0, lat=40.7, sog=0.5, n_hours=10)
        result = matcher.match(ais)
        assert len(result) == 1

    def test_multiple_vessels_at_different_ports(self, matcher):
        ais1 = _ais_df(mmsi=100, lon=-74.0, lat=40.7, sog=0.5)     # NewYorkPort
        ais2 = _ais_df(mmsi=200, lon=-118.2, lat=33.7, sog=0.5)    # LosAngelesPort
        result = matcher.match(pd.concat([ais1, ais2], ignore_index=True))

        assert len(result) == 2
        assert set(result["portName"]) == {"NewYorkPort", "LosAngelesPort"}

    def test_empty_ais_returns_empty(self, matcher):
        ais = pd.DataFrame(
            columns=["mmsi", "longitude", "latitude", "sog", "base_date_time"]
        )
        result = matcher.match(ais)
        assert len(result) == 0


# ---------------------------------------------------------------------------
# calculate_port_duration()
# ---------------------------------------------------------------------------

class TestCalculatePortDuration:
    def test_duration_is_correct(self):
        df = pd.DataFrame(
            {
                "mmsi": [111, 111],
                "portName": ["A", "A"],
                "base_date_time": [BASE_TIME, BASE_TIME + timedelta(hours=4)],
            }
        )
        result = PortMatcher.calculate_port_duration(df, timestamp_col="base_date_time")

        assert "duration_hours" in result.columns
        assert result.iloc[0]["duration_hours"] == pytest.approx(4.0)

    def test_duration_computed_per_mmsi_and_port(self):
        df = pd.DataFrame(
            {
                "mmsi":          [111, 111, 222],
                "portName":      ["A",  "A",  "B"],
                "base_date_time": [
                    BASE_TIME,
                    BASE_TIME + timedelta(hours=6),
                    BASE_TIME + timedelta(hours=2),
                ],
            }
        )
        result = PortMatcher.calculate_port_duration(df, timestamp_col="base_date_time")

        assert len(result) == 2
        row_a = result.loc[result["portName"] == "A"].iloc[0]
        assert row_a["duration_hours"] == pytest.approx(6.0)

    def test_single_record_has_zero_duration(self):
        df = pd.DataFrame(
            {
                "mmsi": [111],
                "portName": ["A"],
                "base_date_time": [BASE_TIME],
            }
        )
        result = PortMatcher.calculate_port_duration(df, timestamp_col="base_date_time")
        assert result.iloc[0]["duration_hours"] == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# add_port_call_counts()
# ---------------------------------------------------------------------------

class TestAddPortCallCounts:
    def test_counts_unique_vessels_per_port(self, matcher):
        matched = pd.DataFrame(
            {
                "mmsi":     [111, 222, 333, 444],
                "portName": ["NewYorkPort", "NewYorkPort", "NewYorkPort", "LosAngelesPort"],
            }
        )
        result = matcher.add_port_call_counts(matched)

        ny = result.loc[result["portName"] == "NewYorkPort", "port_call_count"].iloc[0]
        la = result.loc[result["portName"] == "LosAngelesPort", "port_call_count"].iloc[0]
        assert ny == 3
        assert la == 1

    def test_port_with_no_visits_gets_zero(self, matcher):
        matched = pd.DataFrame(
            {"mmsi": [111], "portName": ["NewYorkPort"]}
        )
        result = matcher.add_port_call_counts(matched)

        la = result.loc[result["portName"] == "LosAngelesPort", "port_call_count"].iloc[0]
        assert la == 0

    def test_same_vessel_visiting_port_multiple_times_counts_once(self, matcher):
        matched = pd.DataFrame(
            {"mmsi": [111, 111, 111], "portName": ["NewYorkPort"] * 3}
        )
        result = matcher.add_port_call_counts(matched)

        ny = result.loc[result["portName"] == "NewYorkPort", "port_call_count"].iloc[0]
        assert ny == 1

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
