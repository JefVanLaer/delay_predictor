"""Tests for VoyageCreator."""
import pytest
import pandas as pd
import geopandas as gpd
from datetime import datetime, timedelta
from shapely.geometry import Point

from src.voyage_creator import VoyageCreator

BASE_TIME = datetime(2025, 1, 1, 0, 0, 0)


# ---------------------------------------------------------------------------
# Fixtures & helpers
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
def creator(ports_gdf):
    return VoyageCreator(ports_gdf, radius_nm=10, max_speed_knots=1.5, gap_threshold_h=24)


def _ais_pings(mmsi, lon, lat, sog, start_time=BASE_TIME, n_hours=3):
    """Create n_hours hourly AIS pings for one vessel at a fixed position."""
    timestamps = [start_time + timedelta(hours=i) for i in range(n_hours)]
    return pd.DataFrame({
        "mmsi": mmsi,
        "longitude": lon,
        "latitude": lat,
        "sog": sog,
        "base_date_time": timestamps,
    })


# ---------------------------------------------------------------------------
# find_port_visits()
# ---------------------------------------------------------------------------

class TestFindPortVisits:
    def test_returns_expected_columns(self, creator):
        ais = _ais_pings(mmsi=111, lon=-74.0, lat=40.7, sog=0.5)
        result = creator.find_port_visits(ais)
        assert set(result.columns) >= {"mmsi", "portName", "entry_time", "exit_time", "duration_hours"}

    def test_vessel_at_port_creates_one_visit(self, creator):
        ais = _ais_pings(mmsi=111, lon=-74.0, lat=40.7, sog=0.5, n_hours=4)
        result = creator.find_port_visits(ais)
        assert len(result) == 1
        assert result.iloc[0]["mmsi"] == 111
        assert result.iloc[0]["portName"] == "NewYorkPort"

    def test_visit_entry_exit_times_are_correct(self, creator):
        ais = _ais_pings(mmsi=111, lon=-74.0, lat=40.7, sog=0.5, n_hours=5)
        result = creator.find_port_visits(ais)
        row = result.iloc[0]
        assert row["entry_time"] == pd.Timestamp(BASE_TIME)
        assert row["exit_time"] == pd.Timestamp(BASE_TIME + timedelta(hours=4))
        assert row["duration_hours"] == pytest.approx(4.0)

    def test_fast_vessel_produces_no_visits(self, creator):
        ais = _ais_pings(mmsi=222, lon=-74.0, lat=40.7, sog=5.0)
        result = creator.find_port_visits(ais)
        assert len(result) == 0

    def test_vessel_outside_radius_produces_no_visits(self, creator):
        # ~170 km west of NewYorkPort — well outside 10 nm radius
        ais = _ais_pings(mmsi=333, lon=-76.0, lat=40.7, sog=0.5)
        result = creator.find_port_visits(ais)
        assert len(result) == 0

    def test_gap_larger_than_threshold_splits_into_two_visits(self, creator):
        block1 = _ais_pings(mmsi=111, lon=-74.0, lat=40.7, sog=0.5,
                             start_time=BASE_TIME, n_hours=3)
        block2 = _ais_pings(mmsi=111, lon=-74.0, lat=40.7, sog=0.5,
                             start_time=BASE_TIME + timedelta(hours=48), n_hours=3)
        ais = pd.concat([block1, block2], ignore_index=True)
        result = creator.find_port_visits(ais)
        assert len(result) == 2
        assert (result["portName"] == "NewYorkPort").all()

    def test_gap_smaller_than_threshold_stays_one_visit(self, creator):
        block1 = _ais_pings(mmsi=111, lon=-74.0, lat=40.7, sog=0.5,
                             start_time=BASE_TIME, n_hours=3)
        block2 = _ais_pings(mmsi=111, lon=-74.0, lat=40.7, sog=0.5,
                             start_time=BASE_TIME + timedelta(hours=5), n_hours=3)
        ais = pd.concat([block1, block2], ignore_index=True)
        result = creator.find_port_visits(ais)
        assert len(result) == 1

    def test_multiple_vessels_at_different_ports(self, creator):
        ais1 = _ais_pings(mmsi=100, lon=-74.0, lat=40.7, sog=0.5)    # NewYorkPort
        ais2 = _ais_pings(mmsi=200, lon=-118.2, lat=33.7, sog=0.5)   # LosAngelesPort
        result = creator.find_port_visits(pd.concat([ais1, ais2], ignore_index=True))
        assert len(result) == 2
        assert set(result["portName"]) == {"NewYorkPort", "LosAngelesPort"}

    def test_empty_ais_returns_empty(self, creator):
        ais = pd.DataFrame(columns=["mmsi", "longitude", "latitude", "sog", "base_date_time"])
        result = creator.find_port_visits(ais)
        assert len(result) == 0


# ---------------------------------------------------------------------------
# label_pings()
# ---------------------------------------------------------------------------

class TestLabelPings:
    # Scenario: vessel sails NY -> sea -> LA
    # Pings t=0..2 at NY, t=3..5 at sea, t=6..8 at LA
    # Visits: NY (entry=t0, exit=t2), LA (entry=t6, exit=t8)

    @pytest.fixture
    def scenario(self):
        t = [BASE_TIME + timedelta(hours=i) for i in range(9)]
        ais = pd.DataFrame({
            "mmsi": 111,
            "longitude": [-74.0] * 3 + [-100.0] * 3 + [-118.2] * 3,
            "latitude":  [40.7]  * 3 + [35.0]   * 3 + [33.7]   * 3,
            "sog": [0.5] * 3 + [10.0] * 3 + [0.5] * 3,
            "base_date_time": t,
        })
        visits = pd.DataFrame({
            "mmsi":           [111,            111],
            "portName":       ["NewYorkPort",  "LosAngelesPort"],
            "entry_time":     [t[0],           t[6]],
            "exit_time":      [t[2],           t[8]],
            "duration_hours": [2.0,            2.0],
        })
        return ais, visits

    def test_returns_expected_columns(self, scenario):
        ais, visits = scenario
        result = VoyageCreator.label_pings(ais, visits)
        for col in ("current_port", "origin_port", "destination_port", "voyage_id"):
            assert col in result.columns

    def test_ping_in_port_window_gets_current_port(self, scenario):
        ais, visits = scenario
        result = VoyageCreator.label_pings(ais, visits)
        ny_pings = result[result["base_date_time"] <= BASE_TIME + timedelta(hours=2)]
        assert ny_pings["current_port"].eq("NewYorkPort").all()

    def test_sea_ping_has_no_current_port(self, scenario):
        ais, visits = scenario
        result = VoyageCreator.label_pings(ais, visits)
        sea_pings = result[
            (result["base_date_time"] > BASE_TIME + timedelta(hours=2)) &
            (result["base_date_time"] < BASE_TIME + timedelta(hours=6))
        ]
        assert sea_pings["current_port"].isna().all()

    def test_sea_ping_origin_is_last_departed_port(self, scenario):
        ais, visits = scenario
        result = VoyageCreator.label_pings(ais, visits)
        sea_pings = result[result["current_port"].isna()]
        assert sea_pings["origin_port"].eq("NewYorkPort").all()

    def test_sea_ping_destination_is_next_arrival_port(self, scenario):
        ais, visits = scenario
        result = VoyageCreator.label_pings(ais, visits)
        sea_pings = result[result["current_port"].isna()]
        assert sea_pings["destination_port"].eq("LosAngelesPort").all()

    def test_port_ping_has_no_origin_or_destination(self, scenario):
        ais, visits = scenario
        result = VoyageCreator.label_pings(ais, visits)
        port_pings = result[result["current_port"].notna()]
        assert port_pings["origin_port"].isna().all()
        assert port_pings["destination_port"].isna().all()

    def test_all_voyage_ids_start_as_na(self, scenario):
        ais, visits = scenario
        result = VoyageCreator.label_pings(ais, visits)
        assert result["voyage_id"].isna().all()


# ---------------------------------------------------------------------------
# build_voyages()
# ---------------------------------------------------------------------------

class TestBuildVoyages:
    # Scenario: one vessel, two port visits with sea pings in between.
    # NY visit: entry=t0, exit=t5
    # Sea pings: t6, t7, t8, t9
    # LA visit: entry=t10, exit=t15

    @pytest.fixture
    def voyage_scenario(self):
        t = lambda h: BASE_TIME + timedelta(hours=h)
        visits = pd.DataFrame({
            "mmsi":           [111,        111],
            "portName":       ["NewYorkPort", "LosAngelesPort"],
            "entry_time":     [t(0),  t(10)],
            "exit_time":      [t(5),  t(15)],
            "duration_hours": [5.0,   5.0],
        })
        sea_times  = [t(h) for h in (6, 7, 8, 9)]
        port_times = [t(h) for h in (0, 1, 2, 3, 4, 5, 10, 11, 12)]
        all_times  = sorted(sea_times + port_times)
        labeled = pd.DataFrame({
            "mmsi": 111,
            "base_date_time": all_times,
            "current_port": [
                "NewYorkPort"    if t(0) <= ts <= t(5)  else
                "LosAngelesPort" if ts >= t(10)         else
                None
                for ts in all_times
            ],
            "origin_port": None,
            "destination_port": None,
            "voyage_id": pd.NA,
        })
        return labeled, visits

    def test_returns_tuple_of_two_dataframes(self, voyage_scenario):
        labeled, visits = voyage_scenario
        result = VoyageCreator.build_voyages(labeled, visits)
        assert isinstance(result, tuple) and len(result) == 2
        df_labeled, df_voyages = result
        assert isinstance(df_labeled, pd.DataFrame)
        assert isinstance(df_voyages, pd.DataFrame)

    def test_one_voyage_created_for_two_consecutive_visits(self, voyage_scenario):
        labeled, visits = voyage_scenario
        _, df_voyages = VoyageCreator.build_voyages(labeled, visits)
        assert len(df_voyages) == 1

    def test_voyage_has_correct_ports(self, voyage_scenario):
        labeled, visits = voyage_scenario
        _, df_voyages = VoyageCreator.build_voyages(labeled, visits)
        voyage = df_voyages.iloc[0]
        assert voyage["departure_port"] == "NewYorkPort"
        assert voyage["arrival_port"]   == "LosAngelesPort"

    def test_voyage_has_correct_times_and_duration(self, voyage_scenario):
        labeled, visits = voyage_scenario
        t = lambda h: BASE_TIME + timedelta(hours=h)
        _, df_voyages = VoyageCreator.build_voyages(labeled, visits)
        voyage = df_voyages.iloc[0]
        assert voyage["departure_time"] == t(5)
        assert voyage["arrival_time"]   == t(10)
        assert voyage["duration_hours"] == pytest.approx(5.0)

    def test_voyage_ping_count_is_correct(self, voyage_scenario):
        labeled, visits = voyage_scenario
        _, df_voyages = VoyageCreator.build_voyages(labeled, visits)
        assert df_voyages.iloc[0]["ping_count"] == 4  # t=6,7,8,9

    def test_sea_pings_get_voyage_id(self, voyage_scenario):
        labeled, visits = voyage_scenario
        t = lambda h: BASE_TIME + timedelta(hours=h)
        df_labeled, df_voyages = VoyageCreator.build_voyages(labeled, visits)
        sea_pings = df_labeled[
            (df_labeled["base_date_time"] > t(5)) &
            (df_labeled["base_date_time"] < t(10))
        ]
        expected_id = df_voyages.iloc[0]["voyage_id"]
        assert sea_pings["voyage_id"].eq(expected_id).all()

    def test_port_pings_keep_na_voyage_id(self, voyage_scenario):
        labeled, visits = voyage_scenario
        df_labeled, _ = VoyageCreator.build_voyages(labeled, visits)
        port_pings = df_labeled[df_labeled["current_port"].notna()]
        assert port_pings["voyage_id"].isna().all()

    def test_single_visit_produces_no_voyages(self):
        t = lambda h: BASE_TIME + timedelta(hours=h)
        visits = pd.DataFrame({
            "mmsi":           [111],
            "portName":       ["NewYorkPort"],
            "entry_time":     [t(0)],
            "exit_time":      [t(5)],
            "duration_hours": [5.0],
        })
        labeled = pd.DataFrame({
            "mmsi": 111,
            "base_date_time": [t(0), t(1), t(2)],
            "current_port": "NewYorkPort",
            "origin_port": None,
            "destination_port": None,
            "voyage_id": pd.NA,
        })
        _, df_voyages = VoyageCreator.build_voyages(labeled, visits)
        assert len(df_voyages) == 0

    def test_overlapping_visits_are_skipped(self):
        """Visit B's entry_time <= visit A's exit_time → no valid voyage."""
        t = lambda h: BASE_TIME + timedelta(hours=h)
        visits = pd.DataFrame({
            "mmsi":           [111,    111],
            "portName":       ["PortA", "PortB"],
            "entry_time":     [t(0),   t(3)],  # PortB enters before PortA exits
            "exit_time":      [t(5),   t(8)],
            "duration_hours": [5.0,    5.0],
        })
        labeled = pd.DataFrame({
            "mmsi": 111,
            "base_date_time": [t(i) for i in range(9)],
            "current_port": None,
            "origin_port": None,
            "destination_port": None,
            "voyage_id": pd.NA,
        })
        _, df_voyages = VoyageCreator.build_voyages(labeled, visits)
        assert len(df_voyages) == 0

    def test_multiple_vessels_get_independent_voyages(self):
        """Two vessels each with two port visits → two voyages, one per vessel."""
        t = lambda h: BASE_TIME + timedelta(hours=h)
        visits = pd.DataFrame({
            "mmsi":           [111,    111,    222,    222],
            "portName":       ["PortA", "PortB", "PortC", "PortD"],
            "entry_time":     [t(0),  t(10), t(0),  t(10)],
            "exit_time":      [t(5),  t(15), t(5),  t(15)],
            "duration_hours": [5.0] * 4,
        })
        labeled = pd.DataFrame({
            "mmsi": [111] * 4 + [222] * 4,
            "base_date_time": [t(6), t(7), t(8), t(9)] * 2,
            "current_port": None,
            "origin_port": None,
            "destination_port": None,
            "voyage_id": pd.NA,
        })
        _, df_voyages = VoyageCreator.build_voyages(labeled, visits)
        assert len(df_voyages) == 2
        assert set(df_voyages["mmsi"]) == {111, 222}


if __name__ == "__main__":
    pytest.main([__file__, "-v"])