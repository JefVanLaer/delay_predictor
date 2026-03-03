import numpy as np
import pandas as pd

from src.port_matcher import PortMatcher


class VoyageCreator:
    """
    Builds a voyage dataset from AIS tracking data and a port reference GeoDataFrame.

    A voyage is the sea leg between two consecutive port visits for the same vessel.
    Port visits are detected via PortMatcher.find_port_visits(), which applies spatial,
    speed, and gap-based filters. VoyageCreator then labels every AIS ping with
    routing context and groups consecutive visits into voyage records.

    Parameters
    ----------
    ports_gdf : GeoDataFrame
        Port locations in EPSG:4326.
    radius_nm : float
        Port buffer radius in nautical miles.
    max_speed_knots : float
        Maximum SOG (knots) for a ping to be considered stationary.
    gap_threshold_h : float
        Hours gap between consecutive pings at the same port that triggers a new visit.
        Should be larger than the maximum expected gap within a single continuous stay
        but smaller than the minimum gap between two distinct visits.
    """

    def __init__(self, ports_gdf, radius_nm=10, max_speed_knots=1.5, gap_threshold_h=24):
        self.matcher = PortMatcher(ports_gdf, radius_nm=radius_nm, max_speed_knots=max_speed_knots)
        self.gap_threshold_h = gap_threshold_h

    def find_port_visits(self, ais_df, timestamp_col='base_date_time'):
        """
        Return a DataFrame of individual port visits derived from AIS data.

        Delegates to PortMatcher.find_port_visits() using this instance's
        gap_threshold_h. Each row represents one contiguous stay at a port.

        Parameters
        ----------
        ais_df : DataFrame
            AIS records with at least mmsi, longitude, latitude, sog, and timestamp_col.
        timestamp_col : str
            Name of the timestamp column.

        Returns
        -------
        DataFrame with columns: mmsi, portName, entry_time, exit_time, duration_hours
        """
        return self.matcher.find_port_visits(
            ais_df,
            gap_threshold_h=self.gap_threshold_h,
            timestamp_col=timestamp_col,
        )

    @staticmethod
    def label_pings(ais_df, port_visits, timestamp_col='base_date_time'):
        """
        Add current_port, origin_port, destination_port, and voyage_id columns
        to every AIS ping.

        - current_port     : port name when the ping falls within a visit window; NaN at sea.
        - origin_port      : last port departed before this ping (sea pings only).
        - destination_port : next port to arrive at after this ping (sea pings only).
        - voyage_id        : set later by build_voyages(); NaN here.

        Parameters
        ----------
        ais_df : DataFrame
            AIS records (sorted by mmsi, timestamp_col).
        port_visits : DataFrame
            Output of find_port_visits().
        timestamp_col : str
            Name of the timestamp column.

        Returns
        -------
        DataFrame — copy of ais_df with four new columns.
        """
        df = ais_df.copy()
        df[timestamp_col] = pd.to_datetime(df[timestamp_col])

        # Sort globally by timestamp (required by merge_asof's monotone 'on' column).
        pings_sorted    = df.sort_values(timestamp_col).reset_index(drop=True)
        visits_by_entry = port_visits[['mmsi', 'portName', 'entry_time', 'exit_time']].sort_values('entry_time')
        visits_by_exit  = port_visits[['mmsi', 'portName', 'exit_time']].sort_values('exit_time')

        # --- current_port ---
        # Replace the O(N×M) cartesian merge+filter with an O(N log N) merge_asof.
        # Find each ping's most-recent visit whose entry_time ≤ ping timestamp, then
        # confirm the ping also falls within that visit's exit_time window.
        current = pd.merge_asof(
            pings_sorted[['mmsi', timestamp_col]],
            visits_by_entry.rename(columns={'portName': 'current_port'}),
            by='mmsi', left_on=timestamp_col, right_on='entry_time', direction='backward',
        )
        in_visit_mask = (
            current['exit_time'].notna().values
            & (pings_sorted[timestamp_col].values <= current['exit_time'].values)
        )
        pings_sorted['current_port'] = current['current_port'].where(in_visit_mask)

        # --- origin_port / destination_port ---
        origin = pd.merge_asof(
            pings_sorted[['mmsi', timestamp_col]],
            visits_by_exit.rename(columns={'portName': 'origin_port'}),
            by='mmsi', left_on=timestamp_col, right_on='exit_time', direction='backward',
        )['origin_port']

        destination = pd.merge_asof(
            pings_sorted[['mmsi', timestamp_col]],
            visits_by_entry[['mmsi', 'portName', 'entry_time']].rename(columns={'portName': 'destination_port'}),
            by='mmsi', left_on=timestamp_col, right_on='entry_time', direction='forward',
        )['destination_port']

        pings_sorted['origin_port']      = origin.values
        pings_sorted['destination_port'] = destination.values
        pings_sorted['voyage_id']        = pd.NA

        # Pings at port are not en route
        at_port = pings_sorted['current_port'].notna()
        pings_sorted.loc[at_port, ['origin_port', 'destination_port']] = None

        return pings_sorted.sort_values(['mmsi', timestamp_col]).reset_index(drop=True)

    @staticmethod
    def build_voyages(df_labeled, port_visits, timestamp_col='base_date_time'):
        """
        Group consecutive port visits into voyage records and stamp each sea
        ping with its voyage_id.

        A voyage runs from the exit_time of visit i to the entry_time of visit i+1
        for the same vessel. Overlapping visits (e.g. two ports whose buffers both
        contain the vessel) are skipped.

        Parameters
        ----------
        df_labeled : DataFrame
            Output of label_pings() — modified in-place to fill voyage_id.
        port_visits : DataFrame
            Output of find_port_visits().
        timestamp_col : str
            Name of the timestamp column.

        Returns
        -------
        tuple(df_labeled, df_voyages)
            df_labeled  : input DataFrame with voyage_id filled for sea pings.
            df_voyages  : one row per voyage with departure/arrival metadata.
        """
        voyage_records    = []
        voyage_id_counter = 0

        # Sort once so each vessel's pings form a contiguous, time-ordered block.
        # This lets us locate per-vessel slices and voyage windows with binary search
        # instead of scanning the full DataFrame for every voyage (O(V·N) → O(N + V log N)).
        df_labeled = df_labeled.sort_values(['mmsi', timestamp_col]).reset_index(drop=True)

        mmsi_arr      = df_labeled['mmsi'].values
        ts_arr        = df_labeled[timestamp_col].values
        voyage_id_col = df_labeled.columns.get_loc('voyage_id')

        for mmsi, visits in port_visits.groupby('mmsi'):
            # Find this vessel's contiguous row range via binary search on the sorted mmsi array.
            v_start   = int(np.searchsorted(mmsi_arr, mmsi, side='left'))
            v_end     = int(np.searchsorted(mmsi_arr, mmsi, side='right'))
            vessel_ts = ts_arr[v_start:v_end]

            visits_sorted = visits.sort_values('entry_time').reset_index(drop=True)

            for i in range(len(visits_sorted) - 1):
                dep = visits_sorted.iloc[i]
                arr = visits_sorted.iloc[i + 1]

                # Skip overlapping port buffers
                if arr['entry_time'] <= dep['exit_time']:
                    continue

                # Binary search for the sea-ping window: dep.exit_time < ts < arr.entry_time
                lo = int(np.searchsorted(vessel_ts, dep['exit_time'], side='right'))
                hi = int(np.searchsorted(vessel_ts, arr['entry_time'], side='left'))
                ping_count = hi - lo

                if ping_count > 0:
                    df_labeled.iloc[v_start + lo : v_start + hi, voyage_id_col] = voyage_id_counter

                voyage_records.append({
                    'voyage_id':      voyage_id_counter,
                    'mmsi':           mmsi,
                    'departure_port': dep['portName'],
                    'departure_time': dep['exit_time'],
                    'arrival_port':   arr['portName'],
                    'arrival_time':   arr['entry_time'],
                    'duration_hours': (arr['entry_time'] - dep['exit_time']).total_seconds() / 3600,
                    'ping_count':     ping_count,
                })
                voyage_id_counter += 1

        df_voyages = pd.DataFrame(voyage_records)
        return df_labeled, df_voyages
