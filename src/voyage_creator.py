import pandas as pd

from src.port_matcher import PortMatcher


class VoyageCreator:
    """
    Builds a voyage dataset from AIS tracking data and a port reference GeoDataFrame.

    A voyage is the sea leg between two consecutive port visits for the same vessel.
    Port visits are detected using PortMatcher's spatial join (speed + proximity
    filters) and then split on time gaps so that repeated visits to the same port
    across sparse AIS snapshot files are kept as separate events.

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

        Each row represents one contiguous stay at a port. Stays are split when
        consecutive pings for the same (mmsi, portName) are more than
        gap_threshold_h hours apart.

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
        near_port = self.matcher.find_candidates(ais_df, timestamp_col=timestamp_col)
        return self._split_visits(near_port, timestamp_col)

    def _split_visits(self, near_port_df, timestamp_col):
        """
        Split matched pings into individual visits using time-gap detection.
        """
        records = []
        for (mmsi, portName), grp in near_port_df.groupby(['mmsi', 'portName']):
            grp = grp.sort_values(timestamp_col)

            gap_hours = grp[timestamp_col].diff().dt.total_seconds() / 3600
            visit_num = (gap_hours > self.gap_threshold_h).cumsum()

            for _, visit in grp.groupby(visit_num):
                entry    = visit[timestamp_col].min()
                exit_    = visit[timestamp_col].max()
                duration = (exit_ - entry).total_seconds() / 3600
                records.append({
                    'mmsi':           mmsi,
                    'portName':       portName,
                    'entry_time':     entry,
                    'exit_time':      exit_,
                    'duration_hours': duration,
                })

        if not records:
            return pd.DataFrame(columns=['mmsi', 'portName', 'entry_time', 'exit_time', 'duration_hours'])

        return (
            pd.DataFrame(records)
            .sort_values(['mmsi', 'entry_time'])
            .reset_index(drop=True)
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
        ais_df = ais_df.copy()
        ais_df[timestamp_col] = pd.to_datetime(ais_df[timestamp_col])

        # --- current_port ---
        in_visit = (
            ais_df[['mmsi', timestamp_col]]
            .merge(port_visits[['mmsi', 'portName', 'entry_time', 'exit_time']], on='mmsi', how='left')
            .query('entry_time <= base_date_time <= exit_time')[['mmsi', timestamp_col, 'portName']]
            .drop_duplicates(['mmsi', timestamp_col])
        )
        df = ais_df.merge(
            in_visit.rename(columns={'portName': 'current_port'}),
            on=['mmsi', timestamp_col],
            how='left',
        )

        # --- origin_port / destination_port ---
        # merge_asof requires the 'on' column to be globally monotone, so sort by
        # timestamp alone (not by mmsi first); by='mmsi' handles per-vessel grouping.
        pings_sorted        = df.sort_values(timestamp_col).reset_index(drop=True)
        visits_by_exit      = port_visits[['mmsi', 'portName', 'exit_time' ]].sort_values('exit_time' )
        visits_by_entry     = port_visits[['mmsi', 'portName', 'entry_time']].sort_values('entry_time')

        origin = pd.merge_asof(
            pings_sorted[['mmsi', timestamp_col]],
            visits_by_exit.rename(columns={'portName': 'origin_port'}),
            by='mmsi', left_on=timestamp_col, right_on='exit_time', direction='backward',
        )['origin_port']

        destination = pd.merge_asof(
            pings_sorted[['mmsi', timestamp_col]],
            visits_by_entry.rename(columns={'portName': 'destination_port'}),
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

        for mmsi, visits in port_visits.groupby('mmsi'):
            visits_sorted = visits.sort_values('entry_time').reset_index(drop=True)

            for i in range(len(visits_sorted) - 1):
                dep = visits_sorted.iloc[i]
                arr = visits_sorted.iloc[i + 1]

                # Skip overlapping port buffers
                if arr['entry_time'] <= dep['exit_time']:
                    continue

                sea_mask = (
                    (df_labeled['mmsi'] == mmsi)
                    & (df_labeled[timestamp_col] > dep['exit_time'])
                    & (df_labeled[timestamp_col] < arr['entry_time'])
                )
                df_labeled.loc[sea_mask, 'voyage_id'] = voyage_id_counter

                voyage_records.append({
                    'voyage_id':      voyage_id_counter,
                    'mmsi':           mmsi,
                    'departure_port': dep['portName'],
                    'departure_time': dep['exit_time'],
                    'arrival_port':   arr['portName'],
                    'arrival_time':   arr['entry_time'],
                    'duration_hours': (arr['entry_time'] - dep['exit_time']).total_seconds() / 3600,
                    'ping_count':     int(sea_mask.sum()),
                })
                voyage_id_counter += 1

        df_voyages = pd.DataFrame(voyage_records)
        return df_labeled, df_voyages
