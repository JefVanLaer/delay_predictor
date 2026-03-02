from geopandas import GeoDataFrame, read_file
from shapely.geometry import Point
import matplotlib.pyplot as plt
import folium
import pandas as pd

class PortMatcher:

    us_filepath = "https://www2.census.gov/geo/tiger/GENZ2022/shp/cb_2022_us_state_500k.zip"

    def __init__(self, ports_gdf, radius_nm=10, max_speed_knots=1.5, min_time_in_port=1):
        self.ports = ports_gdf
        self.radius_m = radius_nm * 1852  # convert to metres
        self.max_speed = max_speed_knots
        self.min_time_in_port = min_time_in_port

    def find_candidates(self, ais_df, lat_col='latitude', lon_col='longitude', timestamp_col='base_date_time'):
        """
        Return all slow-moving AIS pings that fall within a port buffer, with
        a portName column attached. No duration filtering is applied.

        Parameters:
        -----------
        ais_df : pandas.DataFrame
            Input AIS data
        lat_col : str
            Name of latitude column (default: 'latitude')
        lon_col : str
            Name of longitude column (default: 'longitude')
        timestamp_col : str
            Name of timestamp column. Converted to datetime if provided.

        Returns:
        --------
        GeoDataFrame with all original columns plus portName
        """
        candidates = ais_df[ais_df['sog'] <= self.max_speed].copy()

        if timestamp_col is not None:
            candidates[timestamp_col] = pd.to_datetime(candidates[timestamp_col])

        geometry = [Point(xy) for xy in zip(candidates[lon_col], candidates[lat_col])]
        candidates_gdf = GeoDataFrame(candidates, geometry=geometry, crs='EPSG:4326')

        ports_buffered = self.ports.copy()
        if ports_buffered.crs != candidates_gdf.crs:
            ports_buffered = ports_buffered.to_crs(candidates_gdf.crs)

        ports_buffered = ports_buffered.to_crs('EPSG:3857')
        candidates_gdf = candidates_gdf.to_crs('EPSG:3857')
        ports_buffered['geometry'] = ports_buffered.geometry.buffer(self.radius_m)

        return candidates_gdf.sjoin(
            ports_buffered[['portName', 'geometry']],
            predicate='within'
        )

    def find_port_visits(self, ais_df, gap_threshold_h=24,
                         lat_col='latitude', lon_col='longitude', timestamp_col='base_date_time'):
        """
        Return a DataFrame of individual port visits derived from AIS data.

        Each row represents one contiguous stay at a port. Consecutive pings
        for the same (mmsi, portName) that are separated by more than
        gap_threshold_h hours are treated as separate visits. Making this
        suitable for sparse AIS datasets where the same vessel may appear at
        the same port across multiple snapshot files.

        Parameters:
        -----------
        ais_df : pandas.DataFrame
            Input AIS data
        gap_threshold_h : float
            Hour gap between consecutive pings that triggers a new visit (default: 24).
        lat_col : str
            Name of latitude column (default: 'latitude')
        lon_col : str
            Name of longitude column (default: 'longitude')
        timestamp_col : str
            Name of timestamp column (default: 'base_date_time')

        Returns:
        --------
        DataFrame with columns: mmsi, portName, entry_time, exit_time, duration_hours
        """
        candidates = self.find_candidates(ais_df, lat_col=lat_col, lon_col=lon_col, timestamp_col=timestamp_col)

        records = []
        for (mmsi, portName), grp in candidates.groupby(['mmsi', 'portName']):
            grp = grp.sort_values(timestamp_col)
            gap_hours = grp[timestamp_col].diff().dt.total_seconds() / 3600
            visit_num = (gap_hours > gap_threshold_h).cumsum()

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

    def match(self, ais_df, gap_threshold_h=24, lat_col='latitude', lon_col='longitude', timestamp_col='base_date_time'):
        """
        Return unique (mmsi, portName) pairs for vessels that spent at least
        min_time_in_port hours at a port.

        Uses gap-aware visit splitting so that sparse AIS snapshots with the
        same vessel at the same port months apart are not merged into a single
        artificially long stay.

        Parameters:
        -----------
        ais_df : pandas.DataFrame
            Input AIS data
        gap_threshold_h : float
            Hour gap that splits a continuous stay into separate visits (default: 24).
        lat_col : str
            Name of latitude column (default: 'latitude')
        lon_col : str
            Name of longitude column (default: 'longitude')
        timestamp_col : str
            Name of timestamp column (default: 'base_date_time')
        """
        visits = self.find_port_visits(
            ais_df,
            gap_threshold_h=gap_threshold_h,
            lat_col=lat_col,
            lon_col=lon_col,
            timestamp_col=timestamp_col,
        )

        if visits.empty:
            return pd.DataFrame(columns=['mmsi', 'portName'])

        filtered = visits[visits['duration_hours'] >= self.min_time_in_port]
        return filtered[['mmsi', 'portName']].drop_duplicates().reset_index(drop=True)

    def add_port_call_counts(self, matched_df):
        """
        Takes the output of match() and adds a 'port_call_count' column
        to self.ports based on the number of unique vessels visiting each port.
        """
        # count unique MMSI per port
        port_call_counts = matched_df.groupby('portName')['mmsi'].nunique().reset_index()
        port_call_counts.columns = ['portName', 'port_call_count']

        # merge with ports GeoDataFrame
        self.ports = self.ports.merge(port_call_counts, on='portName', how='left')
        self.ports['port_call_count'] = self.ports['port_call_count'].fillna(0).astype(int)

        return self.ports

    def visualize_port_calls(self):
        """
        Visualize the ports with a color scale based on port_call_count.
        Requires matplotlib and geopandas.
        """

        fig, ax = plt.subplots(figsize=(16, 10))

        # Plot US map
        usa = read_file(self.us_filepath)
        usa.plot(ax=ax, color='lightgray', edgecolor='black')

        self.ports.plot(
            ax=ax,
            alpha=0.6,
            markersize=self.ports['port_call_count'] * 10,  # scale marker size by call count
            color='red',
            edgecolor='red',
            linewidth=0.5,
            label='Ports',
            legend=True
        )

        ax.set_xlabel('Longitude', fontsize=12)
        ax.set_ylabel('Latitude', fontsize=12)
        ax.set_title('Geographic Distribution of Ports in the US (GeoPandas)', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=10)


        # Center the graph on US coastal areas
        ax.set_xlim(-125, -66)
        ax.set_ylim(24, 50)

        plt.tight_layout()
        plt.show()

    def visualize_port_calls_folium(self, n=10):
        """
        Visualize the ports with a color scale based on port_call_count using Folium.
        """

        ports_to_show = self.ports.sort_values('port_call_count', ascending=False).head(n)

        # Create a Folium map centered on the US
        center = [ports_to_show.geometry.y.mean(), ports_to_show.geometry.x.mean()]
        m = folium.Map(
            location=center,
            zoom_start=6,
            min_zoom=2,      # allow zooming out
            max_zoom=18,     # allow zooming in
            zoom_control=True,   # show zoom buttons
            control_scale=True   # optional: show scale bar
        )

        folium.GeoJson(
            ports_to_show.to_crs('EPSG:4326'),
            tooltip=folium.GeoJsonTooltip(
                fields=["portName", "port_call_count"],         # Columns to show
                aliases=["Name:", "Count:"],      # Optional labels
                localize=True                     # Format numbers nicely
            )
        ).add_to(m)

        return m
