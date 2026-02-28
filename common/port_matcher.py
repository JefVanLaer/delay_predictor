from geopandas import GeoDataFrame, read_file
from shapely.geometry import Point
import matplotlib.pyplot as plt
import folium

class PortMatcher:

    us_filepath = "https://www2.census.gov/geo/tiger/GENZ2022/shp/cb_2022_us_state_500k.zip"

    def __init__(self, ports_gdf, radius_nm=10, max_speed_knots=1.5):
        self.ports = ports_gdf
        self.radius_m = radius_nm * 1852  # convert to metres
        self.max_speed = max_speed_knots

    def match(self, ais_df):
        """
        For each AIS record, find the nearest port within radius_m
        where SOG is below max_speed. Takes a pandas DataFrame with
        latitude and longitude columns. Returns GeoDataFrame with a new
        portName column.
        """
        # filter to slow-moving records first
        candidates = ais_df[ais_df['sog'] <= self.max_speed].copy()

        # convert pandas dataframe to geopandas dataframe
        geometry = [Point(xy) for xy in zip(candidates['longitude'], candidates['latitude'])]
        candidates_gdf = GeoDataFrame(candidates, geometry=geometry, crs='EPSG:4326')

        # ensure ports have same CRS
        ports_buffered = self.ports.copy()
        if ports_buffered.crs != candidates_gdf.crs:
            ports_buffered = ports_buffered.to_crs(candidates_gdf.crs)

        # buffer in a projected CRS (meters) for accurate distances
        ports_buffered = ports_buffered.to_crs('EPSG:3857')  # Web Mercator
        candidates_gdf = candidates_gdf.to_crs('EPSG:3857')
        ports_buffered['geometry'] = ports_buffered.geometry.buffer(self.radius_m)

        matched = candidates_gdf.sjoin(
            ports_buffered[['portName', 'geometry']],
            predicate='within'
        )

        # return unique mmsi and portName combinations
        return matched[['mmsi', 'portName']].drop_duplicates()

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
            markersize=self.ports['port_call_count'] * 10 + 20,  # scale marker size by call count
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

