# streamlit_app_with_elevation.py
import streamlit as st
import geopandas as gpd
import pydeck as pdk
import pandas as pd
import rasterio
from rasterio.warp import transform

# --- Load your GDB layer ---
gdb_path = "/Users/brianbare/Downloads/Deep_ML/NEWHANOVER.gdb"
gdf = gpd.read_file(gdb_path, layer='NEWHANOVER')

# --- Convert to lon/lat if needed ---
if gdf.crs is None:
    gdf.set_crs(epsg=2264, inplace=True)  # example NC State Plane
gdf_wgs = gdf.to_crs(epsg=4326)  # lon/lat for PyDeck

# --- Prepare DataFrame for PyDeck ---
df = gdf_wgs[['geometry', 'Full_Address', 'Place_Type']].copy()
df['lon'] = df.geometry.x
df['lat'] = df.geometry.y

# --- Load DEM and sample elevation at each point ---
dem_path = "/Users/brianbare/Downloads/Deep_ML/newhanover-DEM03/newhanover-DEM03.tif"

with rasterio.open(dem_path) as src:
    # Reproject points from WGS84 to DEM CRS
    xs, ys = transform(
        'EPSG:4326',  # from lon/lat
        src.crs,      # DEM CRS
        df['lon'].values,
        df['lat'].values
    )
    # Sample DEM at each coordinate
    df['elevation'] = [val[0] for val in src.sample(zip(xs, ys))]

# Fill any missing elevations with 0
df['elevation'] = df['elevation'].fillna(0)

# --- Optional: highlight critical addresses ---
critical_keywords = ['school', 'hospital', 'clinic', 'fire', 'police', 'university', 'college']
def is_critical(address):
    if pd.isnull(address):
        return False
    return any(keyword in address.lower() for keyword in critical_keywords)
df['is_critical'] = df['Full_Address'].apply(is_critical)
df['color'] = df['is_critical'].apply(lambda x: [255, 0, 0] if x else [0, 0, 255])

# --- Streamlit UI ---
st.title("Interactive 3D Map of Wilmington Addresses with Elevation")

layer = pdk.Layer(
    "ColumnLayer",              # vertical columns
    data=df,
    get_position='[lon, lat]',
    get_elevation='elevation',  # height in meters
    elevation_scale=10,         # scale factor for visualization
    radius=50,                  # column radius in meters
    get_fill_color='color',
    pickable=True,
    auto_highlight=True
)

view_state = pdk.ViewState(
    longitude=-77.9447,
    latitude=34.2257,
    zoom=11,
    pitch=45,  # tilt camera for 3D
    bearing=0
)

tooltip = {"html": "<b>Address:</b> {Full_Address}<br><b>Elevation:</b> {elevation}m"}

r = pdk.Deck(layers=[layer], initial_view_state=view_state, tooltip=tooltip)

st.pydeck_chart(r)
