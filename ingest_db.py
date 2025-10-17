# streamlit_hurricane_predictor.py
import streamlit as st
import geopandas as gpd
import pydeck as pdk
import pandas as pd
import rasterio
from rasterio.warp import transform
import numpy as np
import joblib  # for loading your ML model
from shapely.geometry import Point

# ----------------------------
# Load GDB layer (addresses)
# ----------------------------
gdb_path = "/Users/brianbare/Downloads/Deep_ML/NEWHANOVER.gdb"
gdf = gpd.read_file(gdb_path, layer='NEWHANOVER')

if gdf.crs is None:
    gdf.set_crs(epsg=2264, inplace=True)
gdf_wgs = gdf.to_crs(epsg=4326)

df = gdf_wgs[['geometry', 'Full_Address', 'Place_Type']].copy()
df['lon'] = df.geometry.x
df['lat'] = df.geometry.y

# ----------------------------
# Load DEM and extract elevation
# ----------------------------
dem_path = "/Users/brianbare/Downloads/Deep_ML/newhanover-DEM03/newhanover-DEM03.tif"

with rasterio.open(dem_path) as src:
    xs, ys = transform('EPSG:4326', src.crs, df['lon'].values, df['lat'].values)
    df['elevation'] = [val[0] for val in src.sample(zip(xs, ys))]

df['elevation'] = df['elevation'].fillna(0)

# ----------------------------
# Mark critical addresses
# ----------------------------
critical_keywords = ['school', 'hospital', 'clinic', 'fire', 'police', 'university', 'college']
def is_critical(address):
    if pd.isnull(address):
        return False
    return any(keyword in address.lower() for keyword in critical_keywords)
df['is_critical'] = df['Full_Address'].apply(is_critical)
df['color'] = df['is_critical'].apply(lambda x: [255, 0, 0] if x else [0, 0, 255])

# ----------------------------
# Streamlit UI
# ----------------------------
st.title("🌪 Wilmington Hurricane Impact Simulator")

st.sidebar.header("Hurricane Builder")
st.sidebar.write("Adjust parameters to generate a hurricane and predict its affected area.")

# User inputs for hurricane parameters
max_winds = st.sidebar.slider("Max Winds (kt)", 50, 200, 100)
central_pressure = st.sidebar.slider("Central Pressure (mb)", 880, 1020, 960)
rmw = st.sidebar.slider("Radius of Maximum Winds (nm)", 5, 100, 25)
oci = st.sidebar.slider("Outer Core Intensity (nm)", 50, 400, 150)
sea_surface_temp = st.sidebar.slider("Sea Surface Temp (°C)", 20.0, 32.0, 28.0)

# Placeholder model (you can load your trained one)
# model = joblib.load("path_to_your_model.pkl")
# X_input = np.array([[max_winds, central_pressure, rmw, oci, sea_surface_temp]])
# predicted_size = model.predict(X_input)[0]

# For demonstration, we’ll create a mock prediction:
predicted_size = max(10, (max_winds - (central_pressure - 900) / 5 + sea_surface_temp * 2) / 5)

# ----------------------------
# Visualization
# ----------------------------
wilmington_center = (-77.9447, 34.2257)

# Hurricane circle (purple)
hurricane_df = pd.DataFrame({
    'lon': [wilmington_center[0]],
    'lat': [wilmington_center[1]],
    'radius': [predicted_size * 1000],  # scale nm -> meters (approx)
    'color': [[128, 0, 128, 80]]  # semi-transparent purple
})

# Layers
address_layer = pdk.Layer(
    "ColumnLayer",
    data=df,
    get_position='[lon, lat]',
    get_elevation='elevation',
    elevation_scale=10,
    radius=50,
    get_fill_color='color',
    pickable=True,
    auto_highlight=True
)

hurricane_layer = pdk.Layer(
    "ScatterplotLayer",
    data=hurricane_df,
    get_position='[lon, lat]',
    get_radius='radius',
    get_fill_color='color',
    stroked=True,
    filled=True,
    line_width_min_pixels=2,
)

view_state = pdk.ViewState(
    longitude=wilmington_center[0],
    latitude=wilmington_center[1],
    zoom=9,
    pitch=45,
)

tooltip = {
    "html": "<b>Address:</b> {Full_Address}<br><b>Elevation:</b> {elevation} m"
}

st.pydeck_chart(pdk.Deck(
    layers=[address_layer, hurricane_layer],
    initial_view_state=view_state,
    tooltip=tooltip
))

# ----------------------------
# Results Display
# ----------------------------
st.markdown(f"""
### 🌀 Predicted Hurricane Impact
**Predicted 'Size Affected':** {predicted_size:.2f} nm  
**Epicenter:** Wilmington, NC  
**Visualization:** Purple circle scaled to predicted radius
""")
