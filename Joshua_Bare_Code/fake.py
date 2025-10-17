# streamlit_hurricane_predictor.py
import streamlit as st
import geopandas as gpd
import pydeck as pdk
import pandas as pd
import rasterio
from rasterio.warp import transform
import numpy as np
import joblib
import json
from pathlib import Path

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
df['is_critical'] = df['Full_Address'].apply(is_critical) if 'is_ritical' in globals() else df['Full_Address'].apply(is_critical)
df['color'] = df['is_critical'].apply(lambda x: [255, 0, 0] if x else [0, 0, 255])

# ----------------------------
# Load ML artifacts (model + feature schema + medians)
# ----------------------------
ART_DIR = Path(".")  # put your artifacts in the same folder as this script
MODEL_PATH = ART_DIR / "/Users/brianbare/PycharmProjects/DeepMLProj/hurricane_rf_model.joblib"
FEAT_PATH = ART_DIR / "/Users/brianbare/PycharmProjects/DeepMLProj/feature_columns.txt"
MEDIANS_PATH = ART_DIR / "/Users/brianbare/PycharmProjects/DeepMLProj/training_medians.json"

@st.cache_resource
def load_model_and_artifacts():
    rf = joblib.load(MODEL_PATH)
    with open(FEAT_PATH) as f:
        feature_cols = [ln.strip() for ln in f if ln.strip()]
    with open(MEDIANS_PATH) as f:
        train_medians = pd.Series(json.load(f))
    return rf, feature_cols, train_medians

rf, FEATURE_COLS, TRAIN_MEDIANS = load_model_and_artifacts()

# ----------------------------
# Streamlit UI
# ----------------------------
st.title("🌪 Wilmington Hurricane Impact Simulator")

st.sidebar.header("Hurricane Builder")
st.sidebar.write("Adjust parameters to generate a hurricane and predict its affected area.")

# Location (defaults to Wilmington, NC)
st.sidebar.subheader("Storm Location & Time")
center_lat = st.sidebar.number_input("Center Latitude (°N + / °S −)", value=34.2257, step=0.01, format="%.5f")
center_lon = st.sidebar.number_input("Center Longitude (°E + / °W −)", value=-77.9447, step=0.01, format="%.5f")

year  = st.sidebar.number_input("Year", value=2005, step=1)
month = st.sidebar.slider("Month", 1, 12, 9)
day   = st.sidebar.slider("Day", 1, 31, 20)
hour  = st.sidebar.slider("Hour (0–23)", 0, 23, 12)

st.sidebar.subheader("Storm Metrics")
max_winds = st.sidebar.slider("Max Winds (kt)", 0, 200, 100)
central_pressure = st.sidebar.slider("Central Pressure (mb)", 850, 1050, 960)
rmw = st.sidebar.slider("Radius of Maximum Winds (nm)", 0, 150, 25)
oci = st.sidebar.slider("OCI_mb (Outer Core Pressure) (mb)", 850, 1050, 1010)
sshws = st.sidebar.slider("SSHWS (Category 0–5)", 0, 5, 2)

st.sidebar.caption("These inputs feed the trained RandomForest model. "
                   "The predicted Size_nm is visualized as a circle radius.")

# ----------------------------
# Build a single-row feature frame to match training schema
# ----------------------------
def build_features_for_model():
    row = {
        "Lat": center_lat,
        "Long": center_lon,
        "Central_Pressure": float(central_pressure),
        "Max_Winds_kt": float(max_winds),
        "RMW_nm": float(rmw),
        "OCI_mb": float(oci),
        "SSHWS": float(sshws),
        "Year": int(year),
        "Month": int(month),
        "Day": int(day),
        "Hour": int(hour),
        # Derived features used in training:
        "Abs_Lat": abs(center_lat),
        "Wind_to_Pressure": (float(max_winds) / (float(central_pressure) + 1e-6)) if central_pressure else np.nan,
    }

    # Align with training columns (order + missing handling)
    X_new = pd.DataFrame([{c: row.get(c, np.nan) for c in FEATURE_COLS}])
    X_new = X_new.apply(pd.to_numeric, errors="coerce").fillna(TRAIN_MEDIANS).fillna(0.0)
    return X_new, row

X_new, debug_row = build_features_for_model()

# ----------------------------
# Predict with your trained RF model
# ----------------------------
predicted_size = float(rf.predict(X_new)[0])  # Size in nautical miles (nm)

# ----------------------------
# Visualization
# ----------------------------
wilmington_center = (center_lon, center_lat)

# Hurricane circle (purple). Convert nm -> meters: 1 nm = 1852 m
hurricane_df = pd.DataFrame({
    'lon': [wilmington_center[0]],
    'lat': [wilmington_center[1]],
    'radius': [predicted_size * 1852.0],  # meters
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
**Predicted Size (radius):** {predicted_size:.2f} nm  
**Center:** ({center_lat:.4f}, {center_lon:.4f})  
**Visualization:** Purple circle (radius = predicted size × 1852 meters)
""")

with st.expander("🔧 Model feature values used for this prediction"):
    st.dataframe(pd.DataFrame([debug_row]))
