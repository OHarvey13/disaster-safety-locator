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
# ----------------------------
# Streamlit UI - Dramatic Hurricane Inputs
# ----------------------------
st.title("🌪 Wilmington Hurricane Impact Simulator")
st.sidebar.header("Hurricane Builder")
st.sidebar.write("Select storm parameters to simulate a dramatic hurricane impact.")

# Predefined location: Wilmington, NC
center_lat, center_lon = 34.2257, -77.9447

st.sidebar.subheader("Storm Metrics (Dropdown Inputs)")

# Dropdown menus for metrics
max_winds = st.sidebar.selectbox("Max Winds (kt)", options=list(range(0, 201, 5)), index=20)
central_pressure = st.sidebar.selectbox("Central Pressure (mb)", options=list(range(850, 1051, 5)), index=21)
rmw = st.sidebar.selectbox("Radius of Maximum Winds (nm)", options=list(range(0, 151, 5)), index=5)
oci = st.sidebar.selectbox("OCI_mb (Outer Core Pressure)", options=list(range(850, 1051, 5)), index=32)
sshws = st.sidebar.selectbox("SSHWS (Category)", options=[0, 1, 2, 3, 4, 5], index=2)

# Predict button
predict_button = st.sidebar.button("🔮 Predict Hurricane Impact")

# ----------------------------
# Prediction + Visualization
# ----------------------------
# ----------------------------
# Prediction + Visualization
# ----------------------------
if predict_button:
    # Build features for model
    row = {
        "Lat": center_lat,
        "Long": center_lon,
        "Central_Pressure": float(central_pressure),
        "Max_Winds_kt": float(max_winds),
        "RMW_nm": float(rmw),
        "OCI_mb": float(oci),
        "SSHWS": float(sshws),
        "Abs_Lat": abs(center_lat),
        "Wind_to_Pressure": (float(max_winds) / (float(central_pressure) + 1e-6)) if central_pressure else np.nan,
    }
    X_new = pd.DataFrame([{c: row.get(c, np.nan) for c in FEATURE_COLS}])
    X_new = X_new.apply(pd.to_numeric, errors="coerce").fillna(TRAIN_MEDIANS).fillna(0.0)

    predicted_size = float(rf.predict(X_new)[0])

    # Normal visualization: radius matches predicted size (in meters)
    hurricane_df = pd.DataFrame({
        'lon': [center_lon],
        'lat': [center_lat],
        'radius': [predicted_size * 1852.0],  # convert nm to meters
        'color': [[128, 0, 128, 80]]
    })

    address_layer = pdk.Layer(
        "ColumnLayer",
        data=df,
        get_position='[lon, lat]',
        get_elevation='elevation',
        elevation_scale=1,
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
        longitude=center_lon,
        latitude=center_lat,
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

    st.markdown(f"""
    ### 🌀 Predicted Hurricane Impact
    **Predicted Size (radius):** {predicted_size:.2f} nm  
    **Visualization:** Purple circle (scaled to predicted size)
    """)

    with st.expander("🔧 Model feature values used for this prediction"):
        st.dataframe(pd.DataFrame([row]))

else:
    st.info("👆 Select parameters from the dropdowns, then click **Predict Hurricane Impact** to simulate.")
