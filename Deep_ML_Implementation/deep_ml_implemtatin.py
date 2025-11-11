import os
import streamlit as st
import geopandas as gpd
import pandas as pd
import numpy as np
import joblib
import pydeck as pdk

# ---------------------------------
# SAFETY: Disable GDAL multithreading (prevents lock)
# ---------------------------------
os.environ["CPL_MULTITHREAD"] = "NO"
os.environ["GDAL_DISABLE_READDIR_ON_OPEN"] = "YES"
os.environ["USE_PYGEOS"] = "0"

# ---------------------------------
# PATHS
# ---------------------------------
GDB_PATH = "/Users/brianbare/Downloads/Deep_ML/NEWHANOVER.gdb"
MODEL_PATH = "/Users/brianbare/PycharmProjects/DeepMLProj/hurricane_model.joblib"

CENTER_LAT, CENTER_LON = 34.2257, -77.9447
NM_TO_M = 1852.0

# ---------------------------------
# CATEGORY PRESETS
# ---------------------------------
CAT_PRESETS = {
    0: {"Max_Winds_kt": list(range(20, 64, 5)), "Central_Pressure": list(range(990, 1011, 5)), "RMW_nm": list(range(25, 70, 5)), "OCI_mb": list(range(120, 300, 10))},
    1: {"Max_Winds_kt": list(range(64, 83, 2)), "Central_Pressure": list(range(980, 1001, 5)), "RMW_nm": list(range(20, 60, 5)), "OCI_mb": list(range(140, 320, 10))},
    2: {"Max_Winds_kt": list(range(83, 96, 2)), "Central_Pressure": list(range(965, 980, 5)), "RMW_nm": list(range(15, 50, 5)), "OCI_mb": list(range(150, 330, 10))},
    3: {"Max_Winds_kt": list(range(96, 113, 2)), "Central_Pressure": list(range(945, 965, 5)), "RMW_nm": list(range(10, 40, 5)), "OCI_mb": list(range(160, 340, 10))},
    4: {"Max_Winds_kt": list(range(113, 137, 2)), "Central_Pressure": list(range(920, 945, 5)), "RMW_nm": list(range(8, 35, 5)), "OCI_mb": list(range(170, 350, 10))},
    5: {"Max_Winds_kt": list(range(137, 181, 2)), "Central_Pressure": list(range(870, 921, 5)), "RMW_nm": list(range(5, 30, 5)), "OCI_mb": list(range(180, 360, 10))},
}

# ---------------------------------
# LOAD ADDRESS DATA
# ---------------------------------
@st.cache_data
def load_addresses():
    try:
        gdf = gpd.read_file(GDB_PATH, layer="NEWHANOVER")
        if gdf.crs is None:
            gdf.set_crs(epsg=2264, inplace=True)
        gdf = gdf.to_crs(epsg=4326)
        df = pd.DataFrame({
            "lon": gdf.geometry.x,
            "lat": gdf.geometry.y,
            "Full_Address": gdf.get("Full_Address", "Unknown")
        })
        # Assign placeholder elevation/colors since DEM can cause locks
        df["elevation_ft"] = np.random.uniform(0, 20, len(df))
        df["color"] = df["elevation_ft"].apply(lambda x: [255, 0, 0] if x <= 5 else [0, 0, 255])
        return df
    except Exception as e:
        st.error(f"Failed to read addresses: {e}")
        return pd.DataFrame(columns=["lon", "lat", "Full_Address", "elevation_ft", "color"])

# ---------------------------------
# LOAD MODEL
# ---------------------------------
@st.cache_resource
def load_model():
    try:
        return joblib.load(MODEL_PATH)
    except Exception as e:
        st.error(f"Failed to load model: {e}")
        return None

# ---------------------------------
# STREAMLIT APP
# ---------------------------------
st.set_page_config(page_title="Hurricane Impact Predictor", layout="wide")
st.title("🌪️ Wilmington Hurricane Impact Predictor")

addresses = load_addresses()
model = load_model()

# Sidebar inputs
st.sidebar.header("Hurricane Inputs")
sshws = st.sidebar.selectbox("Category (SSHWS)", options=[0, 1, 2, 3, 4, 5], index=2)
opts = CAT_PRESETS[sshws]
max_winds = st.sidebar.selectbox("Max Winds (kt)", options=opts["Max_Winds_kt"])
central_pressure = st.sidebar.selectbox("Central Pressure (mb)", options=opts["Central_Pressure"])
rmw = st.sidebar.selectbox("Radius of Max Winds (nm)", options=opts["RMW_nm"])
oci = st.sidebar.selectbox("Outer Core Pressure Δ (OCI_mb)", options=opts["OCI_mb"])
predict_button = st.sidebar.button("Predict Hurricane Impact")

predicted_size_nm, radius_nm = None, None

if predict_button:
    if model is None:
        st.error("Model not loaded. Check file path.")
    else:
        row = pd.DataFrame([{
            "Lat": CENTER_LAT,
            "Long": CENTER_LON,
            "Central_Pressure": float(central_pressure),
            "Max_Winds_kt": float(max_winds),
            "RMW_nm": float(rmw),
            "OCI_mb": float(oci),
            "SSHWS": float(sshws),
            "Abs_Lat": abs(CENTER_LAT),
            "Wind_to_Pressure": float(max_winds) / (float(central_pressure) + 1e-6),
        }])

        try:
            predicted_size_nm = float(model.predict(row)[0])
            radius_nm = predicted_size_nm / 2.0
            radius_m = radius_nm * NM_TO_M

            hurricane_df = pd.DataFrame({
                "lon": [CENTER_LON],
                "lat": [CENTER_LAT],
                "radius_m": [radius_m],
                "color": [[128, 0, 128, 60]],
            })

            hurricane_layer = pdk.Layer(
                "ScatterplotLayer",
                data=hurricane_df,
                get_position="[lon, lat]",
                get_radius="radius_m",
                get_fill_color="color",
                stroked=True,
                filled=True,
                line_width_min_pixels=2,
            )

            st.success(f"Predicted Hurricane Radius: **{radius_nm:.2f} nautical miles**")
        except Exception as e:
            st.error(f"Prediction failed: {e}")

# ---------------------------------
# MAP DISPLAY
# ---------------------------------
address_layer = pdk.Layer(
    "ScatterplotLayer",
    data=addresses,
    get_position="[lon, lat]",
    get_fill_color="color",
    get_radius=35,
    pickable=True,
    auto_highlight=True,
)

layers = [address_layer]
if predict_button and radius_nm is not None:
    layers.append(hurricane_layer)

tooltip = {
    "html": "<b>Address:</b> {Full_Address}<br/><b>Elevation (ft):</b> {elevation_ft}",
    "style": {"backgroundColor": "white", "color": "black"},
}

st.pydeck_chart(pdk.Deck(
    layers=layers,
    initial_view_state=pdk.ViewState(latitude=CENTER_LAT, longitude=CENTER_LON, zoom=10, pitch=45),
    tooltip=tooltip
))
