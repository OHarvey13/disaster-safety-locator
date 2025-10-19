# streamlit_hurricane_predictor.py  (robust: no circle until predict; category inputs; path checks; cache arg fix)
import os
from pathlib import Path
import json

import streamlit as st
import geopandas as gpd
import pydeck as pdk
import pandas as pd
import numpy as np
import joblib
import rasterio
from rasterio.warp import transform

# ----------------------------
# Original paths (UNCHANGED)
# ----------------------------
gdb_path = "/Users/dimitrimontgomery/Downloads/CSC 432/Demo/NEWHANOVER.gdb"
dem_path = "/Users/dimitrimontgomery/Downloads/CSC 432/Demo/newhanover-DEM03/newhanover-DEM03.tif"

# Artifacts (kept exactly as you had them; yes these are absolute paths joined to Path(".")) 
ART_DIR = Path(".")
MODEL_PATH = ART_DIR / "/Users/dimitrimontgomery/Downloads/CSC 432/Demo/hurricane_rf_model.joblib"
FEAT_PATH = ART_DIR / "/Users/dimitrimontgomery/Downloads/CSC 432/Demo/feature_columns.txt"
MEDIANS_PATH = ART_DIR / "/Users/dimitrimontgomery/Downloads/CSC 432/Demo/training_medians.json"

# Wilmington center
CENTER_LAT, CENTER_LON = 34.2257, -77.9447

# ----------------------------------------------
# Category presets (SSHWS → dropdown value sets)
# ----------------------------------------------
CAT_PRESETS = {
    0: {  # TS/TD bucketed as 0 here
        "Max_Winds_kt": list(range(20, 64, 5)),
        "Central_Pressure": list(range(990, 1011, 5)),
        "RMW_nm": list(range(25, 70, 5)),
        "OCI_mb": list(range(120, 300, 10)),
    },
    1: {  # 64–82 kt
        "Max_Winds_kt": list(range(64, 83, 2)),
        "Central_Pressure": list(range(980, 1001, 5)),
        "RMW_nm": list(range(20, 60, 5)),
        "OCI_mb": list(range(140, 320, 10)),
    },
    2: {  # 83–95 kt
        "Max_Winds_kt": list(range(83, 96, 2)),
        "Central_Pressure": list(range(965, 980, 5)),
        "RMW_nm": list(range(15, 50, 5)),
        "OCI_mb": list(range(150, 330, 10)),
    },
    3: {  # 96–112 kt
        "Max_Winds_kt": list(range(96, 113, 2)),
        "Central_Pressure": list(range(945, 965, 5)),
        "RMW_nm": list(range(10, 40, 5)),
        "OCI_mb": list(range(160, 340, 10)),
    },
    4: {  # 113–136 kt
        "Max_Winds_kt": list(range(113, 137, 2)),
        "Central_Pressure": list(range(920, 945, 5)),
        "RMW_nm": list(range(8, 35, 5)),
        "OCI_mb": list(range(170, 350, 10)),
    },
    5: {  # ≥137 kt
        "Max_Winds_kt": list(range(137, 181, 2)),
        "Central_Pressure": list(range(870, 921, 5)),
        "RMW_nm": list(range(5, 30, 5)),
        "OCI_mb": list(range(180, 360, 10)),
    },
}

def _exists(p: str | Path) -> bool:
    try:
        return Path(p).exists()
    except Exception:
        return False

# ----------------------------
# Caching
# ----------------------------
@st.cache_resource(show_spinner=False)
def load_addresses_df():
    """
    Load addresses from the FileGDB.
    If the GDB is missing/unavailable, return an empty DataFrame with required columns
    so the app still renders without crashing.
    """
    if not _exists(gdb_path):
        st.warning(f"📂 Address GDB not found at:\n`{gdb_path}`\n"
                   "Map will render without address points. Update the path or download/sync the dataset.")
        return pd.DataFrame(columns=["lon", "lat", "Full_Address", "elevation", "color"])

    try:
        gdf = gpd.read_file(gdb_path, layer="NEWHANOVER")
        if gdf.crs is None:
            gdf.set_crs(epsg=2264, inplace=True)  # example NC State Plane
        gdf = gdf.to_crs(epsg=4326)

        df = gdf[["geometry", "Full_Address"]].copy()
        df["lon"] = df.geometry.x
        df["lat"] = df.geometry.y
        df["color"] = [[0, 0, 255]] * len(df)   # blue points
        # placeholder elevation; attach later
        df["elevation"] = 0.0
        return df
    except Exception as e:
        st.error(f"Failed to read GDB layer 'NEWHANOVER' from:\n`{gdb_path}`\n\n**Error:** {e}")
        return pd.DataFrame(columns=["lon", "lat", "Full_Address", "elevation", "color"])

# IMPORTANT FIX:
# Prefix the parameter with '_' so Streamlit won't try to hash a (Geo)DataFrame.
@st.cache_resource(show_spinner=False)
def attach_elevation(_df_in: pd.DataFrame) -> pd.DataFrame:
    """
    Attach elevation from DEM. If DEM is missing or any error occurs, return df with elevation=0.
    The leading underscore prevents Streamlit from hashing this arg (GeoDataFrame/Pandas isn't hashable).
    """
    df = _df_in.copy()
    if df.empty:
        return df

    if not _exists(dem_path):
        st.info(f"🗺️ DEM not found at:\n`{dem_path}`\nContinuing with elevation=0.")
        return df

    try:
        with rasterio.open(dem_path) as src:
            xs, ys = transform("EPSG:4326", src.crs, df["lon"].values, df["lat"].values)
            df["elevation"] = [val[0] for val in src.sample(zip(xs, ys))]
        df["elevation"] = df["elevation"].fillna(0.0)
        return df
    except Exception as e:
        st.info(f"DEM could not be sampled from:\n`{dem_path}`\nContinuing with elevation=0.\n\nDetails: {e}")
        return df

@st.cache_resource(show_spinner=False)
def load_model_and_artifacts():
    """
    Load model + feature schema + training medians.
    If any artifact is missing/unloadable, return (None, [], Series()) so we can gracefully disable prediction.
    """
    try:
        if not (_exists(MODEL_PATH) and _exists(FEAT_PATH) and _exists(MEDIANS_PATH)):
            st.warning(
                "🧠 Model artifacts not found.\n\n"
                f"- MODEL: `{MODEL_PATH}`\n- FEATS: `{FEAT_PATH}`\n- MEDIANS: `{MEDIANS_PATH}`\n\n"
                "Prediction will be disabled until these files are available."
            )
            return None, [], pd.Series(dtype=float)

        rf = joblib.load(MODEL_PATH)
        with open(FEAT_PATH) as f:
            feature_cols = [ln.strip() for ln in f if ln.strip()]
        with open(MEDIANS_PATH) as f:
            train_medians = pd.Series(json.load(f))
        return rf, feature_cols, train_medians
    except Exception as e:
        st.error(f"Failed to load model artifacts. Prediction disabled.\n\n**Error:** {e}")
        return None, [], pd.Series(dtype=float)

# ----------------------------
# UI
# ----------------------------
st.set_page_config(page_title="Wilmington Hurricane Impact", layout="wide")
st.title("Wilmington Hurricane Impact Simulator")

# Base data (cached)
base_df = load_addresses_df()
df = attach_elevation(base_df)

# Sidebar
st.sidebar.header("Hurricane Builder")
sshws = st.sidebar.selectbox("SSHWS (Category)", options=[0, 1, 2, 3, 4, 5], index=2)

opts = CAT_PRESETS[sshws]
max_winds = st.sidebar.selectbox("Max Winds (kt)", options=opts["Max_Winds_kt"])
central_pressure = st.sidebar.selectbox("Central Pressure (mb)", options=opts["Central_Pressure"])
rmw = st.sidebar.selectbox("Radius of Max Winds (nm)", options=opts["RMW_nm"])
oci = st.sidebar.selectbox("Outer Core Pressure Δ (OCI_mb)", options=opts["OCI_mb"])

predict_button = st.sidebar.button("Predict Hurricane Impact")

# Base addresses layer (always shown)
address_layer = pdk.Layer(
    "ScatterplotLayer",
    data=df,
    get_position="[lon, lat]",
    get_radius=30,              # small for perf
    get_fill_color="color",
    pickable=True,
)
layers = [address_layer]       # no circle until Predict

# Only compute model + circle AFTER Predict, and only if model is available
rf, FEATURE_COLS, TRAIN_MEDIANS = load_model_and_artifacts()
predicted_size_nm = None

if predict_button:
    if rf is None or not FEATURE_COLS:
        st.error("Prediction unavailable: model artifacts not found. See warnings above for the expected paths.")
    else:
        row = {
            "Lat": CENTER_LAT,
            "Long": CENTER_LON,
            "Central_Pressure": float(central_pressure),
            "Max_Winds_kt": float(max_winds),
            "RMW_nm": float(rmw),
            "OCI_mb": float(oci),
            "SSHWS": float(sshws),
            "Abs_Lat": abs(CENTER_LAT),
            "Wind_to_Pressure": (float(max_winds) / (float(central_pressure) + 1e-6)),
        }
        # align to model features; fill missing from medians, then zeros
        X_new = pd.DataFrame([{c: row.get(c, np.nan) for c in FEATURE_COLS}])
        X_new = X_new.apply(pd.to_numeric, errors="coerce").fillna(TRAIN_MEDIANS).fillna(0.0)

        try:
            predicted_size_nm = float(rf.predict(X_new)[0])
            hurricane_df = pd.DataFrame({
                "lon": [CENTER_LON],
                "lat": [CENTER_LAT],
                "radius_m": [predicted_size_nm * 1852.0],  # nm → meters
                "color": [[128, 0, 128, 80]],
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
            layers.append(hurricane_layer)
        except Exception as e:
            st.error(f"Model failed to predict. Check your artifacts and scikit-learn version.\n\n**Error:** {e}")

# View state + render
view_state = pdk.ViewState(
    longitude=CENTER_LON,
    latitude=CENTER_LAT,
    zoom=10,
    pitch=45,
)
st.pydeck_chart(pdk.Deck(layers=layers, initial_view_state=view_state))

# Info panel
if predicted_size_nm is not None:
    st.markdown(
        f"### Predicted Hurricane Radius: **{predicted_size_nm:.2f} nm**  "
        "*(purple circle reflects the model output)*"
    )
else:
    st.info("Pick a **category** and inputs in the sidebar, then click **Predict Hurricane Impact** to draw the radius.")

with st.expander("Notes / Performance"):
    st.markdown(
        "- App caches address data, elevation sampling, and model artifacts to avoid reloading on each interaction.\n"
        "- If you see scikit-learn version warnings when loading the model, either pin the version used to save it "
        '(`pip install "scikit-learn==1.6.1"`) **or** re-save the model with your current version.'
    )









gdb_path = "/Users/dimitrimontgomery/Downloads/CSC 432/Demo/NEWHANOVER.gdb"
dem_path = "/Users/dimitrimontgomery/Downloads/CSC 432/Demo/newhanover-DEM03/newhanover-DEM03.tif"

# Artifacts (kept exactly as you had them; yes these are absolute paths joined to Path(".")) 
ART_DIR = Path(".")
MODEL_PATH = ART_DIR / "/Users/dimitrimontgomery/Downloads/CSC 432/Demo/hurricane_rf_model.joblib"
FEAT_PATH = ART_DIR / "/Users/dimitrimontgomery/Downloads/CSC 432/Demo/feature_columns.txt"
MEDIANS_PATH = ART_DIR / "/Users/dimitrimontgomery/Downloads/CSC 432/Demo/training_medians.json"