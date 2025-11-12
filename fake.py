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

# Artifacts (kept exactly as you had them; absolute paths are fine)
ART_DIR = Path(".")
MODEL_PATH = ART_DIR / "/Users/dimitrimontgomery/Downloads/CSC 432/Demo/hurricane_model.joblib"  # UPDATED: use the new neural-net joblib
FEAT_PATH = ART_DIR / "/Users/dimitrimontgomery/Downloads/CSC 432/Demo/feature_columns.txt"
MEDIANS_PATH = ART_DIR / "/Users/dimitrimontgomery/Downloads/CSC 432/Demo/training_medians.json"

# Wilmington center
CENTER_LAT, CENTER_LON = 34.2257, -77.9447
NM_TO_M = 1852.0  # nautical miles → meters

# ----------------------------------------------
# Category presets (SSHWS → dropdown value sets)
# ----------------------------------------------
CAT_PRESETS = {
    0: {"Max_Winds_kt": list(range(20, 64, 5)), "Central_Pressure": list(range(990, 1011, 5)), "RMW_nm": list(range(25, 70, 5)), "OCI_mb": list(range(120, 300, 10))},
    1: {"Max_Winds_kt": list(range(64, 83, 2)), "Central_Pressure": list(range(980, 1001, 5)), "RMW_nm": list(range(20, 60, 5)), "OCI_mb": list(range(140, 320, 10))},
    2: {"Max_Winds_kt": list(range(83, 96, 2)), "Central_Pressure": list(range(965, 980, 5)), "RMW_nm": list(range(15, 50, 5)), "OCI_mb": list(range(150, 330, 10))},
    3: {"Max_Winds_kt": list(range(96, 113, 2)), "Central_Pressure": list(range(945, 965, 5)), "RMW_nm": list(range(10, 40, 5)), "OCI_mb": list(range(160, 340, 10))},
    4: {"Max_Winds_kt": list(range(113, 137, 2)), "Central_Pressure": list(range(920, 945, 5)), "RMW_nm": list(range(8, 35, 5)), "OCI_mb": list(range(170, 350, 10))},
    5: {"Max_Winds_kt": list(range(137, 181, 2)), "Central_Pressure": list(range(870, 921, 5)), "RMW_nm": list(range(5, 30, 5)), "OCI_mb": list(range(180, 360, 10))},
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
    # Ensure columns exist even if empty
    empty_cols = ["lon", "lat", "Full_Address", "elevation", "color"]
    if not _exists(gdb_path):
        st.warning(f"📂 Address GDB not found at:\n`{gdb_path}`\n"
                   "Map will render without address points. Update the path or download/sync the dataset.")
        return pd.DataFrame(columns=empty_cols)

    try:
        gdf = gpd.read_file(gdb_path, layer="NEWHANOVER")
        if gdf.crs is None:
            gdf.set_crs(epsg=2264, inplace=True)  # example NC State Plane
        gdf = gdf.to_crs(epsg=4326)

        df = gdf[["geometry", "Full_Address"]].copy()
        df["lon"] = df.geometry.x
        df["lat"] = df.geometry.y
        df["color"] = [[0, 0, 255]] * len(df)   # blue points
        df["elevation"] = 0.0  # placeholder elevation; attach later
        return df[empty_cols]
    except Exception as e:
        st.error(f"Failed to read GDB layer 'NEWHANOVER' from:\n`{gdb_path}`\n\n**Error:** {e}")
        return pd.DataFrame(columns=empty_cols)

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
    Load ONLY the new neural-network joblib model.
    - If feature/median artifacts are present, use them.
    - If they are missing, provide a sensible default feature list and zero medians
      so predictions can still run without warnings.
    """
    try:
        if not _exists(MODEL_PATH):
            st.warning(f"🧠 Model not found at `{MODEL_PATH}`.\nPrediction will be disabled until this file is available.")
            return None, [], pd.Series(dtype=float)

        model = joblib.load(MODEL_PATH)

        feature_cols = []
        train_medians = pd.Series(dtype=float)

        if _exists(FEAT_PATH):
            with open(FEAT_PATH) as f:
                feature_cols = [ln.strip() for ln in f if ln.strip()]

        if _exists(MEDIANS_PATH):
            with open(MEDIANS_PATH) as f:
                train_medians = pd.Series(json.load(f))

        if not feature_cols:
            feature_cols = [
                "Lat", "Long", "Central_Pressure", "Max_Winds_kt",
                "RMW_nm", "OCI_mb", "SSHWS", "Abs_Lat", "Wind_to_Pressure",
            ]
        if train_medians.empty:
            train_medians = pd.Series({c: 0.0 for c in feature_cols})

        return model, feature_cols, train_medians

    except Exception as e:
        st.error(f"Failed to load model. Prediction disabled.\n\n**Error:** {e}")
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

# --- NEW: Basemap that doesn't require Mapbox token ---
# This guarantees the map canvas renders even without MAPBOX_API_KEY.
base_tile_layer = pdk.Layer(
    "TileLayer",
    data="https://c.tile.openstreetmap.org/{z}/{x}/{y}.png",
    min_zoom=0,
    max_zoom=19,
    tile_size=256,
)

# Base addresses layer (always shown) — pickable for tooltip
address_layer = pdk.Layer(
    "ScatterplotLayer",
    data=df,
    get_position="[lon, lat]",
    get_radius=30,              # small for perf
    get_fill_color="color",
    pickable=True,
    auto_highlight=True,
)

layers = [base_tile_layer, address_layer]  # include basemap FIRST

# Only compute model + circle AFTER Predict, and only if model is available
rf, FEATURE_COLS, TRAIN_MEDIANS = load_model_and_artifacts()
predicted_size_nm = None
radius_nm_used = None

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
        X_new = pd.DataFrame([{c: row.get(c, np.nan) for c in FEATURE_COLS}])
        X_new = X_new.apply(pd.to_numeric, errors="coerce").fillna(TRAIN_MEDIANS).fillna(0.0)

        X_use = X_new
        try:
            n_expected = None
            if hasattr(rf, "input_shape") and rf.input_shape is not None:
                n_expected = int(rf.input_shape[-1])
            elif hasattr(rf, "n_features_in_"):
                n_expected = int(rf.n_features_in_)
            if n_expected is not None:
                cur_n = X_new.shape[1]
                if cur_n < n_expected:
                    pad = pd.DataFrame(
                        np.zeros((1, n_expected - cur_n), dtype=float),
                        columns=[f"__pad_{i}" for i in range(n_expected - cur_n)]
                    )
                    X_use = pd.concat([X_new.reset_index(drop=True), pad], axis=1)
                elif cur_n > n_expected:
                    X_use = X_new.iloc[:, :n_expected]
        except Exception:
            X_use = X_new

        try:
            y_pred = rf.predict(np.asarray(X_use).astype("float32"))
            predicted_size_nm = float(np.asarray(y_pred).reshape(-1)[0])

            radius_nm_used = predicted_size_nm / 2.0
            radius_m = radius_nm_used * NM_TO_M

            hurricane_df = pd.DataFrame({
                "lon": [CENTER_LON],
                "lat": [CENTER_LAT],
                "radius_m": [radius_m],
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
            st.error(f"Model failed to predict. Check your artifacts and scikit-learn/TensorFlow versions.\n\n**Error:** {e}")

# View state + render (with tooltip showing elevation)
view_state = pdk.ViewState(
    longitude=CENTER_LON,
    latitude=CENTER_LAT,
    zoom=10,
    pitch=45,
)

tooltip = {
    "html": "<b>Address:</b> {Full_Address}<br/><b>Elevation (m):</b> {elevation}",
    "style": {"backgroundColor": "white", "color": "black"}
}

# --- IMPORTANT: disable Mapbox style so no token is needed; use OSM TileLayer instead ---
deck = pdk.Deck(
    layers=layers,
    initial_view_state=view_state,
    tooltip=tooltip,
    map_style=None,  # <- key change so it won't try to use Mapbox
)

st.pydeck_chart(deck, use_container_width=True)

# Info panel
if predicted_size_nm is not None:
    st.markdown(
        f"### Predicted Hurricane **Radius**: **{radius_nm_used:.2f} nm**  \n"
        f"*(purple circle shows radius = ½ × model size; model size was {predicted_size_nm:.2f} nm)*"
    )
else:
    st.info("Pick a **category** and inputs in the sidebar, then click **Predict Hurricane Impact** to draw the radius.")

with st.expander("Notes / Performance"):
    st.markdown(
        "- App caches address data, elevation sampling, and model artifacts to avoid reloading on each interaction.\n"
        "- If you see scikit-learn/TensorFlow version warnings when loading the model, either pin the versions used to save it "
        'or re-save the model with your current environment.'
    )
