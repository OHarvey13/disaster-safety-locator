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
MODEL_PATH = ART_DIR / "/Users/dimitrimontgomery/Downloads/CSC 432/Demo/hurricane_rf_model.joblib"  # UPDATED: use the new neural-net joblib
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
df["ring_label"] = "Address point"
df["radius_m"] = 0.0
df["radius_label"] = "—"
df["tooltip_html"] = (
    "<b>Address point</b><br/><b>Details:</b> "
    + df["Full_Address"].fillna("Address").astype(str)
    + "<br/><b>Elevation (m):</b> "
    + df["elevation"].fillna(0.0).round(2).astype(str)
)

# Sidebar
st.sidebar.header("Hurricane Builder")

def _cat_default(cat: int, key: str) -> float:
    vals = CAT_PRESETS[cat][key]
    return float(vals[len(vals) // 2])

def _apply_defaults(cat: int):
    st.session_state["sshws"] = cat
    st.session_state["max_winds"] = _cat_default(cat, "Max_Winds_kt")
    st.session_state["central_pressure"] = _cat_default(cat, "Central_Pressure")
    st.session_state["rmw"] = _cat_default(cat, "RMW_nm")
    st.session_state["oci"] = _cat_default(cat, "OCI_mb")

if "sshws" not in st.session_state:
    _apply_defaults(2)

sshws = st.sidebar.selectbox(
    "SSHWS (Category)",
    options=[0, 1, 2, 3, 4, 5],
    key="sshws",
)

opts = CAT_PRESETS[sshws]

if st.sidebar.button("Use category defaults"):
    _apply_defaults(sshws)

def _clamp(val: float, arr: list[int]) -> float:
    return float(min(max(val, float(min(arr))), float(max(arr))))

def _step(arr: list[int]) -> float:
    return float(arr[1] - arr[0]) if len(arr) > 1 else 1.0


def _haversine_m(lat1, lon1, lat2, lon2) -> np.ndarray:
    """
    Great-circle distance in meters for arrays/scalars of lat/lon.
    Vectorized so we can score many address points against the impact center.
    """
    r_earth = 6371000.0
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat / 2.0) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2.0) ** 2
    return 2 * r_earth * np.arcsin(np.sqrt(a))


def _extract_click_coords(evt):
    """
    Try to pull lon/lat from a pydeck chart event (works across new/old Streamlit chart-events APIs).
    Accepts dict-like or attribute-style payloads.
    """
    if evt is None:
        return None

    # Prefer dict-style access first
    if isinstance(evt, dict):
        lat = evt.get("lat") or evt.get("latitude")
        lon = evt.get("lon") or evt.get("lng") or evt.get("longitude")
        if lat is not None and lon is not None:
            return float(lat), float(lon)

    # Then inspect common attributes
    for attr in ("point", "coordinate", "coordinates", "location"):
        if hasattr(evt, attr):
            data = getattr(evt, attr)
            if isinstance(data, dict):
                lat = data.get("lat") or data.get("latitude")
                lon = data.get("lon") or data.get("lng") or data.get("longitude")
                if lat is not None and lon is not None:
                    return float(lat), float(lon)
            if isinstance(data, (list, tuple)) and len(data) >= 2:
                # Assume [lon, lat]
                return float(data[1]), float(data[0])

    return None


def _extract_view_state(evt):
    """Pull lon/lat from a view-state change event (map drag/pan)."""
    if evt is None:
        return None
    # Sometimes events are lists (multiple events); just grab the last one
    if isinstance(evt, (list, tuple)) and evt:
        evt = evt[-1]
    if not isinstance(evt, dict):
        return None
    vs = evt.get("view_state") or evt.get("viewState") or evt.get("view_state_change")
    if isinstance(vs, dict):
        lat = vs.get("latitude")
        lon = vs.get("longitude")
        if lat is not None and lon is not None:
            return float(lat), float(lon)
    return None

# Impact center (default Wilmington; click map or edit numbers)
if "impact_lat" not in st.session_state:
    st.session_state["impact_lat"] = CENTER_LAT
if "impact_lon" not in st.session_state:
    st.session_state["impact_lon"] = CENTER_LON

st.sidebar.subheader("Impact center")
st.sidebar.caption("Click on the map to reposition the rings, or fine-tune below.")
st.session_state["impact_lat"] = st.sidebar.number_input(
    "Latitude",
    min_value=-90.0,
    max_value=90.0,
    value=float(st.session_state["impact_lat"]),
    step=1.0,
    format="%.5f",
)
st.session_state["impact_lon"] = st.sidebar.number_input(
    "Longitude",
    min_value=-180.0,
    max_value=180.0,
    value=float(st.session_state["impact_lon"]),
    step=1.0,
    format="%.5f",
)
if st.sidebar.button("Reset to Wilmington"):
    st.session_state["impact_lat"] = CENTER_LAT
    st.session_state["impact_lon"] = CENTER_LON

impact_lat = float(st.session_state["impact_lat"])
impact_lon = float(st.session_state["impact_lon"])

st.session_state["max_winds"] = _clamp(st.session_state.get("max_winds", _cat_default(sshws, "Max_Winds_kt")), opts["Max_Winds_kt"])
st.session_state["central_pressure"] = _clamp(st.session_state.get("central_pressure", _cat_default(sshws, "Central_Pressure")), opts["Central_Pressure"])
st.session_state["rmw"] = _clamp(st.session_state.get("rmw", _cat_default(sshws, "RMW_nm")), opts["RMW_nm"])
st.session_state["oci"] = _clamp(st.session_state.get("oci", _cat_default(sshws, "OCI_mb")), opts["OCI_mb"])

max_winds = st.sidebar.slider(
    "Max Winds (kt)",
    min_value=float(min(opts["Max_Winds_kt"])),
    max_value=float(max(opts["Max_Winds_kt"])),
    value=st.session_state["max_winds"],
    step=_step(opts["Max_Winds_kt"]),
)
central_pressure = st.sidebar.slider(
    "Central Pressure (mb)",
    min_value=float(min(opts["Central_Pressure"])),
    max_value=float(max(opts["Central_Pressure"])),
    value=st.session_state["central_pressure"],
    step=_step(opts["Central_Pressure"]),
)
rmw = st.sidebar.slider(
    "Radius of Max Winds (nm)",
    min_value=float(min(opts["RMW_nm"])),
    max_value=float(max(opts["RMW_nm"])),
    value=st.session_state["rmw"],
    step=_step(opts["RMW_nm"]),
)
oci = st.sidebar.slider(
    "Outer Core Pressure Δ (OCI_mb)",
    min_value=float(min(opts["OCI_mb"])),
    max_value=float(max(opts["OCI_mb"])),
    value=st.session_state["oci"],
    step=_step(opts["OCI_mb"]),
)

st.session_state["max_winds"] = max_winds
st.session_state["central_pressure"] = central_pressure
st.session_state["rmw"] = rmw
st.session_state["oci"] = oci

st.sidebar.subheader("Layers")
show_addresses = st.sidebar.toggle("Show addresses", value=True)
show_rings = st.sidebar.toggle("Show impact rings", value=True)
color_by_risk = st.sidebar.toggle("Color addresses by risk", value=False, help="Uses predicted radius + low elevation.")
low_elev_ref = 5.0
if color_by_risk:
    low_elev_ref = st.sidebar.slider(
        "Low-elevation emphasis (m)",
        min_value=0.0,
        max_value=15.0,
        value=5.0,
        step=0.5,
        help="Risk increases when elevation is below this height.",
    )

predict_button = st.sidebar.button("Predict Hurricane Impact")

# --- NEW: Basemap that doesn't require Mapbox token ---
# This guarantees the map canvas renders even without MAPBOX_API_KEY.
ESRI_WORLD_TOPO = "https://services.arcgisonline.com/ArcGIS/rest/services/World_Topo_Map/MapServer/tile/{z}/{y}/{x}"
base_tile_layer = pdk.Layer(
    "TileLayer",
    data=ESRI_WORLD_TOPO,  # topo basemap with roads/rivers/labels/waterbodies
    min_zoom=0,
    max_zoom=19,
    tile_size=256,
)

ring_layers = []

# Only compute model + circle AFTER Predict, and only if model is available
rf, FEATURE_COLS, TRAIN_MEDIANS = load_model_and_artifacts()
predicted_size_nm = st.session_state.get("predicted_size_nm")
radius_nm_used = st.session_state.get("predicted_radius_nm")
radius_m = None

if predict_button:
    if rf is None or not FEATURE_COLS:
        st.error("Prediction unavailable: model artifacts not found. See warnings above for the expected paths.")
        st.session_state["predicted_size_nm"] = None
        st.session_state["predicted_radius_nm"] = None
    else:
        row = {
            "Lat": impact_lat,
            "Long": impact_lon,
            "Central_Pressure": float(central_pressure),
            "Max_Winds_kt": float(max_winds),
            "RMW_nm": float(rmw),
            "OCI_mb": float(oci),
            "SSHWS": float(sshws),
            "Abs_Lat": abs(impact_lat),
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
            st.session_state["predicted_size_nm"] = predicted_size_nm
            st.session_state["predicted_radius_nm"] = radius_nm_used
        except Exception as e:
            st.session_state["predicted_size_nm"] = None
            st.session_state["predicted_radius_nm"] = None
            st.error(f"Model failed to predict. Check your artifacts and scikit-learn/TensorFlow versions.\n\n**Error:** {e}")

if radius_nm_used is not None:
    radius_m = radius_nm_used * NM_TO_M
    zones = [
        ("Outer ring (full radius)", radius_m, [255, 215, 0, 20]),   # lighter fill for map readability
        ("Middle ring (~2/3)", radius_m * 0.66, [255, 140, 0, 35]),
        ("Inner ring (~1/3)", radius_m * 0.33, [200, 0, 0, 55]),
    ]
    for label, rad_m, color in zones:
        rad_nm = rad_m / NM_TO_M
        zone_df = pd.DataFrame({
            "lon": [impact_lon],
            "lat": [impact_lat],
            "radius_m": [rad_m],
            "color": [color],
            "Full_Address": [label],
            "ring_label": [label],
            "radius_label": [f"{rad_nm:.2f} nm"],
            "tooltip_html": [f"<b>{label}</b><br/>Radius: {rad_nm:.2f} nm"],
        })
        zone_layer = pdk.Layer(
            "ScatterplotLayer",
            data=zone_df,
            get_position="[lon, lat]",
            get_radius="radius_m",
            get_fill_color="color",
            stroked=True,
            filled=True,
            line_width_min_pixels=2,
            pickable=False,  # keep legend, but don't steal hover from addresses
        )
        ring_layers.append(zone_layer)

# Risk-aware address coloring + exposure counts
address_data = df.copy()
risk_summary = {
    "total": len(address_data),
    "in_radius": 0,
    "low_elev_in_radius": 0,
    "avg_elev_in_radius": None,
}

if color_by_risk and radius_m is None:
    st.info("Predict to color addresses by risk (combines distance to center and low elevation).")

if not address_data.empty and radius_m is not None:
    dists = _haversine_m(address_data["lat"].values, address_data["lon"].values, impact_lat, impact_lon)
    elev = address_data["elevation"].fillna(0.0).astype(float).values
    dist_factor = np.clip((radius_m - dists) / max(radius_m, 1e-6), 0, 1)
    elev_factor = np.clip((low_elev_ref - elev) / max(low_elev_ref, 1e-6), 0, 1)
    risk_score = dist_factor * elev_factor

    address_data["distance_m"] = dists
    address_data["risk_score"] = risk_score
    address_data["risk_level"] = pd.cut(
        risk_score,
        bins=[-0.01, 0.3, 0.6, 1.0],
        labels=["Low", "Medium", "High"],
    )

    color_map = {"High": (200, 0, 0), "Medium": (255, 140, 0), "Low": (0, 170, 0)}
    risk_level_str = address_data["risk_level"].astype(str).replace("nan", np.nan)
    default_color = list(color_map["Low"])
    mapped = risk_level_str.map(color_map)
    address_data["risk_color"] = mapped.apply(
        lambda c: list(c) if isinstance(c, tuple) and len(c) == 3 else default_color
    )

    risk_level_display = risk_level_str.fillna("Low")
    address_data["tooltip_html"] = (
        "<b>Address point</b><br/><b>Details:</b> "
        + address_data["Full_Address"].fillna("Address").astype(str)
        + "<br/><b>Elevation (m):</b> "
        + address_data["elevation"].fillna(0.0).round(2).astype(str)
        + "<br/><b>Distance to center (km):</b> "
        + (address_data["distance_m"] / 1000.0).round(2).astype(str)
        + "<br/><b>Risk:</b> "
        + risk_level_display.astype(str)
    )

    in_radius_mask = dists <= radius_m
    low_elev_mask = elev <= low_elev_ref
    risk_summary["in_radius"] = int(in_radius_mask.sum())
    low_elev_count = int((in_radius_mask & low_elev_mask).sum())
    risk_summary["low_elev_in_radius"] = min(low_elev_count, risk_summary["in_radius"])
    if in_radius_mask.any():
        risk_summary["avg_elev_in_radius"] = float(address_data.loc[in_radius_mask, "elevation"].mean())

if "risk_color" not in address_data:
    address_data["risk_color"] = address_data.get("color", [[0, 0, 255]] * len(address_data))

color_field = "risk_color" if (color_by_risk and radius_m is not None) else "color"
address_layer = pdk.Layer(
    "ScatterplotLayer",
    data=address_data,
    get_position="[lon, lat]",
    get_radius=30,
    get_fill_color=color_field,
    pickable=True,
    auto_highlight=True,
)

layers = [base_tile_layer]
if show_addresses:
    layers.append(address_layer)
if show_rings:
    layers.extend(ring_layers)

# Quick exposure metrics
metric_col1, metric_col2, metric_col3 = st.columns(3)
metric_col1.metric("Addresses loaded", risk_summary["total"])
metric_col2.metric("In predicted ring", risk_summary["in_radius"] if radius_m is not None else "—")
metric_col3.metric(
    f"Low elevation < {low_elev_ref:.1f} m in ring",
    "—",
)

# View state + render (with tooltip showing elevation)
view_state = pdk.ViewState(
    longitude=impact_lon,
    latitude=impact_lat,
    zoom=10,
    pitch=45,
)

tooltip = {
    "html": "{tooltip_html}",
    "style": {"backgroundColor": "white", "color": "black"}
}

# --- IMPORTANT: disable Mapbox style so no token is needed; use OSM TileLayer instead ---
deck = pdk.Deck(
    layers=layers,
    initial_view_state=view_state,
    tooltip=tooltip,
    map_style=None,  # <- key change so it won't try to use Mapbox
)

map_col, legend_col = st.columns([4, 1])
click_event = None
try:
    click_event = map_col.pydeck_chart(
        deck,
        use_container_width=True,
        height=600,
        on_click=True,
        on_view_state_change=True,
        key="impact_map",
    )
except TypeError:
    # Older Streamlit versions without chart events
    click_event = map_col.pydeck_chart(deck, use_container_width=True, height=600, key="impact_map")
map_col.caption("Tip: drag/pan the map or click to move the impact center. Coordinates also editable in the sidebar.")

coords = _extract_click_coords(click_event)
view_coords = _extract_view_state(click_event)

new_center = coords or view_coords
if new_center:
    new_lat, new_lon = new_center
    if (
        abs(new_lat - st.session_state["impact_lat"]) > 1e-9
        or abs(new_lon - st.session_state["impact_lon"]) > 1e-9
    ):
        st.session_state["impact_lat"] = new_lat
        st.session_state["impact_lon"] = new_lon
        # Force immediate refresh so the map recenters on the new spot
        rerun = getattr(st, "rerun", None) or getattr(st, "experimental_rerun", None)
        if rerun:
            rerun()

if radius_nm_used is not None:
    legend_html = f"""
    <div style="padding: 10px 12px; background: rgba(255,255,255,0.98); border: 1px solid #ddd; border-radius: 8px; font-size: 13px; color: #1f2d3d;">
      <strong>Impact Rings</strong><br/>
      <span style="display:inline-block;width:12px;height:12px;border-radius:50%;background:rgba(255,215,0,0.7);border:1px solid #b89b00;margin-right:6px;"></span>Outer (full radius) — {radius_nm_used:.2f} nm<br/>
      <span style="display:inline-block;width:12px;height:12px;border-radius:50%;background:rgba(255,140,0,0.7);border:1px solid #b86100;margin-right:6px;"></span>Middle (~2/3) — {radius_nm_used * 0.66:.2f} nm<br/>
      <span style="display:inline-block;width:12px;height:12px;border-radius:50%;background:rgba(200,0,0,0.8);border:1px solid #7a0000;margin-right:6px;"></span>Inner (~1/3) — {radius_nm_used * 0.33:.2f} nm<br/>
      <div style="margin-top:6px; color:#3b4a5a;">Toggles on the left let you hide/show address points or rings.</div>
    </div>
    """
else:
    legend_html = """
    <div style="padding: 10px 12px; background: rgba(255,255,255,0.98); border: 1px solid #ddd; border-radius: 8px; font-size: 13px; color: #1f2d3d;">
      <strong>Impact Rings</strong><br/>
      <span style="display:inline-block;width:12px;height:12px;border-radius:50%;background:rgba(255,215,0,0.7);border:1px solid #b89b00;margin-right:6px;"></span>Outer (full radius)<br/>
      <span style="display:inline-block;width:12px;height:12px;border-radius:50%;background:rgba(255,140,0,0.7);border:1px solid #b86100;margin-right:6px;"></span>Middle (~2/3)<br/>
      <span style="display:inline-block;width:12px;height:12px;border-radius:50%;background:rgba(200,0,0,0.8);border:1px solid #7a0000;margin-right:6px;"></span>Inner (~1/3)<br/>
      <div style="margin-top:6px; color:#3b4a5a;">Predict to see ring distances in nautical miles.</div>
    </div>
    """

legend_col.markdown(legend_html, unsafe_allow_html=True)

# Info panel
if predicted_size_nm is not None:
    st.markdown(
        f"### Predicted Hurricane **Radius**: **{radius_nm_used:.2f} nm**  \n"
        f"*(outer yellow = full radius; middle orange ~2/3; inner red ~1/3; model size before halving was {predicted_size_nm:.2f} nm)*"
    )
else:
    st.info("Pick a **category** and inputs in the sidebar, then click **Predict Hurricane Impact** to draw the radius.")

with st.expander("Notes / Performance"):
    st.markdown(
        "- App caches address data, elevation sampling, and model artifacts to avoid reloading on each interaction.\n"
        "- If you see scikit-learn/TensorFlow version warnings when loading the model, either pin the versions used to save it "
        'or re-save the model with your current environment.'
    )
