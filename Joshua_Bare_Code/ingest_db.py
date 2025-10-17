# streamlit_hurricane_predictor_with_model.py
import streamlit as st
import geopandas as gpd
import pydeck as pdk
import pandas as pd
import numpy as np
import re
import rasterio
from rasterio.warp import transform
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.dummy import DummyRegressor

# ======================
# 1. Load & Train Model
# ======================

@st.cache_resource
def load_and_train_model():
    csv_path = "/content/Hurricane_dataset - Sheet1.csv"

    df = pd.read_csv(csv_path)

    # --- Parsers ---
    def parse_lat(x):
        if pd.isna(x): return np.nan
        s = str(x).strip()
        m = re.match(r"^([+-]?\d+(?:\.\d+)?)([NnSs])$", s)
        if m:
            val = float(m.group(1))
            return val if m.group(2).upper()=="N" else -val
        try: return float(s)
        except: return np.nan

    def parse_lon(x):
        if pd.isna(x): return np.nan
        s = str(x).strip()
        m = re.match(r"^([+-]?\d+(?:\.\d+)?)([EeWw])$", s)
        if m:
            val = float(m.group(1))
            return -val if m.group(2).upper()=="W" else val
        try: return float(s)
        except: return np.nan

    def parse_rmw(x):
        if pd.isna(x): return np.nan
        s = str(x).strip()
        if s in {"---","--","-"}: return np.nan
        s = s.replace(",", "")
        try: return float(s)
        except: return np.nan

    def parse_pressure(x):
        if pd.isna(x): return np.nan
        s = str(x).strip().replace(",", "")
        try: return abs(float(s))
        except: return np.nan

    def parse_time_to_hour(t):
        if pd.isna(t): return np.nan
        s = str(t).strip().upper().replace("Z","")
        m = re.match(r"^(\d{2})(\d{2})$", s)
        if m: return int(m.group(1))
        try:
            h = int(s)
            return h if 0 <= h <= 23 else np.nan
        except:
            return np.nan

    # --- Clean dataset ---
    for c in list(df.columns):
        if "Unnamed" in c:
            df.drop(columns=c, inplace=True, errors="ignore")

    df.columns = [c.strip().replace(" ", "_") for c in df.columns]

    if "Lat" in df.columns: df["Lat"] = df["Lat"].apply(parse_lat)
    if "Long" in df.columns: df["Long"] = df["Long"].apply(parse_lon)
    if "RMW_nm" in df.columns: df["RMW_nm"] = df["RMW_nm"].apply(parse_rmw)
    if "Central_Pressure" in df.columns: df["Central_Pressure"] = df["Central_Pressure"].apply(parse_pressure)

    if "Date" in df.columns:
        df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
        df["Year"]  = df["Date"].dt.year
        df["Month"] = df["Date"].dt.month
        df["Day"]   = df["Date"].dt.day

    if "Time" in df.columns:
        df["Hour"] = df["Time"].apply(parse_time_to_hour)

    if "Lat" in df.columns:
        df["Abs_Lat"] = df["Lat"].abs()
    if {"Max_Winds_kt", "Central_Pressure"}.issubset(df.columns):
        df["Wind_to_Pressure"] = df["Max_Winds_kt"] / (df["Central_Pressure"] + 1e-6)

    TARGET = "Size_nm"
    drop_cols = [c for c in ["num","Date","Time","States_Affected","Storm_Name"] if c in df.columns]
    df = df.drop(columns=drop_cols, errors="ignore")

    df = df.dropna(subset=[TARGET]).copy()

    feature_cols = [c for c in df.columns if c != TARGET]
    df[feature_cols] = df[feature_cols].apply(pd.to_numeric, errors="coerce")

    X = df[feature_cols].fillna(df[feature_cols].median(numeric_only=True))
    y = df[TARGET].astype(float)

    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.25, random_state=42)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

    rf = RandomForestRegressor(
        n_estimators=400,
        max_features="sqrt",
        random_state=42,
        n_jobs=-1
    )
    rf.fit(X_train, y_train)

    return rf, feature_cols

rf_model, model_features = load_and_train_model()


# ======================
# 2. Streamlit UI + Map
# ======================
st.title("🌪 Wilmington Hurricane Impact Simulator (with ML Model)")

st.sidebar.header("Hurricane Builder")
st.sidebar.write("Adjust parameters to generate a hurricane and predict its affected area.")

# --- User Inputs ---
inputs = {
    "Max_Winds_kt": st.sidebar.slider("Max Winds (kt)", 50, 200, 120),
    "Central_Pressure": st.sidebar.slider("Central Pressure (mb)", 880, 1020, 950),
    "RMW_nm": st.sidebar.slider("Radius of Max Winds (nm)", 5, 100, 25),
    "OCI_mb": st.sidebar.slider("Outer Core Intensity (mb)", 50, 400, 150),
    "Abs_Lat": abs(34.2),
    "Wind_to_Pressure": st.sidebar.slider("Wind/Pressure Ratio", 0.05, 0.3, 0.15),
}

# Match model feature order (fill missing with zeros)
X_input = pd.DataFrame([inputs])
for f in model_features:
    if f not in X_input.columns:
        X_input[f] = 0.0

predicted_size = rf_model.predict(X_input[model_features])[0]

# ======================
# 3. Map Visualization
# ======================
# Load elevation + address map
gdb_path = "/Users/brianbare/Downloads/Deep_ML/NEWHANOVER.gdb"
gdf = gpd.read_file(gdb_path, layer='NEWHANOVER')

if gdf.crs is None:
    gdf.set_crs(epsg=2264, inplace=True)
gdf_wgs = gdf.to_crs(epsg=4326)
df_map = gdf_wgs[['geometry', 'Full_Address']].copy()
df_map['lon'] = df_map.geometry.x
df_map['lat'] = df_map.geometry.y
df_map['color'] = [ [0, 0, 255] ] * len(df_map)

wilmington_center = (-77.9447, 34.2257)

# Hurricane circle (purple)
hurricane_df = pd.DataFrame({
    'lon': [wilmington_center[0]],
    'lat': [wilmington_center[1]],
    'radius': [predicted_size * 1000],  # nm → meters
    'color': [[128, 0, 128, 80]]
})

address_layer = pdk.Layer(
    "ScatterplotLayer",
    data=df_map,
    get_position='[lon, lat]',
    get_radius=30,
    get_fill_color='color',
    pickable=True,
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

st.pydeck_chart(pdk.Deck(
    layers=[address_layer, hurricane_layer],
    initial_view_state=view_state,
))

# ======================
# 4. Results
# ======================
st.markdown(f"""
### 🌀 Model Prediction
**Predicted Size Affected:** {predicted_size:.2f} nm  
**Visualization:** Purple circle centered on Wilmington  
**Model:** Random Forest Regressor (trained on Hurricane dataset)
""")
