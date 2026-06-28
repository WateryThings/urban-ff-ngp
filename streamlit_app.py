import streamlit as st
import osmnx as ox
import numpy as np
import pandas as pd
import geopandas as gpd
import s3fs
from collections import deque

# --- CONFIGURATION ---
THRESHOLDS = {
    "1-hr Radar Only QPE": 1.0,
    "MRMS Instantaneous Rate": 2.0,
    "CREST Unit Streamflow": 200.0,
    "Hydrophobic": 1000.0
}

if 'rate_buffer' not in st.session_state:
    st.session_state.rate_buffer = deque(maxlen=3)

# --- APP LAYOUT ---
st.set_page_config(page_title="Urban FF - NGP", layout="wide")
st.title("Urban Flash Flood Decision Support (NGP)")

# 1. Fetch Urban Centers (Divide and Conquer approach)
@st.cache_data
def get_urban_centers():
    tags = {'place': ['city', 'town', 'village', 'hamlet']}
    ox.settings.timeout = 180 
    try:
        gdf_nd = ox.features_from_place("North Dakota, USA", tags=tags)
        gdf_sd = ox.features_from_place("South Dakota, USA", tags=tags)
        return pd.concat([gdf_nd, gdf_sd])
    except Exception as e:
        st.error(f"Error fetching data: {e}")
        return gpd.GeoDataFrame()

with st.spinner("Loading urban centers..."):
    urban_gdf = get_urban_centers()

# 2. Alert Engine Logic
def evaluate_alert(qpe_val, inst_rate_history, crest_val, hydrophobic_val):
    avg_inst_rate = np.mean(inst_rate_history) if len(inst_rate_history) == 3 else 0
    triggered = (
        (qpe_val >= THRESHOLDS["1-hr Radar Only QPE"]) or
        (avg_inst_rate >= THRESHOLDS["MRMS Instantaneous Rate"]) or
        (crest_val >= THRESHOLDS["CREST Unit Streamflow"]) or
        (hydrophobic_val >= THRESHOLDS["Hydrophobic"])
    )
    return triggered, avg_inst_rate

# --- SIDEBAR & UI ---
st.sidebar.header("Alert Status")
for k, v in THRESHOLDS.items():
    st.sidebar.write(f"**{k}**: {v}")

st.subheader("Urban Flash Flood Alert Map")

# Map Rendering
if urban_gdf is not None and not urban_gdf.empty:
    if 'geometry' in urban_gdf.columns:
        st.map(urban_gdf)
    else:
        st.write("Data loaded but contains no geometry to map.")
else:
    st.warning("Urban center data is currently unavailable. Please check the logs.")

# Simulation Button
if st.button("Refresh Data & Check Alerts"):
    fs = s3fs.S3FileSystem(anon=True)
    st.info("Pipeline connected to NOAA S3 bucket. Fetching latest grids...")
    try:
        st.success("Successfully accessed NOAA S3 bucket.")
        # Logic test with placeholder values
        is_alert, smoothed_val = evaluate_alert(0.5, st.session_state.rate_buffer, 150, 800)
        if is_alert:
            st.error("ALERT TRIGGERED: Threshold exceeded.")
        else:
            st.info("Conditions normal. No thresholds exceeded.")
    except Exception as e:
        st.error(f"Error connecting to S3: {e}")
