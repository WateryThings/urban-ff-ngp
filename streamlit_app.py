import streamlit as st
import osmnx as ox
import numpy as np
import pandas as pd
import geopandas as gpd
from collections import deque

# --- CONFIGURATION ---
# Thresholds for our "OR" alert logic
THRESHOLDS = {
    "1-hr Radar Only QPE": 1.0,
    "MRMS Instantaneous Rate": 2.0,
    "CREST Unit Streamflow": 200.0,
    "Hydrophobic" Unit Streamflow: 1000.0
}

# Persistent buffer for 6-minute smoothing (3 scans * 2 mins = 6 mins)
if 'rate_buffer' not in st.session_state:
    st.session_state.rate_buffer = deque(maxlen=3)

# --- APP LAYOUT ---
st.set_page_config(page_title="Urban FF - NGP", layout="wide")
st.title("Urban Flash Flood Decision Support (NGP)")

# 1. Fetch Urban Centers (Cached)
@st.cache_data
def get_urban_centers():
    tags = {'place': ['city', 'town', 'village', 'hamlet']}
    return ox.features_from_place(["North Dakota, USA", "South Dakota, USA"], tags=tags)

with st.spinner("Loading urban centers..."):
    urban_gdf = get_urban_centers()

# 2. Alert Logic Function
def evaluate_alert(qpe_val, inst_rate_history, crest_val, hydrophobic_val):
    # Calculate smoothed rate (mean of last 3 scans)
    avg_inst_rate = np.mean(inst_rate_history) if len(inst_rate_history) == 3 else 0
    
    # The OR logic gate
    triggered = (
        (qpe_val >= THRESHOLDS["1-hr Radar Only QPE"]) or
        (avg_inst_rate >= THRESHOLDS["MRMS Instantaneous Rate"]) or
        (crest_val >= THRESHOLDS["CREST Unit Streamflow"]) or
        (hydrophobic_val >= THRESHOLDS["Hydrophobic"])
    )
    return triggered, avg_inst_rate

# --- SIDEBAR & UI ---
st.sidebar.header("Alert Status")
st.sidebar.write("Active Thresholds:")
for k, v in THRESHOLDS.items():
    st.sidebar.text(f"{k}: {v}")

# Display Map
st.subheader("Urban Flash Flood Alert Map")
st.map(urban_gdf)

# Simulation Button (Placeholder for AWS Data Fetching)
if st.button("Refresh Data & Check Alerts"):
    st.success("Data pipeline pinged successfully. Buffer status: " + 
               f"{len(st.session_state.rate_buffer)}/3 scans captured.")
    
    # Logic test example using the updated 'hydrophobic_val'
    is_alert, smoothed_val = evaluate_alert(0.5, st.session_state.rate_buffer, 150, 800)
    if is_alert:
        st.error("ALERT TRIGGERED: Threshold exceeded in one or more variables.")
    else:
        st.info("Conditions normal. No thresholds exceeded.")
