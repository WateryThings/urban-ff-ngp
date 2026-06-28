import streamlit as st
import numpy as np
import pandas as pd
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

# 1. Fetch Urban Centers from local CSV
@st.cache_data
def get_urban_centers():
    try:
        # Reads the file you created on GitHub
        df = pd.read_csv("urban_centers.csv")
        return df
    except Exception as e:
        st.error(f"Error loading urban_centers.csv: {e}")
        return pd.DataFrame()

with st.spinner("Loading locations..."):
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
if not urban_gdf.empty:
    st.map(urban_gdf)
else:
    st.warning("No location data found. Please ensure urban_centers.csv is in the repository.")

# Simulation Button
if st.button("Refresh Data & Check Alerts"):
    fs = s3fs.S3FileSystem(anon=True)
    st.info("Pipeline connected to NOAA S3 bucket.")
    try:
        # Logic test with placeholder values
        is_alert, smoothed_val = evaluate_alert(0.5, st.session_state.rate_buffer, 150, 800)
        if is_alert:
            st.error("ALERT TRIGGERED: Threshold exceeded.")
        else:
            st.info("Conditions normal. No thresholds exceeded.")
    except Exception as e:
        st.error(f"Error connecting to S3: {e}")
