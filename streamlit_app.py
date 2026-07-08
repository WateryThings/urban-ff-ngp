import streamlit as st
import numpy as np
import pandas as pd
import s3fs
import xarray as xr
import gzip
import shutil
import os
import json
import urllib.request
import pydeck as pdk
import time
import copy
from streamlit_autorefresh import st_autorefresh
from datetime import datetime, timezone, timedelta

# --- CONFIGURATION & METRIC THRESHOLDS ---
PRODUCTS = {
    "RadarOnly_QPE_01H_00.00": 25.4,               # 1.0 inch -> 25.4 mm
    "FLASH_CREST_MAXUNITSTREAMFLOW_00.00": 2.19,   # 200 cfs/sq mi -> 2.19 m³/s/km²
    "FLASH_HP_MAXUNITSTREAMFLOW_00.00": 10.93      # 1000 cfs/sq mi -> 10.93 m³/s/km²
}

# --- APP LAYOUT & BREAKOUT SPACING FIX ---
st.set_page_config(page_title="Urban FF - NGP", layout="wide")

st.html("""
    <style>
        .block-container {
            padding-top: 5.5rem !important;
            padding-bottom: 1rem !important;
            max-width: 98% !important;
        }
        h1, h2, h3, h4 {
            margin-top: 0.2rem !important;
            margin-bottom: 0.2rem !important;
            padding-top: 0px !important;
        }
        .stElementContainer {
            margin-bottom: 0.4rem !important;
        }
        .custom-caution-banner {
            background-color: #FFE600 !important;
            color: #000000 !important;
            padding: 12px 20px !important;
            border-radius: 6px !important;
            font-weight: bold !important;
            font-size: 14px !important;
            margin-bottom: 1rem !important;
            border: 1px solid #E6D000 !important;
            box-shadow: 0px 2px 4px rgba(0,0,0,0.05) !important;
            display: block !important;
        }
    </style>
    
    <div class="custom-caution-banner">
         CAUTION: EVEN HAN SOLO DOESN'T THINK THIS THING WILL HEAR HIM OR HOLD TOGETHER. 
    </div>
""")

# --- AUTOMATED OPERATIONS TIMER (UPDATED TO 5-MINUTES FOR OPTIMAL SPEED) ---
count = st_autorefresh(interval=300000, limit=None, key="mrms_auto_scanner")

# --- HEADER & TIMESTAMP GRID ---
header_col, time_col = st.columns([3, 1])

with header_col:
    st.title("NGP Urban and Small Towns: Flash Flood Decision Support")

with time_col:
    utc_now = datetime.now(timezone.utc)
    cdt_now = utc_now - timedelta(hours=5)
    local_time_str = cdt_now.strftime("%I:%M %p cdt").lower()
    utc_time_str = utc_now.strftime("%H:%M UTC")
    st.info(f"⏳ **Last Scanner Update:** {local_time_str} ({utc_time_str}) | *Cycle: {count}*")

st.markdown("---")

# --- BLUF & OPERATIONAL USER GUIDE ---
st.markdown("""
**BLUF:** This real-time tool will flash red for any city or small town that is at risk for flash flooding when **at least 2 out of 3** product thresholds are met strictly within the city limits.
""")

col1, col2, col3 = st.columns([2, 2, 1])

with col1:
    st.markdown("""
    #### Monitored Products & Thresholds (2/3 must be met):
    * MRMS 1-hr QPE: $\ge$ 1.0"
    * FLASH CREST Max Unit Streamflow: $\ge$ 200 cfs/sq. mi.
    * FLASH Hydrophobic Max Unit Streamflow: $\ge$ 1000 cfs/sq. mi.
    """)

with col2:
    st.markdown("""
    #### Map Symbology:
    * **Dark Gray Polygons:** Spatial boundary extent of all 1,146 monitored urban areas and small towns.
    * **Solid Red Polygons:** At least 2 out of 3 MRMS products exceed the thresholds anywhere strictly within the city boundaries.
    * **Alert Timing:** Alerts update live. To account for urban runoff and drainage lag, alerts will remain active 30 minutes after product thresholds have dropped below the required criteria.
    * **Automated Refresh:** Updates every 5-minutes.
    """, unsafe_allow_html=True)

with col3:
    st.markdown("#### Map Layers:")
    toggle_radar = st.checkbox("Overlay Base Reflectivity", value=False, help="Toggles live IEM NEXRAD Base Reflectivity mosaic over the map area.")
    radar_opacity = st.slider(
        "Radar Opacity", 
        min_value=0.0, max_value=1.0, value=0.55, step=0.05,
        help="Adjust the transparency of the Base Reflectivity overlay layer."
    )
    toggle_warnings = st.checkbox("Overlay FAYs and
