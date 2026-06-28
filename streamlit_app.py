import streamlit as st
import numpy as np
import pandas as pd
import s3fs
import xarray as xr
import datetime
from collections import deque

# --- CONFIGURATION: Match your exact folder names ---
PRODUCTS = {
    "RadarOnly_QPE_01H": 1.0,
    "PrecipRate_00.00": 2.0,
    "FLASH_CREST_MAXUNITSTREAMFLOW_00.00": 200.0,
    "FLASH_HP_MAXUNITSTREAMFLOW_00.00": 1000.0
}

# --- APP LAYOUT ---
st.set_page_config(page_title="Urban FF - NGP", layout="wide")
st.title("Urban Flash Flood Decision Support (NGP)")

@st.cache_data
def get_urban_centers():
    try:
        return pd.read_csv("urban_centers.csv")
    except Exception as e:
        st.error(f"Error loading urban_centers.csv: {e}")
        return pd.DataFrame()

urban_gdf = get_urban_centers()

# --- MRMS SCANNER ENGINE ---
def get_latest_mrms_file(fs, product_name):
    now = datetime.datetime.now(datetime.UTC)
    date_str = now.strftime("%Y%m%d")
    # Using your precise folder name structure
    path = f"noaa-mrms-pds/CONUS/{product_name}/{date_str}/"
    try:
        files = fs.ls(path)
        # Returns the most recent file
        return f"s3://{files[-1]}"
    except:
        return None

def scan_data():
    fs = s3fs.S3FileSystem(anon=True)
    results = {}
    
    st.info("Scanning NOAA S3 products...")
    
    for product, threshold in PRODUCTS.items():
        file_path = get_latest_mrms_file(fs, product)
        if file_path:
            try:
                ds = xr.open_dataset(file_path, engine="cfgrib")
                
                # IMPORTANT: This finds the actual data variable name
                # (GRIB2 files often use names like 'tp' or 'prate')
                data_var = list(ds.data_vars)[0]
                
                for _, row in urban_gdf.iterrows():
                    val = ds.sel(latitude=row['lat'], longitude=row['lon'], method='nearest')[data_var].values
                    if val >= threshold:
                        results[row['name']] = f"{product}: {val:.2f} (Threshold: {threshold})"
            except Exception as e:
                st.warning(f"Could not scan {product}: {e}")
        else:
            st.warning(f"No files found for {product}")
            
    return results

# --- UI & INTERACTION ---
st.subheader("Urban Flash Flood Alert Map")
if not urban_gdf.empty:
    st.map(urban_gdf)

if st.button("Refresh Data & Scan Alerts"):
    alert_results = scan_data()
    if alert_results:
        st.error("🚨 THRESHOLDS EXCEEDED:")
        st.json(alert_results)
    else:
        st.success("✅ All systems normal. No thresholds exceeded.")
