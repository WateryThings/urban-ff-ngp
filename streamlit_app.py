import streamlit as st
import numpy as np
import pandas as pd
import s3fs
import xarray as xr
import datetime

# --- CONFIGURATION ---
# We use the folder names you identified
PRODUCTS = {
    "RadarOnly_QPE_01H_00.00": 1.0,
    "PrecipRate_00.00": 2.0,
    "FLASH_CREST_MAXUNITSTREAMFLOW_00.00": 200.0,
    "FLASH_HP_MAXUNITSTREAMFLOW_00.00": 1000.0
}

# --- APP LAYOUT ---
st.set_page_config(page_title="Urban FF - NGP", layout="wide")
st.title("Urban Flash Flood Decision Support (NGP)")

@st.cache_data
def get_urban_centers():
    return pd.read_csv("urban_centers.csv")

urban_gdf = get_urban_centers()

# --- THE "FAIL-SAFE" SCANNER ---
def get_latest_file_from_s3(fs, product_name):
    now = datetime.datetime.now(datetime.UTC)
    # Check today, then yesterday
    for days_back in [0, 1]:
        date_str = (now - datetime.timedelta(days=days_back)).strftime("%Y%m%d")
        path = f"noaa-mrms-pds/CONUS/{product_name}/{date_str}/"
        try:
            files = fs.ls(path)
            # Find the most recent .grib2.gz file
            grib_files = [f for f in files if f.endswith('.grib2.gz')]
            if grib_files:
                return f"s3://{grib_files[-1]}"
        except:
            continue
    return None

def scan_data():
    fs = s3fs.S3FileSystem(anon=True)
    results = {}
    
    for product, threshold in PRODUCTS.items():
        file_path = get_latest_file_from_s3(fs, product)
        if not file_path:
            continue
            
        try:
            # Use chunks to save memory
            ds = xr.open_dataset(file_path, engine="cfgrib")
            # Automatically grab the first data variable (the radar/model value)
            var_name = list(ds.data_vars)[0]
            
            for _, row in urban_gdf.iterrows():
                val = ds.sel(latitude=row['lat'], longitude=row['lon'], method='nearest')[var_name].values
                if val >= threshold:
                    results[row['name']] = f"{product}: {val:.2f}"
        except Exception as e:
            st.warning(f"Skipping {product} due to error.")
            
    return results

# --- UI ---
st.subheader("Urban Flash Flood Alert Map")
st.map(urban_gdf)

if st.button("Refresh & Scan"):
    with st.spinner("Analyzing live radar data..."):
        alert_results = scan_data()
        if alert_results:
            st.error("🚨 THRESHOLDS EXCEEDED:")
            st.json(alert_results)
        else:
            st.success("✅ All systems normal.")
