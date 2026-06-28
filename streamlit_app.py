import streamlit as st
import numpy as np
import pandas as pd
import s3fs
import xarray as xr
import gzip
import shutil
import os

# --- CONFIGURATION ---
# Using the verified AWS folder names and our converted METRIC thresholds
PRODUCTS = {
    "RadarOnly_QPE_01H_00.00": 25.4,               # 1.0 inch -> 25.4 mm
    "PrecipRate_00.00": 50.8,                      # 2.0 in/hr -> 50.8 mm/hr
    "FLASH_CREST_MAXUNITSTREAMFLOW_00.00": 2.19,   # 200 cfs/sq mi -> 2.19 m³/s/km²
    "FLASH_HP_MAXUNITSTREAMFLOW_00.00": 10.93      # 1000 cfs/sq mi -> 10.93 m³/s/km²
}

# --- APP LAYOUT ---
st.set_page_config(page_title="Urban FF - NGP", layout="wide")
st.title("Urban Flash Flood Decision Support (NGP)")

@st.cache_data
def get_urban_centers():
    # Make sure your CSV has 'lat', 'lon', and 'name' columns!
    return pd.read_csv("urban_centers.csv")

urban_gdf = get_urban_centers()

# --- THE "FAIL-SAFE" SCANNER ---
def get_and_extract_latest_file(fs, product_name):
    """Downloads the newest .grib2.gz file and extracts it for cfgrib to read."""
    # MRMS rolling archive usually drops files right into this root prefix
    path = f"noaa-mrms-pds/CONUS/{product_name}/"
    
    try:
        files = fs.ls(path)
        # Find the most recent .grib2.gz file
        grib_files = [f for f in files if f.endswith('.grib2.gz')]
        
        if not grib_files:
            return None
            
        latest_s3_file = grib_files[-1]
        local_gz = f"temp_{product_name}.grib2.gz"
        local_grib = f"temp_{product_name}.grib2"
        
        # 1. Download the file from S3
        fs.get(latest_s3_file, local_gz)
        
        # 2. Decompress the file (cfgrib cannot read .gz natively)
        with gzip.open(local_gz, 'rb') as f_in:
            with open(local_grib, 'wb') as f_out:
                shutil.copyfileobj(f_in, f_out)
                
        # Clean up the compressed file to save space
        os.remove(local_gz)
        
        return local_grib
    except Exception as e:
        st.warning(f"Failed to fetch {product_name}: {e}")
        return None

def scan_data():
    fs = s3fs.S3FileSystem(anon=True)
    results = {}
    
    for product, threshold in PRODUCTS.items():
        local_file_path = get_and_extract_latest_file(fs, product)
        if not local_file_path:
            continue
            
        try:
            # Now we can safely open the unzipped file
            ds = xr.open_dataset(local_file_path, engine="cfgrib")
            
            # Automatically grab the first data variable (the radar/model value)
            var_name = list(ds.data_vars)[0]
            
            for _, row in urban_gdf.iterrows():
                # Note: If this fails, check if MRMS lons are 0-360 instead of -180 to 180!
                val = ds.sel(latitude=row['lat'], longitude=row['lon'], method='nearest')[var_name].values
                
                if val >= threshold:
                    if row['name'] not in results:
                        results[row['name']] = []
                    results[row['name']].append(f"{product}: {val:.2f} (Threshold: {threshold})")
            
            # Clean up the unzipped file to keep the server clean
            ds.close()
            os.remove(local_file_path)
            
        except Exception as e:
            st.warning(f"Skipping processing for {product} due to error: {e}")
            
    return results

# --- UI ---
st.subheader("Urban Flash Flood Alert Map")

# Streamlit natively handles plotting latitudes and longitudes from a dataframe
st.map(urban_gdf)

if st.button("Refresh & Scan"):
    with st.spinner("Downloading MRMS grids and analyzing thresholds..."):
        alert_results = scan_data()
        
        if alert_results:
            st.error("🚨 THRESHOLDS EXCEEDED:")
            st.json(alert_results)
        else:
            st.success("✅ All systems normal. No locations currently exceed thresholds.")
