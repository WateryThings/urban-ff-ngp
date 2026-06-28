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
    # Load our massive new dataset matching your 5 WFO CWA boundaries perfectly
    return pd.read_csv("urban_centers.csv")

urban_gdf = get_urban_centers()

# --- THE "FAIL-SAFE" SCANNER ---
def get_and_extract_latest_file(fs, product_name):
    """Downloads the newest .grib2.gz file and extracts it for cfgrib to read."""
    path = f"noaa-mrms-pds/CONUS/{product_name}/"
    
    try:
        files = fs.ls(path)
        grib_files = [f for f in files if f.endswith('.grib2.gz')]
        
        if not grib_files:
            return None
            
        latest_s3_file = grib_files[-1]
        local_gz = f"temp_{product_name}.grib2.gz"
        local_grib = f"temp_{product_name}.grib2"
        
        # 1. Download the file from S3 anonymously
        fs.get(latest_s3_file, local_gz)
        
        # 2. Decompress the file
        with gzip.open(local_gz, 'rb') as f_in:
            with open(local_grib, 'wb') as f_out:
                shutil.copyfileobj(f_in, f_out)
                
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
            ds = xr.open_dataset(local_file_path, engine="cfgrib")
            var_name = list(ds.data_vars)[0]
            
            for _, row in urban_gdf.iterrows():
                # Define our 5-mile search box around the town
                buffer = 0.07 
                lat = row['lat']
                lon = row['lon'] % 360  # Handling the 0-360 MRMS coordinate framework
                
                # Slice the MRMS grid to our 5-mile box and find the MAX value inside it
                try:
                    val = ds.sel(
                        latitude=slice(lat + buffer, lat - buffer),
                        longitude=slice(lon - buffer, lon + buffer)
                    )[var_name].max().values
                    
                    # Check against the metric threshold settings
                    if pd.notna(val) and val >= threshold:
                        unique_key = f"{row['name']}, {row['state']}"
                        if unique_key not in results:
                            results[unique_key] = []
                        results[unique_key].append(f"{product}: {val:.2f} (Threshold: {threshold})")
                except KeyError:
                    continue
            
            ds.close()
            os.remove(local_file_path)
            
        except Exception as e:
            st.warning(f"Skipping processing for {product} due to error: {e}")
            
    return results

# --- UI ---
st.subheader("Regional CWA Flash Flood Alert Map")

# 1. Set clean default styles for the map configuration
urban_gdf['color'] = '#A9A9A9'  # Cool Gray background dots for safe towns
urban_gdf['size'] = 20          # Small baseline indicator footprint

# 2. Create an in-place placeholder for smooth interface transitions
map_placeholder = st.empty()
map_placeholder.map(urban_gdf, color='color', size='size')

if st.button("Refresh & Scan"):
    with st.spinner("Downloading MRMS grids and analyzing regional CWA footprints..."):
        alert_results = scan_data()
        
        if alert_results:
            st.error("🚨 THRESHOLDS EXCEEDED WITHIN OPERATIONAL REGIONS:")
            
            # Match alert results cleanly back to our custom dataframe
            # We look at the unique string key "Town, State" to safely handle duplicate town names
            alerted_towns = [key.split(",")[0].strip() for key in alert_results.keys()]
            alerted_states = [key.split(",")[1].strip() for key in alert_results.keys()]
            
            # Adjust the dynamic mapping visual states for alerted domains
            # This turns only the threatened towns bright red and makes them huge
            for name, state in zip(alerted_towns, alerted_states):
                urban_gdf.loc[(urban_gdf['name'] == name) & (urban_gdf['state'] == state), 'color'] = '#FF0000'
                urban_gdf.loc[(urban_gdf['name'] == name) & (urban_gdf['state'] == state), 'size'] = 1000
            
            # Redraw map seamlessly
            map_placeholder.map(urban_gdf, color='color', size='size')
            st.json(alert_results)
            
        else:
            st.success("✅ All systems normal across all 5 operational WFO domains.")
            map_placeholder.map(urban_gdf, color='color', size='size')
