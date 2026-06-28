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
    # Load our massive new Census dataset (make sure the columns are name, state, lat, lon!)
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
                # Define our 5-mile search box around the town
                buffer = 0.07 
                lat = row['lat']
                lon = row['lon'] % 360  # Catching that 0-360 longitude trap!
                
                # Slice the MRMS grid to our 5-mile box and find the MAX value inside it
                # Note: MRMS latitudes are stored North to South, so we slice max to min
                try:
                    val = ds.sel(
                        latitude=slice(lat + buffer, lat - buffer),
                        longitude=slice(lon - buffer, lon + buffer)
                    )[var_name].max().values
                    
                    # Ensure the value isn't empty data (NaN), and check against threshold
                    if pd.notna(val) and val >= threshold:
                        if row['name'] not in results:
                            results[row['name']] = []
                        results[row['name']].append(f"{product}: {val:.2f} (Threshold: {threshold})")
                except KeyError:
                    # If the town falls completely off the MRMS grid edge, gracefully skip it
                    continue
            
            # Clean up the unzipped file to keep the server clean
            ds.close()
            os.remove(local_file_path)
            
        except Exception as e:
            st.warning(f"Skipping processing for {product} due to error: {e}")
            
    return results

# --- UI ---
st.subheader("Urban Flash Flood Alert Map")

# 1. Set default styles for the map: Gray and Small
urban_gdf['color'] = '#A9A9A9'  # Hex code for Gray
urban_gdf['size'] = 20          # Small dot

# 2. Create a placeholder so we can update the map in-place later
map_placeholder = st.empty()
map_placeholder.map(urban_gdf, color='color', size='size')

if st.button("Refresh & Scan"):
    with st.spinner("Downloading MRMS grids and analyzing spatial thresholds..."):
        alert_results = scan_data()
        
        if alert_results:
            st.error("🚨 THRESHOLDS EXCEEDED:")
            
            # Find exactly which towns triggered the alert
            alerted_towns = list(alert_results.keys())
            
            # 3. Light them up! Change alerted towns to Bright Red and make them HUGE
            urban_gdf.loc[urban_gdf['name'].isin(alerted_towns), 'color'] = '#FF0000'
            urban_gdf.loc[urban_gdf['name'].isin(alerted_towns), 'size'] = 1000
            
            # 4. Redraw the map with the new danger zones highlighted
            map_placeholder.map(urban_gdf, color='color', size='size')
            
            # Print the detailed JSON readouts below the map
            st.json(alert_results)
