import streamlit as st
import numpy as np
import pandas as pd
import s3fs
import xarray as xr
import gzip
import shutil
import os
import json
import pydeck as pdk
from streamlit_autorefresh import st_autorefresh

# --- CONFIGURATION ---
PRODUCTS = {
    "RadarOnly_QPE_01H_00.00": 25.4,               # 1.0 inch -> 25.4 mm
    "PrecipRate_00.00": 50.8,                      # 2.0 in/hr -> 50.8 mm/hr
    "FLASH_CREST_MAXUNITSTREAMFLOW_00.00": 2.19,   # 200 cfs/sq mi -> 2.19 m³/s/km²
    "FLASH_HP_MAXUNITSTREAMFLOW_00.00": 10.93      # 1000 cfs/sq mi -> 10.93 m³/s/km²
}

# --- APP LAYOUT ---
st.set_page_config(page_title="Urban FF - NGP", layout="wide")
st.title("Urban Flash Flood Decision Support (NGP)")

# --- AUTOMATED OPERATIONS TIMER ---
# Automatically refreshes the entire web application every 120,000 milliseconds (2 minutes)
# This keeps data completely synced with rolling MRMS bucket arrivals without user interaction
count = st_autorefresh(interval=120000, limit=None, key="mrms_auto_scanner")

# --- BLUF & OPERATIONAL USER GUIDE ---
st.markdown("""
### Operational Use:
**BLUF:** When any of the below criteria are exceeded within 5-miles of an urban area, the map will light up red.
""")

col1, col2 = st.columns(2)

with col1:
    st.markdown("""
    #### Monitored Products & Thresholds for Urban Areas:
    * **1-hr MRMS QPE:** Greater or equal than 1.0"
    * **MRMS Instantaneous Rain Rates:** Greater than or equal to 2.0"/1hr
    * **FLASH CREST Max Unit Streamflow:** Greater than or equal to 200 cfs/sq. mi.
    * **FLASH Hydrophobic Max Unit Streamflow:** Greater than or equal to 1000 cfs/sq. mi.
    """)

with col2:
    st.markdown("""
    #### Map Symbology & System States
    * **Automated Refresh:** The system automatically executes a full background scan every **2 minutes** to sync with live data cycles.
    * **Manual Interrogation:** Click **"Refresh & Scan"** to manually force an immediate data query of the NOAA MRMS AWS bucket.
    * **Translucent Gray Polygons:** Maximum values within the 5-mile community edge buffer are below active thresholds.
    * **Solid Red Polygons:** One or more MRMS products meet or exceed the listed thresholds within the buffered area. Detailed readings will output via JSON text below the map.
    """)

st.markdown(f"*System Status: Running automated tracking cycle. (Auto-Scan ID: {count})*")
st.markdown("---")

@st.cache_data
def get_urban_centers():
    df = pd.read_csv("urban_centers.csv")
    df['min_lon'] = pd.to_numeric(df['min_lon'], errors='coerce')
    df['max_lon'] = pd.to_numeric(df['max_lon'], errors='coerce')
    df['min_lat'] = pd.to_numeric(df['min_lat'], errors='coerce')
    df['max_lat'] = pd.to_numeric(df['max_lat'], errors='coerce')
    return df.dropna(subset=['min_lon', 'max_lon', 'min_lat', 'max_lat'])

@st.cache_data
def load_json_layer(filepath):
    with open(filepath, "r") as f:
        return json.load(f)

urban_gdf = get_urban_centers()
cwa_geojson = load_json_layer("cwa_outlines.json")
urban_shapes_geojson = load_json_layer("urban_boundaries.json")

# Initialize default quiet styling for all urban polygon shapes
for feature in urban_shapes_geojson["features"]:
    feature["properties"]["fill_color"] = [180, 180, 180, 50]   
    feature["properties"]["line_color"] = [120, 120, 120, 100]  

# --- THE "FAIL-SAFE" SCANNER ---
def get_and_extract_latest_file(fs, product_name):
    path = f"noaa-mrms-pds/CONUS/{product_name}/"
    try:
        files = fs.ls(path)
        grib_files = [f for f in files if f.endswith('.grib2.gz')]
        if not grib_files: return None
        
        latest_s3_file = grib_files[-1]
        local_gz = f"temp_{product_name}.grib2.gz"
        local_grib = f"temp_{product_name}.grib2"
        
        fs.get(latest_s3_file, local_gz)
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
        if not local_file_path: continue
        try:
            ds = xr.open_dataset(local_file_path, engine="cfgrib")
            var_name = list(ds.data_vars)[0]
            
            for _, row in urban_gdf.iterrows():
                min_lon = row['min_lon'] % 360
                max_lon = row['max_lon'] % 360
                min_lat = row['min_lat']
                max_lat = row['max_lat']
                
                try:
                    val = ds.sel(
                        latitude=slice(max_lat, min_lat),  
                        longitude=slice(min_lon, max_lon)
                    )[var_name].max().values
                    
                    if pd.notna(val) and val >= threshold:
                        unique_key = f"{row['name']}, {row['state']}"
                        if unique_key not in results: results[unique_key] = []
                        results[unique_key].append(f"{product}: {val:.2f} (Threshold: {threshold})")
                except KeyError:
                    continue
            ds.close()
            os.remove(local_file_path)
        except Exception as e:
            st.warning(f"Skipping processing for {product} due to error: {e}")
    return results

# --- RENDERING THE ADVANCED MAP INTERFACE ---
st.subheader("Regional CWA Flash Flood Alert Map")

def render_map(cwa_layer, city_shapes):
    outline_layer = pdk.Layer(
        "GeoJsonLayer",
        cwa_layer,
        stroke_width=3,
        get_line_color=[0, 150, 255, 255], 
        get_fill_color=[0, 0, 0, 0],       
        line_width_min_pixels=2,
    )
    
    urban_polygon_layer = pdk.Layer(
        "GeoJsonLayer",
        city_shapes,
        get_line_color="properties.line_color",
        get_fill_color="properties.fill_color",
        pickable=True,
        extruded=False,
    )
    
    view_state = pdk.ViewState(latitude=45.5, longitude=-100.0, zoom=5.5, pitch=0)
    
    return pdk.Deck(
        layers=[outline_layer, urban_polygon_layer],
        initial_view_state=view_state,
        map_style="light",  
        tooltip={"text": "{name}"}
    )

map_placeholder = st.empty()
map_placeholder.pydeck_chart(render_map(cwa_geojson, urban_shapes_geojson))

# Automatically run the scanner on initial page load or when the autorefresh hits
with st.spinner("Analyzing current regional CWA footprints..."):
    alert_results = scan_data()
    
    # Reset colors back to clean baseline gray first
    for feature in urban_shapes_geojson["features"]:
        feature["properties"]["fill_color"] = [180, 180, 180, 50]
        feature["properties"]["line_color"] = [120, 120, 120, 100]
        
    if alert_results:
        st.error("🚨 THRESHOLDS EXCEEDED WITHIN OPERATIONAL REGIONS:")
        alerted_towns = [key.split(",")[0].strip().upper() for key in alert_results.keys()]
        
        for feature in urban_shapes_geojson["features"]:
            feat_name = str(feature["properties"]["name"]).upper()
            if any(town in feat_name for town in alerted_towns):
                feature["properties"]["fill_color"] = [255, 0, 0, 180]  
                feature["properties"]["line_color"] = [150, 0, 0, 255]  
        
        map_placeholder.pydeck_chart(render_map(cwa_geojson, urban_shapes_geojson))
        st.json(alert_results)
    else:
        st.success("✅ All systems normal across all 5 operational WFO domains.")
        map_placeholder.pydeck_chart(render_map(cwa_geojson, urban_shapes_geojson))

# Force manual verification override button
if st.button("Refresh & Scan"):
    st.rerun()
