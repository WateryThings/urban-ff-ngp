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
from datetime import datetime, timezone

# --- CONFIGURATION ---
PRODUCTS = {
    "RadarOnly_QPE_01H_00.00": 25.4,               # 1.0 inch -> 25.4 mm
    "FLASH_CREST_MAXUNITSTREAMFLOW_00.00": 2.19,   # 200 cfs/sq mi -> 2.19 m³/s/km²
    "FLASH_HP_MAXUNITSTREAMFLOW_00.00": 10.93      # 1000 cfs/sq mi -> 10.93 m³/s/km²
}
RAIN_RATE_PROD = "PrecipRate_00.00"
RAIN_RATE_THRESH = 50.8                           # 2.0 in/hr -> 50.8 mm/hr

# --- APP LAYOUT ---
st.set_page_config(page_title="Urban FF - NGP", layout="wide")
st.title("NGP Urban and Small Towns: Flash Flood Decision Support")

# --- AUTOMATED OPERATIONS TIMER ---
count = st_autorefresh(interval=120000, limit=None, key="mrms_auto_scanner")

# --- BLUF & OPERATIONAL USER GUIDE ---
st.markdown("""
**BLUF:** This real-time tool will flash red for any city or small town that is at risk for flash flooding based on any of the below products & criteria being met within a 5-mile buffer.
""")

col1, col2 = st.columns(2)

with col1:
    st.markdown("""
    #### Monitored Products & Thresholds:
    * **MRMS 1-hr QPE:** $\ge$ 1.0"
    * **MRMS Instantaneous Rain Rates:** $\ge$ 2.0"/1-hr *(sustained over at least 3 scans)*
    * **FLASH CREST Max Unit Streamflow:** $\ge$ 200 cfs/sq. mi.
    * **FLASH Hydrophobic Max Unit Streamflow:** $\ge$ 1000 cfs/sq. mi.
    """)

with col2:
    st.markdown("""
    #### Map Symbology:
    * **Translucent Gray Polygons:** Spatial extent of urban and small towns.
    * **Solid Red Polygons:** One or more of the MRMS products exceed the listed thresholds within the buffer area. Details about this area will be displayed below the map.
    * **Automated Refresh:** Updates every 2-minutes to sync with live MRMS data feed.
    """)

# --- TIMESTAMP READOUT ---
local_time_str = datetime.now().strftime("%I:%M %p CDT").lower()
utc_time_str = datetime.now(timezone.utc).strftime("%H:%M UTC")

st.info(f"⏳ **Last Scanner Update:** {local_time_str} ({utc_time_str}) | *Auto-Scan ID Cycle: {count}*")
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

# --- DATA EXTRACTION WITH MEMORY FOR SCANNING ---
def get_latest_files(fs, product_name, num_files=1):
    path = f"noaa-mrms-pds/CONUS/{product_name}/"
    try:
        files = fs.ls(path)
        grib_files = [f for f in files if f.endswith('.grib2.gz')]
        if not grib_files: return []
        return grib_files[-num_files:]
    except Exception as e:
        st.warning(f"Failed to list bucket for {product_name}: {e}")
        return []

def extract_file(fs, s3_path, idx_suffix=""):
    local_gz = f"temp_{idx_suffix}.grib2.gz"
    local_grib = f"temp_{idx_suffix}.grib2"
    try:
        fs.get(s3_path, local_gz)
        with gzip.open(local_gz, 'rb') as f_in:
            with open(local_grib, 'wb') as f_out:
                shutil.copyfileobj(f_in, f_out)
        os.remove(local_gz)
        return local_grib
    except Exception as e:
        if os.path.exists(local_gz): os.remove(local_gz)
        return None

def scan_data():
    fs = s3fs.S3FileSystem(anon=True)
    results = {}
    
    # Part 1: Standard Products Check (Single Latest Scan Evaluation)
    for product, threshold in PRODUCTS.items():
        latest_files = get_latest_files(fs, product, num_files=1)
        if not latest_files: continue
        local_grib = extract_file(fs, latest_files[0], product)
        if not local_grib: continue
        try:
            ds = xr.open_dataset(local_grib, engine="cfgrib")
            var_name = list(ds.data_vars)[0]
            for _, row in urban_gdf.iterrows():
                min_lon, max_lon = row['min_lon'] % 360, row['max_lon'] % 360
                val = ds.sel(latitude=slice(row['max_lat'], row['min_lat']), longitude=slice(min_lon, max_lon))[var_name].max().values
                if pd.notna(val) and val >= threshold:
                    key = f"{row['name']}, {row['state']}"
                    if key not in results: results[key] = []
                    results[key].append(f"{product}: {val:.2f} (Threshold: {threshold})")
            ds.close()
            os.remove(local_grib)
        except Exception:
            if os.path.exists(local_grib): os.remove(local_grib)

    # Part 2: Rain Rate Check (3 Consecutive Scans Continuity Filter)
    rate_history_files = get_latest_files(fs, RAIN_RATE_PROD, num_files=3)
    if len(rate_history_files) == 3:
        local_gribs = [extract_file(fs, f, f"rate_{i}") for i, f in enumerate(rate_history_files)]
        try:
            datasets = [xr.open_dataset(g, engine="cfgrib") for g in local_gribs if g]
            if len(datasets) == 3:
                var_names = [list(d.data_vars)[0] for d in datasets]
                for _, row in urban_gdf.iterrows():
                    min_lon, max_lon = row['min_lon'] % 360, row['max_lon'] % 360
                    
                    # Pull values across all three time-steps
                    v1 = datasets[0].sel(latitude=slice(row['max_lat'], row['min_lat']), longitude=slice(min_lon, max_lon))[var_names[0]].max().values
                    v2 = datasets[1].sel(latitude=slice(row['max_lat'], row['min_lat']), longitude=slice(min_lon, max_lon))[var_names[1]].max().values
                    v3 = datasets[2].sel(latitude=slice(row['max_lat'], row['min_lat']), longitude=slice(min_lon, max_lon))[var_names[2]].max().values
                    
                    # Evaluate if the rate is sustained over all three historical parameters
                    if pd.notna(v1) and pd.notna(v2) and pd.notna(v3):
                        if v1 >= RAIN_RATE_THRESH and v2 >= RAIN_RATE_THRESH and v3 >= RAIN_RATE_THRESH:
                            key = f"{row['name']}, {row['state']}"
                            if key not in results: results[key] = []
                            results[key].append(f"Sustained Rain Rate (3 Scans): Min Peak {min(v1,v2,v3):.2f} (Threshold: {RAIN_RATE_THRESH})")
            for d in datasets: d.close()
        except Exception:
            pass
        for g in local_gribs:
            if g and os.path.exists(g): os.remove(g)
            
    return results

# --- RENDERING THE ADVANCED MAP INTERFACE ---
st.subheader("Regional CWA Flash Flood Alert Map")

def render_map(cwa_layer, city_shapes):
    outline_layer = pdk.Layer(
        "GeoJsonLayer", cwa_layer, stroke_width=3,
        get_line_color=[0, 150, 255, 255], get_fill_color=[0, 0, 0, 0], line_width_min_pixels=2,
    )
    urban_polygon_layer = pdk.Layer(
        "GeoJsonLayer", city_shapes,
        get_line_color="properties.line_color", get_fill_color="properties.fill_color",
        pickable=True, extruded=False,
    )
    return pdk.Deck(
        layers=[outline_layer, urban_polygon_layer],
        initial_view_state=pdk.ViewState(latitude=45.5, longitude=-100.0, zoom=5.5, pitch=0),
        map_style="light", tooltip={"text": "{name}"}
    )

map_placeholder = st.empty()
map_placeholder.pydeck_chart(render_map(cwa_geojson, urban_shapes_geojson))

with st.spinner("Analyzing current regional CWA footprints..."):
    alert_results = scan_data()
    
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

if st.button("Refresh & Scan"):
    st.rerun()
