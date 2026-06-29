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
from streamlit_autorefresh import st_autorefresh
from datetime import datetime, timezone, timedelta

# --- CONFIGURATION & METRIC THRESHOLDS ---
PRODUCTS = {
    "RadarOnly_QPE_01H_00.00": 25.4,               
    "FLASH_CREST_MAXUNITSTREAMFLOW_00.00": 2.19,   
    "FLASH_HP_MAXUNITSTREAMFLOW_00.00": 10.93      
}
RAIN_RATE_PROD = "PrecipRate_00.00"
RAIN_RATE_THRESH = 50.8                           

st.set_page_config(page_title="Urban FF - NGP", layout="wide")

st.html("""
    <style>
        .block-container { padding-top: 1rem !important; padding-bottom: 1rem !important; max-width: 98% !important; }
        h1, h2, h3, h4 { margin-top: 0.2rem !important; margin-bottom: 0.2rem !important; padding-top: 0px !important; }
        .stElementContainer { margin-bottom: 0.4rem !important; }
    </style>
""")

st.warning("⚠️ **CAUTION:** This tool is an experimental prototype (similar to C3P0 in The Phantom Menace) and will GUARANTEE, NO QUESTIONS ASKED FAIL, EVEN NOW, THIS SECOND!")

count = st_autorefresh(interval=120000, limit=None, key="mrms_auto_scanner")

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
    with open(filepath, "r") as f: return json.load(f)

urban_gdf = get_urban_centers()
cwa_geojson = load_json_layer("cwa_outlines.json")
urban_shapes_geojson = load_json_layer("urban_boundaries.json")

# --- CACHED FILE LIST LAYER ---
@st.cache_data(ttl=60, show_spinner=False)
def get_latest_files(product_name, num_files=1):
    fs = s3fs.S3FileSystem(anon=True)
    now_utc = datetime.now(timezone.utc)
    today_str = now_utc.strftime("%Y%m%d")
    yesterday_str = (now_utc - timedelta(days=1)).strftime("%Y%m%d")
    
    path_today = f"noaa-mrms-pds/CONUS/{product_name}/{today_str}/"
    try:
        files = fs.ls(path_today)
        grib_files = [f for f in files if f.endswith('.grib2.gz')]
        if grib_files: return sorted(grib_files)[-num_files:]
    except Exception: pass
        
    path_yesterday = f"noaa-mrms-pds/CONUS/{product_name}/{yesterday_str}/"
    try:
        files = fs.ls(path_yesterday)
        grib_files = [f for f in files if f.endswith('.grib2.gz')]
        if grib_files: return sorted(grib_files)[-num_files:]
    except Exception: pass
    return []

def extract_file(s3_path, idx_suffix=""):
    fs = s3fs.S3FileSystem(anon=True)
    local_gz = f"temp_{idx_suffix}.grib2.gz"
    local_grib = f"temp_{idx_suffix}.grib2"
    try:
        fs.get(s3_path, local_gz)
        with gzip.open(local_gz, 'rb') as f_in:
            with open(local_grib, 'wb') as f_out:
                shutil.copyfileobj(f_in, f_out)
        os.remove(local_gz)
        return local_grib
    except Exception:
        if os.path.exists(local_gz): os.remove(local_gz)
        return None

# --- CONSENSUS CROSS-DATASET EVALUATION ENGINE ---
@st.cache_data(show_spinner=False)
def scan_data(cycle_count):
    results = {}
    logs = [] 
    
    print(f"\n--- ⏱️ STARTING SCAN CYCLE {cycle_count} ---")
    town_tallies = {f"{row['name']}, {row['state']}": {"score": 0, "details": []} for _, row in urban_gdf.iterrows()}
    
    for product, threshold in PRODUCTS.items():
        t0 = time.time()
        latest_files = get_latest_files(product, num_files=1)
        if not latest_files: 
            logs.append(f"❌ No files found for {product}")
            continue
            
        print(f"[{product}] Found files in {time.time()-t0:.2f}s. Extracting...")
        t1 = time.time()
        local_grib = extract_file(latest_files[0], product)
        if not local_grib: 
            logs.append(f"❌ Extraction failed for {product}")
            continue
            
        print(f"[{product}] Extracted in {time.time()-t1:.2f}s. Opening dataset...")
        try:
            t2 = time.time()
            ds = xr.open_dataset(local_grib, engine="cfgrib")
            var_name = list(ds.data_vars)[0]
            
            lat_ascending = bool(ds.latitude[0] < ds.latitude[-1])
            master_lat_slice = slice(41.5, 50.0) if lat_ascending else slice(50.0, 41.5)
            master_lon_slice = slice(360 - 107.0, 360 - 93.5)
            
            # Crop and load into RAM
            ds_cropped = ds.sel(latitude=master_lat_slice, longitude=master_lon_slice).load()
            print(f"[{product}] Cropped and loaded into RAM in {time.time()-t2:.2f}s. Slicing towns...")
            
            t3 = time.time()
            for _, row in urban_gdf.iterrows():
                key = f"{row['name']}, {row['state']}"
                
                # FIX: Absolute safety check on longitude conversions
                lon_raw = row['min_lon'] if row['min_lon'] < 0 else -row['min_lon']
                min_lon = lon_raw % 360
                max_lon = (row['max_lon'] if row['max_lon'] < 0 else -row['max_lon']) % 360
                
                lats = [row['min_lat'], row['max_lat']]
                lat_slice = slice(min(lats), max(lats)) if lat_ascending else slice(max(lats), min(lats))
                
                val = ds_cropped.sel(latitude=lat_slice, longitude=slice(min(min_lon, max_lon), max(min_lon, max_lon)))[var_name].max().values
                
                if pd.notna(val) and val >= threshold:
                    town_tallies[key]["score"] += 1
                    town_tallies[key]["details"].append(f"{product}: {val:.2f}")
            
            print(f"[{product}] Finished scanning 1,146 towns in {time.time()-t3:.2f}s.")
            ds.close()
            os.remove(local_grib)
            logs.append(f"✅ Successfully scanned: {product}")
        except Exception as e:
            logs.append(f"❌ Crash on {product}: {str(e)}")
            if os.path.exists(local_grib): os.remove(local_grib)

    st.session_state['pipeline_diagnostic_logs'] = logs
    for town_key, data in town_tallies.items():
        if data["score"] >= 1:
            results[town_key] = {
                "Consensus Score": f"{data['score']} of 4 Metrics Broken",
                "Trigger Details": data["details"]
            }
    return results

# --- RENDERING THE MAP LAYERS ---
def render_map(cwa_layer, city_shapes, show_radar, radar_opacity_val, warnings_data, show_warnings, lsr_data, show_lsrs):
    layers = []
    # Radar
    layers.append(pdk.Layer(
        "BitmapLayer",
        image="https://mesonet.agron.iastate.edu/cgi-bin/wms/nexrad/n0q.cgi?service=WMS&request=GetMap&version=1.1.1&layers=nexrad-n0q&srs=EPSG:3857&bbox=-12245143.98,4865942.28,-10018754.17,6799982.72&width=2302&height=2000&format=image/png&transparent=true",
        bounds=[-110.0, 40.0, -90.0, 52.0], opacity=radar_opacity_val, visible=show_radar
    ))
    # Outline CWA
    layers.append(pdk.Layer("GeoJsonLayer", cwa_layer, stroke_width=3, get_line_color=[0, 150, 255, 255], get_fill_color=[0, 0, 0, 0]))
    # Towns
    layers.append(pdk.Layer("GeoJsonLayer", city_shapes, get_line_color="properties.line_color", get_fill_color="properties.fill_color", pickable=True))
    
    return pdk.Deck(
        layers=layers, initial_view_state=pdk.ViewState(latitude=45.5, longitude=-100.0, zoom=5.5), map_style="light",
        tooltip={"html": "<b>{name}</b><br/>{hover_info}"}
    )

with st.sidebar:
    st.markdown("#### Map Layers:")
    toggle_radar = st.checkbox("Overlay Base Reflectivity", value=False)
    radar_opacity = st.slider("Radar Opacity", 0.0, 1.0, 0.55, 0.05)
    toggle_warnings = st.checkbox("Overlay FAYs and FFWs", value=False)
    toggle_lsrs = st.checkbox("Overlay Flash Flood LSRs", value=False)

# --- EXECUTE CORE SCANS ---
with st.spinner("Analyzing current regional CWA footprints..."):
    alert_results = scan_data(count)
    live_warnings = {"type": "FeatureCollection", "features": []}
    live_lsrs = {"type": "FeatureCollection", "features": []}

for feature in urban_shapes_geojson["features"]:
    feature["properties"]["fill_color"] = [210, 210, 210, 90]     
    feature["properties"]["line_color"] = [160, 160, 160, 120]     
    feature["properties"]["hover_info"] = "Monitoring 4-Product Hazard Consensus"
    
if alert_results:
    alerted_towns = [key.split(",")[0].strip().upper() for key in alert_results.keys()]
    for feature in urban_shapes_geojson["features"]:
        feat_name = str(feature["properties"].get("name", "")).upper()
        if any(town in feat_name for town in alerted_towns):
            feature["properties"]["fill_color"] = [255, 0, 0, 200]  
            feature["properties"]["line_color"] = [150, 0, 0, 255]
            feature["properties"]["hover_info"] = "🚨 DIAGNOSTIC MODE: 1+ HAZARD THRESHOLD EXCEEDED"

st.subheader("Urban and Small Towns Flash Flood Alert Map")
st.pydeck_chart(render_map(cwa_geojson, urban_shapes_geojson, toggle_radar, radar_opacity, live_warnings, toggle_warnings, live_lsrs, toggle_lsrs))

with st.sidebar.expander("🛠️ Live Data Pipeline Diagnostic Logs", expanded=True):
    if 'pipeline_diagnostic_logs' in st.session_state:
        for log in st.session_state['pipeline_diagnostic_logs']: st.write(log)

if alert_results:
    st.error("🚨 THRESHOLDS EXCEEDED WITHIN OPERATIONAL REGIONS:")
    st.json(alert_results)
else:
    st.success("✅ No hydro hazards detected across operational domains.")
