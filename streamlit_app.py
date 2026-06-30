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
RAIN_RATE_PROD = "PrecipRate_00.00"
RAIN_RATE_THRESH = 50.8                           # 2.0 in/hr -> 50.8 mm/hr

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
        😖 CAUTION: THIS WEBSITE IS CURRENTLY HAVING A MENTAL BREAKDOWN AND IS UNSTABLE. AVOID WITH CAUTION. DO NOT ENGAGE. CRASH IMMINENT. 
    </div>
""")

# --- AUTOMATED OPERATIONS TIMER ---
count = st_autorefresh(interval=120000, limit=None, key="mrms_auto_scanner")

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
**BLUF:** This real-time tool will flash red for any city or small town that is at risk for flash flooding when **at least 3 out of the 4** product thresholds are met strictly within the city limits.
""")

col1, col2, col3 = st.columns([2, 2, 1])

with col1:
    st.markdown("""
    #### Monitored Products & Thresholds (3/4 must be met):
    * MRMS 1-hr QPE: $\ge$ 1.0"
    * MRMS Instantaneous Rain Rates: $\ge$ 2.0"/1-hr (sustained over at least 3 scans)
    * FLASH CREST Max Unit Streamflow: $\ge$ 200 cfs/sq. mi.
    * FLASH Hydrophobic Max Unit Streamflow: $\ge$ 1000 cfs/sq. mi.
    """)

with col2:
    st.markdown("""
    #### Map Symbology:
    * **Dark Gray Polygons:** Spatial boundary extent of all 1,146 monitored urban areas and small towns.
    * **Solid Red Polygons:** 3 out of 4 MRMS products exceed the thresholds anywhere strictly within the city boundaries.
    * **Alert Timing:** Alerts update live. To account for urban runoff and drainage lag, alerts will remain active 10 minutes after product thresholds have dropped below the required criteria.
    * **Automated Refresh:** Updates every 2-minutes to sync with live MRMS data feed.
    """, unsafe_allow_html=True)

with col3:
    st.markdown("#### Map Layers:")
    toggle_radar = st.checkbox("Overlay Base Reflectivity", value=False, help="Toggles live IEM NEXRAD Base Reflectivity mosaic over the map area.")
    radar_opacity = st.slider(
        "Radar Opacity", 
        min_value=0.0, max_value=1.0, value=0.55, step=0.05,
        help="Adjust the transparency of the Base Reflectivity overlay layer."
    )
    toggle_warnings = st.checkbox("Overlay FAYs and FFWs", value=False, help="Toggles active NWS Flood Advisories (Light Green) and Flash Flood Warnings (Dark Green).")
    toggle_lsrs = st.checkbox("Overlay Flash Flood LSRs", value=False, help="Toggles NWS Local Storm Reports (LSRs) for Flash Flooding over the past 24 hours.")


# --- DATABASE LOADING ---
@st.cache_data
def get_urban_centers():
    df = pd.read_csv("urban_centers.csv")
    df['min_lon'] = pd.to_numeric(df['min_lon'], errors='coerce')
    df['max_lon'] = pd.to_numeric(df['max_lon'], errors='coerce')
    df['min_lat'] = pd.to_numeric(df['min_lat'], errors='coerce')
    df['max_lat'] = pd.to_numeric(df['max_lat'], errors='coerce')
    df = df.dropna(subset=['min_lon', 'max_lon', 'min_lat', 'max_lat']).copy()
    
    # --- THE BOUNDING BOX FIX ---
    # Crush the massive 100-sq-mile Overpass buffers down to strict city limits (1-mile radius)
    center_lat = (df['min_lat'] + df['max_lat']) / 2.0
    center_lon = (df['min_lon'] + df['max_lon']) / 2.0
    
    # Detect any boundary bigger than ~3 miles across
    oversized = (df['max_lat'] - df['min_lat']) > 0.04
    
    # Force them into a highly accurate 1-mile box to perfectly match MRMS pixels
    df.loc[oversized, 'min_lat'] = center_lat[oversized] - 0.014
    df.loc[oversized, 'max_lat'] = center_lat[oversized] + 0.014
    df.loc[oversized, 'min_lon'] = center_lon[oversized] - 0.020
    df.loc[oversized, 'max_lon'] = center_lon[oversized] + 0.020
    
    return df

# FIX: Removed the @st.cache_data decorator so Streamlit stops holding onto the old, melted JSON file!
def load_json_layer(filepath):
    try:
        with open(filepath, "r") as f:
            return json.load(f)
    except:
        return {"type": "FeatureCollection", "features": []}

@st.cache_data
def generate_hybrid_urban_shapes(csv_df, existing_geojson):
    """
    Loads detailed large city boundaries from JSON, then injects 
    custom rectangular boundaries for all missing small towns from the CSV.
    """
    combined_geojson = copy.deepcopy(existing_geojson)
    existing_names = [str(feat["properties"].get("name", "")).strip().upper() for feat in combined_geojson.get("features", [])]
    
    for _, row in csv_df.iterrows():
        town_name = f"{row['name']}, {row['state']}".strip().upper()
        
        if town_name not in existing_names:
            min_lon_raw = row['min_lon'] if row['min_lon'] < 0 else -row['min_lon']
            max_lon_raw = row['max_lon'] if row['max_lon'] < 0 else -row['max_lon']
            
            true_min_lon = min(min_lon_raw, max_lon_raw)
            true_max_lon = max(min_lon_raw, max_lon_raw)
            true_min_lat = min(row['min_lat'], row['max_lat'])
            true_max_lat = max(row['min_lat'], row['max_lat'])
            
            feature = {
                "type": "Feature",
                "properties": {
                    "name": f"{row['name']}, {row['state']}"
                },
                "geometry": {
                    "type": "Polygon",
                    "coordinates": [[
                        [true_min_lon, true_min_lat],
                        [true_max_lon, true_min_lat],
                        [true_max_lon, true_max_lat],
                        [true_min_lon, true_max_lat],
                        [true_min_lon, true_min_lat]
                    ]]
                }
            }
            combined_geojson["features"].append(feature)
            
    for feature in combined_geojson["features"]:
        feature["properties"]["fill_color"] = [100, 100, 100, 160]     
        feature["properties"]["line_color"] = [70, 70, 70, 200]     
        feature["properties"]["hover_info"] = "Monitoring 4-Product Hazard Consensus"
        
    return combined_geojson


# Load databases
urban_gdf = get_urban_centers()
cwa_geojson = load_json_layer("cwa_outlines.json")

# Process the new CWA GeoJSON to add dynamic WFO hover tooltips
for feat in cwa_geojson.get("features", []):
    wfo_id = feat.get("properties", {}).get("WFO", "Unknown")
    feat["properties"]["name"] = wfo_id
    feat["properties"]["hover_info"] = ""

raw_urban_boundaries = load_json_layer("urban_boundaries.json")

# Generate the hybrid polygon map featuring all 1,146 locations
urban_shapes_geojson = generate_hybrid_urban_shapes(urban_gdf, raw_urban_boundaries)


# --- NWS WARNINGS ENGINE ---
@st.cache_data(ttl=120, show_spinner=False)
def get_nws_warnings():
    url = "https://api.weather.gov/alerts/active?area=ND,SD,MN,MT,WY"
    req = urllib.request.Request(url, headers={'User-Agent': 'UrbanFF-Prototype'})
    filtered_features = []
    try:
        with urllib.request.urlopen(req, timeout=10) as response:
            data = json.loads(response.read().decode())
            for feature in data.get("features", []):
                event = feature["properties"].get("event", "")
                if event in ["Flash Flood Warning", "Flood Advisory"]:
                    headline = feature["properties"].get("headline", "Active Warning")
                    raw_area = feature["properties"].get("areaDesc", "Unknown Area")
                    formatted_areas = []
                    for a in raw_area.split(";"):
                        a = a.strip()
                        if "County" not in a and a != "Unknown Area":
                            if "," in a:
                                parts = a.split(",", 1)
                                formatted_areas.append(f"{parts[0].strip()} County, {parts[1].strip()}")
                            else:
                                formatted_areas.append(f"{a} County")
                        else:
                            formatted_areas.append(a)
                    clean_area = ", ".join(formatted_areas)
                    feature["properties"]["name"] = f"⚠️ {event}"
                    feature["properties"]["hover_info"] = f"<b>Details:</b> {headline}<br/><b>Affected Counties:</b> {clean_area}"
                    if event == "Flash Flood Warning":
                        feature["properties"]["fill_color"] = [0, 128, 0, 40]       
                        feature["properties"]["line_color"] = [0, 100, 0, 255]      
                    else: 
                        feature["properties"]["fill_color"] = [144, 238, 144, 50]   
                        feature["properties"]["line_color"] = [50, 205, 50, 255]    
                    filtered_features.append(feature)
            return {"type": "FeatureCollection", "features": filtered_features}
    except Exception:
        return {"type": "FeatureCollection", "features": []}


# --- LOCAL STORM REPORTS ENGINE ---
@st.cache_data(ttl=120, show_spinner=False)
def get_lsrs():
    url = "https://mesonet.agron.iastate.edu/geojson/lsr.geojson?states=ND,SD,MN,MT,WY&hours=24"
    req = urllib.request.Request(url, headers={'User-Agent': 'UrbanFF-Prototype'})
    filtered_features = []
    try:
        with urllib.request.urlopen(req, timeout=10) as response:
            data = json.loads(response.read().decode())
            for feature in data.get("features", []):
                event_type = str(feature["properties"].get("type", "")).upper()
                if event_type == "FLASH FLOOD":
                    remark = feature["properties"].get("remark", "No additional details provided.")
                    city = feature["properties"].get("city", "Unknown")
                    county = feature["properties"].get("county", "Unknown")
                    feature["properties"]["name"] = "💧 Flash Flood LSR"
                    feature["properties"]["hover_info"] = f"<b>Location:</b> {city} ({county} County)<br/><b>Report:</b> {remark}"
                    feature["properties"]["fill_color"] = [255, 140, 0, 220]
                    feature["properties"]["line_color"] = [255, 255, 255, 255]
                    filtered_features.append(feature)
            return {"type": "FeatureCollection", "features": filtered_features}
    except Exception:
        return {"type": "FeatureCollection", "features": []}


# --- CACHED FILE LIST LAYER ---
@st.cache_data(ttl=60, show_spinner=False)
def get_latest_files(product_name, num_files=1):
    fs = s3fs.S3FileSystem(anon=True, use_listings_cache=False)
    now_utc = datetime.now(timezone.utc)
    today_str = now_utc.strftime("%Y%m%d")
    yesterday_str = (now_utc - timedelta(days=1)).strftime("%Y%m%d")
    
    all_files = []
    
    path_yesterday = f"noaa-mrms-pds/CONUS/{product_name}/{yesterday_str}/"
    try:
        files = fs.ls(path_yesterday, refresh=True)
        all_files.extend([f for f in files if f.endswith('.grib2.gz')])
    except Exception:
        pass
        
    path_today = f"noaa-mrms-pds/CONUS/{product_name}/{today_str}/"
    try:
        files = fs.ls(path_today, refresh=True)
        all_files.extend([f for f in files if f.endswith('.grib2.gz')])
    except Exception:
        pass
        
    if all_files:
        return sorted(all_files)[-num_files:]
    return []


def extract_file(s3_path, idx_suffix=""):
    fs = s3fs.S3FileSystem(anon=True, use_listings_cache=False)
    temporal_id = time.time_ns()
    local_gz = f"temp_{idx_suffix}_{temporal_id}.grib2.gz"
    local_grib = f"temp_{idx_suffix}_{temporal_id}.grib2"
    try:
        fs.get(s3_path, local_gz)
        with gzip.open(local_gz, 'rb') as f_in:
            with open(local_grib, 'wb') as f_out:
                shutil.copyfileobj(f_in, f_out)
        if os.path.exists(local_gz):
            os.remove(local_gz)
        return local_grib
    except Exception:
        if os.path.exists(local_gz): os.remove(local_gz)
        if os.path.exists(local_grib): os.remove(local_grib)
        return None


# --- CONSENSUS CROSS-DATASET EVALUATION ENGINE ---
@st.cache_data(show_spinner=False)
def scan_data(cycle_count, towns_df):
    results = {}
    logs = []
    feed_health = {
        "RadarOnly_QPE_01H_00.00": "🔴 Offline",
        "PrecipRate_00.00": "🔴 Offline",
        "FLASH_CREST_MAXUNITSTREAMFLOW_00.00": "🔴 Offline",
        "FLASH_HP_MAXUNITSTREAMFLOW_00.00": "🔴 Offline"
    }
    
    town_tallies = {f"{row['name']}, {row['state']}": {"score": 0, "details": []} for _, row in towns_df.iterrows()}
    
    master_lat_box = [41.5, 50.0]
    now_utc = datetime.now(timezone.utc)
    
    # --- SCAN CORE PRODUCTS BLOCK ---
    for product, threshold in PRODUCTS.items():
        latest_files = get_latest_files(product, num_files=1)
        if not latest_files: 
            logs.append(f"❌ Could not find recent files for {product} on NOAA S3.")
            feed_health[product] = "🔴 Missing S3 Data"
            continue
            
        s3_path = latest_files[0]
        try:
            t_str = s3_path.split('_')[-1].split('.')[0]
            dt_obj = datetime.strptime(t_str, "%Y%m%d-%H%M%S").replace(tzinfo=timezone.utc)
            scan_time = dt_obj.strftime("%H:%M UTC")
            
            age_minutes = (now_utc - dt_obj).total_seconds() / 60.0
            if age_minutes > 60:
                logs.append(f"⚠️ {product} is stale ({int(age_minutes)} mins old). Skipping.")
                feed_health[product] = f"🟡 Stale ({int(age_minutes)}m)"
                continue
        except:
            scan_time = "Live Scan"
            
        local_grib = extract_file(s3_path, product)
        if not local_grib: 
            logs.append(f"❌ Failed to extract grib file for {product}.")
            feed_health[product] = "🔴 Extract Failed"
            continue
            
        try:
            ds = xr.open_dataset(local_grib, engine="cfgrib", backend_kwargs={'indexpath': ''})
            var_name = list(ds.data_vars)[0]
            
            # Universal Coordinate Detection: Safely handle 0-360 vs -180/180
            is_360 = bool(np.max(ds.longitude.values) > 180)
            target_min_lon = (360 - 107.0) if is_360 else -107.0
            target_max_lon = (360 - 93.5) if is_360 else -93.5
            master_lon_slice = slice(target_min_lon, target_max_lon)
            
            lat_ascending = bool(ds.latitude[0] < ds.latitude[-1])
            master_lat_slice = slice(min(master_lat_box), max(master_lat_box)) if lat_ascending else slice(max(master_lat_box), min(master_lat_box))
            
            ds_cropped = ds.sel(latitude=master_lat_slice, longitude=master_lon_slice).load()
            
            lats_arr = ds_cropped.latitude.values
            lons_arr = ds_cropped.longitude.values
            
            # Explicit axis transposition guaranteeing [latitude, longitude] ordering
            da = ds_cropped[var_name].squeeze()
            if da.dims != ('latitude', 'longitude'):
                da = da.transpose('latitude', 'longitude')
            data_arr = da.values
            
            for _, row in towns_df.iterrows():
                key = f"{row['name']}, {row['state']}"
                
                c_min_lat, c_max_lat = min(row['min_lat'], row['max_lat']), max(row['min_lat'], row['max_lat'])
                
                # Correctly handle longitude negativity
                c_min_lon_raw = row['min_lon'] if row['min_lon'] < 0 else -row['min_lon']
                c_max_lon_raw = row['max_lon'] if row['max_lon'] < 0 else -row['max_lon']
                true_min_lon = min(c_min_lon_raw, c_max_lon_raw)
                true_max_lon = max(c_min_lon_raw, c_max_lon_raw)
                
                c_target_min_lon = (true_min_lon % 360) if is_360 else true_min_lon
                c_target_max_lon = (true_max_lon % 360) if is_360 else true_max_lon
                
                # STRICT LIMITS: Removed the 0.015 pad to prevent bleeding into neighboring thunderstorms
                lat_idx = np.where((lats_arr >= c_min_lat) & (lats_arr <= c_max_lat))[0]
                lon_idx = np.where((lons_arr >= c_target_min_lon) & (lons_arr <= c_target_max_lon))[0]
                
                # Centroid fallback guarantees we snap to the exact 1km town center if it falls completely between pixels
                if len(lat_idx) == 0:
                    center_lat = (c_min_lat + c_max_lat) / 2.0
                    lat_idx = [np.argmin(np.abs(lats_arr - center_lat))]
                if len(lon_idx) == 0:
                    center_lon = (c_target_min_lon + c_target_max_lon) / 2.0
                    lon_idx = [np.argmin(np.abs(lons_arr - center_lon))]
                
                if len(lat_idx) > 0 and len(lon_idx) > 0:
                    min_lat_i, max_lat_i = np.min(lat_idx), np.max(lat_idx)
                    min_lon_i, max_lon_i = np.min(lon_idx), np.max(lon_idx)
                    
                    slice_data = data_arr[min_lat_i:max_lat_i+1, min_lon_i:max_lon_i+1]
                    
                    # MISSING DATA MASK: Strictly forces the engine to ignore NOAA's -99/-3/9999 no-data flags
                    valid_data = slice_data[(slice_data >= 0) & (slice_data < 99990)]
                    val = np.nanmax(valid_data) if valid_data.size > 0 else np.nan
                else:
                    val = np.nan
                
                if pd.notna(val) and val >= threshold:
                    town_tallies[key]["score"] += 1
                    
                    if product == "RadarOnly_QPE_01H_00.00":
                        v_in = float(val / 25.4)
                        town_tallies[key]["details"].append(f"1-hr QPE: {v_in:.2f} in (Thresh: 1.00 in) @ {scan_time}")
                    elif product == "FLASH_CREST_MAXUNITSTREAMFLOW_00.00":
                        v_cfs = int(round(val * 91.464))
                        town_tallies[key]["details"].append(f"CREST Unit Flow: {v_cfs} cfs/sq mi (Thresh: 200) @ {scan_time}")
                    elif product == "FLASH_HP_MAXUNITSTREAMFLOW_00.00":
                        v_cfs = int(round(val * 91.464))
                        town_tallies[key]["details"].append(f"Hydrophobic Unit Flow: {v_cfs} cfs/sq mi (Thresh: 1000) @ {scan_time}")
                        
            ds.close()
            if os.path.exists(local_grib): os.remove(local_grib)
            logs.append(f"✅ Successfully scanned: {product}")
            feed_health[product] = "🟢 Active & Loaded"
        except Exception as e:
            logs.append(f"❌ Crash on {product}: {str(e)}")
            feed_health[product] = "🟡 Parse Error"
            if os.path.exists(local_grib): os.remove(local_grib)


    # --- SCAN SUSTAINED INSTANTANEOUS RAIN RATE BLOCK (3 SCANS) ---
    rate_history_files = get_latest_files(RAIN_RATE_PROD, num_files=3)
    if len(rate_history_files) == 3:
        time_window = []
        stale_feed = False
        for f in rate_history_files:
            try:
                t_str = f.split('_')[-1].split('.')[0]
                dt_obj = datetime.strptime(t_str, "%Y%m%d-%H%M%S").replace(tzinfo=timezone.utc)
                time_window.append(dt_obj.strftime("%H:%M UTC"))
                if (now_utc - dt_obj).total_seconds() / 60.0 > 60:
                    stale_feed = True
            except:
                pass
        
        if stale_feed:
            logs.append(f"⚠️ {RAIN_RATE_PROD} contains data >60 mins old. Skipping.")
            feed_health[RAIN_RATE_PROD] = "🟡 Stale Feed"
        else:
            if len(time_window) > 1:
                time_range_str = f"[{time_window[0]} to {time_window[-1]}]"
            else:
                time_range_str = "[Live Scans]"
                
            try:
                local_gribs = [extract_file(f, f"rate_{i}") for i, f in enumerate(rate_history_files)]
                datasets = [xr.open_dataset(g, engine="cfgrib", backend_kwargs={'indexpath': ''}) for g in local_gribs if g]
                
                if len(datasets) == 3:
                    is_360 = bool(np.max(datasets[0].longitude.values) > 180)
                    target_min_lon = (360 - 107.0) if is_360 else -107.0
                    target_max_lon = (360 - 93.5) if is_360 else -93.5
                    master_lon_slice = slice(target_min_lon, target_max_lon)
                    
                    lat_ascending = bool(datasets[0].latitude[0] < datasets[0].latitude[-1])
                    master_lat_slice = slice(min(master_lat_box), max(master_lat_box)) if lat_ascending else slice(max(master_lat_box), min(master_lat_box))
                    
                    cropped_ds = [d.sel(latitude=master_lat_slice, longitude=master_lon_slice).load() for d in datasets]
                    var_names = [list(d.data_vars)[0] for d in cropped_ds]
                    
                    lats_arr = cropped_ds[0].latitude.values
                    lons_arr = cropped_ds[0].longitude.values
                    
                    # Ensure perfectly transposed [lat, lon] arrays
                    da0 = cropped_ds[0][var_names[0]].squeeze()
                    if da0.dims != ('latitude', 'longitude'): da0 = da0.transpose('latitude', 'longitude')
                    data_arr_0 = da0.values
                    
                    da1 = cropped_ds[1][var_names[1]].squeeze()
                    if da1.dims != ('latitude', 'longitude'): da1 = da1.transpose('latitude', 'longitude')
                    data_arr_1 = da1.values
                    
                    da2 = cropped_ds[2][var_names[2]].squeeze()
                    if da2.dims != ('latitude', 'longitude'): da2 = da2.transpose('latitude', 'longitude')
                    data_arr_2 = da2.values
                    
                    for _, row in towns_df.iterrows():
                        key = f"{row['name']}, {row['state']}"
                        
                        c_min_lat, c_max_lat = min(row['min_lat'], row['max_lat']), max(row['min_lat'], row['max_lat'])
                        
                        # Correctly handle longitude negativity
                        c_min_lon_raw = row['min_lon'] if row['min_lon'] < 0 else -row['min_lon']
                        c_max_lon_raw = row['max_lon'] if row['max_lon'] < 0 else -row['max_lon']
                        true_min_lon = min(c_min_lon_raw, c_max_lon_raw)
                        true_max_lon = max(c_min_lon_raw, c_max_lon_raw)
                        
                        c_target_min_lon = (true_min_lon % 360) if is_360 else true_min_lon
                        c_target_max_lon = (true_max_lon % 360) if is_360 else true_max_lon
                        
                        # STRICT LIMITS: Removed pad to prevent bleeding
                        lat_idx = np.where((lats_arr >= c_min_lat) & (lats_arr <= c_max_lat))[0]
                        lon_idx = np.where((lons_arr >= c_target_min_lon) & (lons_arr <= c_target_max_lon))[0]
                        
                        if len(lat_idx) == 0:
                            center_lat = (c_min_lat + c_max_lat) / 2.0
                            lat_idx = [np.argmin(np.abs(lats_arr - center_lat))]
                        if len(lon_idx) == 0:
                            center_lon = (c_target_min_lon + c_target_max_lon) / 2.0
                            lon_idx = [np.argmin(np.abs(lons_arr - center_lon))]
                        
                        if len(lat_idx) > 0 and len(lon_idx) > 0:
                            min_lat_i, max_lat_i = np.min(lat_idx), np.max(lat_idx)
                            min_lon_i, max_lon_i = np.min(lon_idx), np.max(lon_idx)
                            
                            slice1 = data_arr_0[min_lat_i:max_lat_i+1, min_lon_i:max_lon_i+1]
                            slice2 = data_arr_1[min_lat_i:max_lat_i+1, min_lon_i:max_lon_i+1]
                            slice3 = data_arr_2[min_lat_i:max_lat_i+1, min_lon_i:max_lon_i+1]
                            
                            # MISSING DATA MASK: Ignores NOAA -99/-3/9999 error codes
                            valid1 = slice1[(slice1 >= 0) & (slice1 < 99990)]
                            valid2 = slice2[(slice2 >= 0) & (slice2 < 99990)]
                            valid3 = slice3[(slice3 >= 0) & (slice3 < 99990)]
                            
                            v1 = np.nanmax(valid1) if valid1.size > 0 else np.nan
                            v2 = np.nanmax(valid2) if valid2.size > 0 else np.nan
                            v3 = np.nanmax(valid3) if valid3.size > 0 else np.nan
                        else:
                            v1, v2, v3 = np.nan, np.nan, np.nan
                        
                        if pd.notna(v1) and pd.notna(v2) and pd.notna(v3):
                            if v1 >= RAIN_RATE_THRESH and v2 >= RAIN_RATE_THRESH and v3 >= RAIN_RATE_THRESH:
                                town_tallies[key]["score"] += 1
                                
                                pk_hr = float(min(v1, v2, v3) / 25.4)
                                town_tallies[key]["details"].append(f"Sustained Rain Rate: {pk_hr:.2f} in/hr (Thresh: 2.00 in/hr) {time_range_str}")
                                
                    for d in datasets: d.close()
                    for g in local_gribs:
                        if g and os.path.exists(g): os.remove(g)
                    logs.append(f"✅ Successfully scanned: {RAIN_RATE_PROD} (3-Scan Multi-Layer History)")
                    feed_health[RAIN_RATE_PROD] = "🟢 Active (3-Scans)"
            except Exception as e:
                logs.append(f"❌ Rain Rate History processing error: {str(e)}")
                feed_health[RAIN_RATE_PROD] = "🟡 Parse Error"
    else:
        logs.append(f"⚠️ {RAIN_RATE_PROD} waiting for more scans ({len(rate_history_files)}/3)")
        feed_health[RAIN_RATE_PROD] = f"🟡 WAITING ({len(rate_history_files)}/3 Scans)"

    st.session_state['pipeline_diagnostic_logs'] = logs

    # LOCKED ALERTS ENGINE TO CRITICAL THRESHOLD SCORE: 3 OUT OF 4 METRICS
    for town_key, data in town_tallies.items():
        if data["score"] >= 3:
            results[town_key] = {
                "Consensus Score": f"{data['score']} of 4 Metrics Broken",
                "Trigger Details": data["details"]
            }
    return results, logs, feed_health

# --- RENDERING THE MAP LAYERS ---
def render_map(cwa_layer, city_shapes, show_radar, radar_opacity_val, warnings_data, show_warnings, lsr_data, show_lsrs):
    layers = []
    radar_layer = pdk.Layer(
        "BitmapLayer",
        image="https://mesonet.agron.iastate.edu/cgi-bin/wms/nexrad/n0q.cgi?service=WMS&request=GetMap&version=1.1.1&layers=nexrad-n0q&srs=EPSG:3857&bbox=-12245143.98,4865942.28,-10018754.17,6799982.72&width=2302&height=2000&format=image/png&transparent=true",
        bounds=[-110.0, 40.0, -90.0, 52.0],
        opacity=radar_opacity_val, visible=show_radar
    )
    layers.append(radar_layer)

    # 1. CWA Perimeters (Light Blue border)
    outline_layer = pdk.Layer(
        "GeoJsonLayer", cwa_layer, 
        stroked=True,
        get_line_color=[135, 206, 250, 255], 
        get_fill_color=[0, 0, 0, 0], 
        get_line_width=3000, 
        line_width_min_pixels=3, 
        pickable=True
    )
    layers.append(outline_layer)

    nws_warnings_layer = pdk.Layer(
        "GeoJsonLayer", warnings_data,
        get_line_color="properties.line_color", get_fill_color="properties.fill_color",
        stroke_width=3, line_width_min_pixels=2, pickable=True, visible=show_warnings
    )
    layers.append(nws_warnings_layer)

    urban_polygon_layer = pdk.Layer(
        "GeoJsonLayer", city_shapes,
        get_line_color="properties.line_color", get_fill_color="properties.fill_color",
        pickable=True, extruded=False,
        update_triggers={"get_fill_color": ["properties.fill_color"]}
    )
    layers.append(urban_polygon_layer)
    
    lsr_layer = pdk.Layer(
        "GeoJsonLayer", lsr_data,
        get_line_color="properties.line_color", get_fill_color="properties.fill_color",
        get_point_radius=3500, point_radius_min_pixels=6, pickable=True, visible=show_lsrs
    )
    layers.append(lsr_layer)
    
    return pdk.Deck(
        layers=layers,
        initial_view_state=pdk.ViewState(latitude=45.5, longitude=-100.0, zoom=5.5, pitch=0),
        map_style="light", 
        tooltip={
            "html": "<b style='color: #4AA4DE;'>{name}</b><br/>{hover_info}", 
            "style": {"backgroundColor": "#222222", "color": "white", "maxWidth": "300px"}
        }
    )

# --- EXECUTE CORE SCANS ---
with st.spinner("Analyzing current regional CWA footprints..."):
    active_alert_results, pipeline_logs, feed_health = scan_data(count, urban_gdf)
    live_warnings = get_nws_warnings()
    live_lsrs = get_lsrs()

# --- 10-MINUTE IMPACT COOLDOWN LOGIC ---
if 'alert_history' not in st.session_state:
    st.session_state['alert_history'] = {}

current_utc_time = datetime.now(timezone.utc)
alert_results = {}

# 1. Update memory history with newly triggered active alerts
for town_key, data in active_alert_results.items():
    st.session_state['alert_history'][town_key] = {
        "time": current_utc_time,
        "data": data
    }
    alert_results[town_key] = data

# 2. Check history for towns resting in the cooldown window
keys_to_remove = []
# Ensure stable iteration over dictionary items
for town_key, hist in list(st.session_state['alert_history'].items()):
    if town_key not in active_alert_results:
        time_since_trigger = current_utc_time - hist["time"]
        if time_since_trigger <= timedelta(minutes=10):
            # Town is in the 10-minute cooldown window
            cooldown_data = hist["data"].copy()
            cooldown_data["Consensus Score"] = "In 10-Min Impact Cooldown (Runoff Lag)"
            alert_results[town_key] = cooldown_data
        else:
            # Cooldown completely expired
            keys_to_remove.append(town_key)

# 3. Clean up expired alerts from background memory
for k in keys_to_remove:
    del st.session_state['alert_history'][k]

st.session_state['pipeline_diagnostic_logs'] = pipeline_logs
st.session_state['feed_health'] = feed_health

# Map the active alerts to the GeoJSON polygon layer 
upper_alert_results = {k.strip().upper(): v for k, v in alert_results.items()}

for feature in urban_shapes_geojson["features"]:
    feat_name = str(feature["properties"].get("name", "")).strip().upper()
    
    # EXACT name match to prevent substring overlap bugs (e.g. "Ray, ND" triggering "Raymond, ND")
    if feat_name in upper_alert_results:
        feature["properties"]["fill_color"] = [255, 0, 0, 200]  
        feature["properties"]["line_color"] = [150, 0, 0, 255]
        
        # Dynamically change the hover tooltip based on whether it is an active threat or just draining
        if "Cooldown" in upper_alert_results[feat_name].get("Consensus Score", ""):
            feature["properties"]["hover_info"] = "⚠️ RUNOFF LAG: 10-Min Drainage Cooldown Active"
        else:
            feature["properties"]["hover_info"] = "🚨 CRITICAL: 3+ HAZARD THRESHOLDS EXCEEDED"
    else:
        # Re-apply base state in case the map re-renders after an alert expires
        feature["properties"]["fill_color"] = [100, 100, 100, 160]     
        feature["properties"]["line_color"] = [70, 70, 70, 200]     
        feature["properties"]["hover_info"] = "Monitoring 4-Product Hazard Consensus"

st.subheader("Urban and Small Towns Flash Flood Alert Map")

# --- NEW: DATA FEED HEALTH DASHBOARD ---
st.markdown("##### 🌱 Live Data Feed Health")
health_cols = st.columns(4)
friendly_names = {
    "RadarOnly_QPE_01H_00.00": "MRMS 1-hr QPE",
    "PrecipRate_00.00": "MRMS Rain Rates",
    "FLASH_CREST_MAXUNITSTREAMFLOW_00.00": "FLASH CREST Flow",
    "FLASH_HP_MAXUNITSTREAMFLOW_00.00": "FLASH Hydrophobic Flow"
}
for i, (prod, name) in enumerate(friendly_names.items()):
    status = st.session_state.get('feed_health', {}).get(prod, "⏳ Pending")
    health_cols[i].info(f"**{name}**\n\n{status}")

st.pydeck_chart(render_map(
    cwa_geojson, urban_shapes_geojson, 
    toggle_radar, radar_opacity, 
    live_warnings, toggle_warnings, 
    live_lsrs, toggle_lsrs
))

with st.sidebar.expander("🛠️ Live Data Pipeline Diagnostic Logs", expanded=True):
    if 'pipeline_diagnostic_logs' in st.session_state:
        for log in st.session_state['pipeline_diagnostic_logs']:
            st.write(log)
    else:
        st.write("Initializing connections to NOAA data feeds...")

if alert_results:
    st.error("🚨 THRESHOLDS EXCEEDED WITHIN OPERATIONAL REGIONS:")
    st.json(alert_results)
else:
    st.success("✅ No urban hydro hazards detected - at ease soldier.")

if st.button("Refresh & Scan"):
    st.rerun()
