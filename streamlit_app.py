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
    toggle_warnings = st.checkbox("Overlay FAYs and FFWs", value=False, help="Toggles active NWS Flood Advisories (Light Green) and Flash Flood Warnings (Dark Green).")
    toggle_lsrs = st.checkbox("Overlay Flash Flood LSRs", value=False, help="Toggles NWS Local Storm Reports (LSRs) for Flash Flooding over the past 24 hours.")


# --- DATABASE LOADING (HIGH-SPEED ZERO-ARGUMENT CACHING) ---
@st.cache_data
def get_urban_centers():
    df = pd.read_csv("urban_centers.csv")
    df['min_lon'] = pd.to_numeric(df['min_lon'], errors='coerce')
    df['max_lon'] = pd.to_numeric(df['max_lon'], errors='coerce')
    df['min_lat'] = pd.to_numeric(df['min_lat'], errors='coerce')
    df['max_lat'] = pd.to_numeric(df['max_lat'], errors='coerce')
    df = df.dropna(subset=['min_lon', 'max_lon', 'min_lat', 'max_lat']).copy()
    
    # Clean leading/trailing spaces and drop duplicates to prevent duplicate enclaves or ghost fallback rendering
    df['name'] = df['name'].astype(str).str.strip()
    df['state'] = df['state'].astype(str).str.strip()
    df = df.drop_duplicates(subset=['name', 'state']).copy()
    
    # --- THE BOUNDING BOX FIX ---
    center_lat = (df['min_lat'] + df['max_lat']) / 2.0
    center_lon = (df['min_lon'] + df['max_lon']) / 2.0
    
    oversized = (df['max_lat'] - df['min_lat']) > 0.04
    
    # Relaxed the "crusher" to a ~2.5 mile radius to ensure we don't miss storms hitting the edges/suburbs of towns
    df.loc[oversized, 'min_lat'] = center_lat[oversized] - 0.035
    df.loc[oversized, 'max_lat'] = center_lat[oversized] + 0.035
    df.loc[oversized, 'min_lon'] = center_lon[oversized] - 0.045
    df.loc[oversized, 'max_lon'] = center_lon[oversized] + 0.045
    
    return df

def load_json_layer(filepath):
    try:
        with open(filepath, "r") as f:
            return json.load(f)
    except:
        return {"type": "FeatureCollection", "features": []}

@st.cache_data
def get_processed_cwa_layer_final():
    """Loads and formats the CWA layer exactly once, calculating dynamic text labels."""
    cwa_geojson = load_json_layer("cwa_outlines.json")
    cwa_copy = copy.deepcopy(cwa_geojson)
    labels = []
    
    for feat in cwa_copy.get("features", []):
        props = feat.get("properties", {})
        # Smart extraction: GeoPandas sometimes saves columns in lowercase
        wfo_id = str(props.get("WFO", props.get("wfo", props.get("cwa", "Unknown")))).upper()
        feat["properties"]["name"] = wfo_id
        feat["properties"]["hover_info"] = ""
        
        # Calculate dynamic text label centroids
        geom_type = feat.get("geometry", {}).get("type", "")
        coords = feat.get("geometry", {}).get("coordinates", [])
        flat_coords = []
        
        if geom_type == "Polygon":
            for ring in coords: flat_coords.extend(ring)
        elif geom_type == "MultiPolygon":
            for poly in coords:
                for ring in poly: flat_coords.extend(ring)
                
        if flat_coords:
            lons = [c[0] for c in flat_coords if isinstance(c, list) and len(c) >= 2]
            lats = [c[1] for c in flat_coords if isinstance(c, list) and len(c) >= 2]
            if lons and lats:
                center_lon = (min(lons) + max(lons)) / 2.0
                center_lat = (min(lats) + max(lats)) / 2.0
                labels.append({
                    "wfo": wfo_id,
                    "coordinates": [center_lon, center_lat]
                })
                
    return cwa_copy, labels

@st.cache_data
def get_hybrid_urban_shapes():
    """Generates the baseline urban shapes map without relying on hashed arguments."""
    csv_df = get_urban_centers()
    existing_geojson = load_json_layer("urban_boundaries.json")
    
    combined_geojson = copy.deepcopy(existing_geojson)
    existing_features = combined_geojson.get("features", [])
    
    cleaned_features = []
    seen_names = set()
    
    # ON-THE-FLY RECONCILIATION & SANITIZATION:
    # Map state suffixes and strip trailing spaces to enable flashing and eliminate duplicate shapes
    for feature in existing_features:
        feat_name = str(feature["properties"].get("name", "")).strip()
        if not feat_name:
            continue
            
        match = csv_df[csv_df['name'].str.upper() == feat_name.upper()]
        if not match.empty:
            state = match.iloc[0]['state']
            full_name = f"{feat_name}, {state}"
        else:
            full_name = feat_name

        full_name_upper = full_name.upper()
        if full_name_upper in seen_names:
            continue  # Vaporizes any duplicate geometric slices in the GeoJSON layer
        seen_names.add(full_name_upper)
        
        feature["properties"]["name"] = full_name
        cleaned_features.append(feature)
        
    combined_geojson["features"] = cleaned_features
    existing_names = list(seen_names)
    
    for _, row in csv_df.iterrows():
        town_name = f"{row['name']}, {row['state']}".strip()
        town_name_upper = town_name.upper()
        
        if town_name_upper not in existing_names:
            existing_names.append(town_name_upper)
            
            min_lon_raw = row['min_lon'] if row['min_lon'] < 0 else -row['min_lon']
            max_lon_raw = row['max_lon'] if row['max_lon'] < 0 else -row['max_lon']
            
            true_min_lon = min(min_lon_raw, max_lon_raw)
            true_max_lon = max(min_lon_raw, max_lon_raw)
            true_min_lat = min(row['min_lat'], row['max_lat'])
            true_max_lat = max(row['min_lat'], row['max_lat'])
            
            feature = {
                "type": "Feature",
                "properties": {
                    "name": town_name
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
        feature["properties"]["hover_info"] = "Monitoring 3-Product Hazard Consensus"
        
    return combined_geojson


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
def scan_data(cycle_count):
    towns_df = get_urban_centers()
    results = {}
    logs = []
    feed_health = {
        "RadarOnly_QPE_01H_00.00": "🔴 Offline",
        "FLASH_CREST_MAXUNITSTREAMFLOW_00.00": "🔴 Offline",
        "FLASH_HP_
