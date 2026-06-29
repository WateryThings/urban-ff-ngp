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
        /* Eliminate master block container top padding */
        .block-container {
            padding-top: 1rem !important;
            padding-bottom: 1rem !important;
            max-width: 98% !important;
        }
        /* Tighten margins on elements to pull content upward */
        h1, h2, h3, h4 {
            margin-top: 0.2rem !important;
            margin-bottom: 0.2rem !important;
            padding-top: 0px !important;
        }
        /* Compress generic element spacing blocks */
        .stElementContainer {
            margin-bottom: 0.4rem !important;
        }
    </style>
""")

# --- CAUTION WARNING BANNER ---
st.warning("⚠️ **CAUTION:** This tool is an experimental prototype (similar to C3P0 in The Phantom Menace) and will GUARANTEE, NO QUESTIONS ASKED FAIL, EVEN NOW, THIS SECOND!")

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

# --- BLUF & OPERATIONAL USER GUIDE (UPDATED FOR 2/4 CONSENSUS) ---
st.markdown("""
**BLUF:** This real-time tool will flash red for any city or small town that is at risk for flash flooding when 2 out of the 4 product thresholds are met within a 5-mile buffer.
""")

col1, col2, col3 = st.columns([2, 2, 1])

with col1:
    st.markdown("""
    #### Monitored Products & Thresholds:
    * MRMS 1-hr QPE: $\ge$ 1.0"
    * MRMS Instantaneous Rain Rates: $\ge$ 2.0"/1-hr (sustained over at least 3 scans)
    * FLASH CREST Max Unit Streamflow: $\ge$ 200 cfs/sq. mi.
    * FLASH Hydrophobic Max Unit Streamflow: $\ge$ 1000 cfs/sq. mi.
    """)

with col2:
    st.markdown("""
    #### Map Symbology:
    * **Translucent Gray Polygons:** Spatial extent of monitored urban areas and small towns.
    * **Solid Red Polygons:** 2 out of the 4 MRMS products exceed the listed thresholds within the buffer area. Details about this area will be displayed below the map.
    * **Automated Refresh:** Updates every 2-minutes to sync with live MRMS data feed.
    """)

with col3:
    st.markdown("#### Map Layers:")
    toggle_radar = st.checkbox("Overlay Base Reflectivity", value=False, help="Toggles live IEM NEXRAD Base Reflectivity mosaic over the map area.")
    
    radar_opacity = st.slider(
        "Radar Opacity", 
        min_value=0.0, 
        max_value=1.0, 
        value=0.55, 
        step=0.05,
        help="Adjust the transparency of the Base Reflectivity overlay layer."
    )
    
    toggle_warnings = st.checkbox("Overlay FAYs and FFWs", value=False, help="Toggles active NWS Flood Advisories (Light Green) and Flash Flood Warnings (Dark Green).")
    toggle_lsrs = st.checkbox("Overlay Flash Flood LSRs", value=False, help="Toggles NWS Local Storm Reports (LSRs) for Flash Flooding over the past 24 hours.")

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

# --- NWS WARNINGS ENGINE (API DATA FETCH) ---
@st.cache_data(ttl=120, show_spinner=False)
def get_nws_warnings():
    url = "https://api.weather.gov/alerts/active?area=ND,SD,MN,MT,WY"
    req = urllib.request.Request(url, headers={'User-Agent': 'UrbanFF-Prototype'})
    filtered_features = []
    try:
        with urllib.request.urlopen(req) as response:
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
                    else: # Flood Advisory
                        feature["properties"]["fill_color"] = [144, 238, 144, 50]   
                        feature["properties"]["line_color"] = [50, 205, 50, 255]    
                    filtered_features.append(feature)
            return {"type": "FeatureCollection", "features": filtered_features}
    except Exception:
        return {"type": "FeatureCollection", "features": []}

# --- LOCAL STORM REPORTS ENGINE (IEM API FETCH) ---
@st.cache_data(ttl=120, show_spinner=False)
def get_lsrs():
    url = "https://mesonet.agron.iastate.edu/geojson/lsr.geojson?states=ND,SD,MN,MT,WY&hours=24"
    req = urllib.request.Request(url, headers={'User-Agent': 'UrbanFF-Prototype'})
    filtered_features = []
    try:
        with urllib.request.urlopen(req) as response:
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

# --- DATA EXTRACTION & DOWNLOADING ---
def get_latest_files(fs, product_name, num_files=1):
    path = f"noaa-mrms-pds/CONUS/{product_name}/"
    try:
        files = fs.ls(path)
        grib_files = [f for f in files if f.endswith('.grib2.gz')]
        if not grib_files: return []
        return grib_files[-num_files:]
    except Exception:
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
    except Exception:
        if os.path.exists(local_gz): os.remove(local_gz)
        return None

# --- CONSENSUS CROSS-DATASET EVALUATION ENGINE ---
@st.cache_data(show_spinner=False)
def scan_data(cycle_count):
    fs = s3fs.S3FileSystem(anon=True)
    results = {}
    town_tallies = {f"{row['name']}, {row['state']}": {"score": 0, "details": []} for _, row in urban_gdf.iterrows()}
    for product, threshold in PRODUCTS.items():
        latest_files = get_latest_files(fs, product, num_files=1)
        if not latest_files: continue
        local_grib = extract_file(fs, latest_files[0], product)
        if not local_grib: continue
        try:
            ds = xr.open_dataset(local_grib, engine="cfgrib")
            var_name = list(ds.data_vars)[0]
            for _, row in urban_gdf.iterrows():
                key = f"{row['name']}, {row['state']}"
                min_lon, max_lon = row['min_lon'] % 360, row['max_lon'] % 360
                val = ds.sel(latitude=slice(row['max_lat'], row['min_lat']), longitude=slice(min_lon, max_lon))[var_name].max().values
                if pd.notna(val) and val >= threshold:
                    town_tallies[key]["score"] += 1
                    town_tallies[key]["details"].append(f"{product}: {val:.2f} (Threshold: {threshold})")
            ds.close()
            os.remove(local_grib)
        except Exception:
            if os.path.exists(local_grib): os.remove(local_grib)
    rate_history_files = get_latest_files(fs, RAIN_RATE_PROD, num_files=3)
    if len(rate_history_files) == 3:
        local_gribs = [extract_file(fs, f"rate_{i}") for i, f in enumerate(rate_history_files)]
        try:
            datasets = [xr.open_dataset(g, engine="cfgrib") for g in local_gribs if g]
            if len(datasets) == 3:
                var_names = [list(d.data_vars)[0] for d in datasets]
                for _, row in urban_gdf.iterrows():
                    key = f"{row['name']}, {row['state']}"
                    min_lon, max_lon = row['min_lon'] % 360, row['max_lon'] % 360
                    v1 = datasets[0].sel(latitude=slice(row['max_lat'], row['min_lat']), longitude=slice(min_lon, max_lon))[var_names[0]].max().values
                    v2 = datasets[1].sel(latitude=slice(row['max_lat'], row['min_lat']), longitude=slice(min_lon, max_lon))[var_names[1]].max().values
                    v3 = datasets[2].sel(latitude=slice(row['max_lat'], row['min_lat']), longitude=slice(min_lon, max_lon))[var_names[2]].max().values
                    if pd.notna(v1) and pd.notna(v2) and pd.notna(v3):
                        if v1 >= RAIN_RATE_THRESH and v2 >= RAIN_RATE_THRESH and v3 >= RAIN_RATE_THRESH:
                            town_tallies[key]["score"] += 1
                            town_tallies[key]["details"].append(f"Sustained Rain Rate (3 Scans Exceeded): Min Peak {min(v1,v2,v3):.2f}")
            for d in datasets: d.close()
        except Exception:
            pass
        for g in local_gribs:
            if g and os.path.exists(g): os.remove(g)
            
    # CRITICAL CHANGE: Triggering alerts at 2 out of 4 matching parameters
    for town_key, data in town_tallies.items():
        if data["score"] >= 2:
            results[town_key] = {
                "Consensus Score": f"{data['score']} of 4 Metrics Broken",
                "Trigger Details": data["details"]
            }
    return results

# --- RENDERING THE MAP LAYERS ---
def render_map(cwa_layer, city_shapes, show_radar, radar_opacity_val, warnings_data, show_warnings, lsr_data, show_lsrs):
    layers = []
    
    radar_layer = pdk.Layer(
        "BitmapLayer",
        image="https://mesonet.agron.iastate.edu/cgi-bin/wms/nexrad/n0q.cgi?service=WMS&request=GetMap&version=1.1.1&layers=nexrad-n0q&srs=EPSG:3857&bbox=-12245143.98,4865942.28,-10018754.17,6799982.72&width=2302&height=2000&format=image/png&transparent=true",
        bounds=[-110.0, 40.0, -90.0, 52.0],
        opacity=radar_opacity_val,
        visible=show_radar
    )
    layers.append(radar_layer)

    outline_layer = pdk.Layer(
        "GeoJsonLayer", cwa_layer, stroke_width=3,
        get_line_color=[0, 150, 255, 255], get_fill_color=[0, 0, 0, 0], line_width_min_pixels=2
    )
    layers.append(outline_layer)
    
    nws_warnings_layer = pdk.Layer(
        "GeoJsonLayer", warnings_data,
        get_line_color="properties.line_color", get_fill_color="properties.fill_color",
        stroke_width=3, line_width_min_pixels=2, pickable=True,
        visible=show_warnings
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
        get_point_radius=3500, point_radius_min_pixels=6,           
        pickable=True, visible=show_lsrs
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
    alert_results = scan_data(count)
    live_warnings = get_nws_warnings()
    live_lsrs = get_lsrs()

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
            # CRITICAL CHANGE: Updated hover banner statement for the new trigger threshold
            feature["properties"]["hover_info"] = "🚨 CRITICAL: 2+ HAZARD THRESHOLDS EXCEEDED"

st.subheader("Urban and Small Towns Flash Flood Alert Map")

st.pydeck_chart(render_map(
    cwa_geojson, urban_shapes_geojson, 
    toggle_radar, radar_opacity, 
    live_warnings, toggle_warnings, 
    live_lsrs, toggle_lsrs
))

if alert_results:
    st.error("🚨 THRESHOLDS EXCEEDED WITHIN OPERATIONAL REGIONS:")
    st.json(alert_results)
else:
    st.success("✅ No hydro hazards detected across operational domains.")

if st.button("Refresh & Scan"):
    st.rerun()
