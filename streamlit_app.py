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

@st.cache_data
def get_urban_centers():
    df = pd.read_csv("urban_centers.csv")
    # Force the pre-calculated edge coordinates from Spyder to be pure decimal fields
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
    feature["properties"]["fill_color"] = [180, 180, 180, 50]   # Translucent gray
    feature["properties"]["line_color"] = [120, 120, 120, 100]  # Soft gray borders

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
                # Extract the 5-mile edge-buffered boundaries and adjust for 0-360 longitude
                min_lon = row['min_lon'] % 360
                max_lon = row['max_lon'] % 360
                min_lat = row['min_lat']
                max_lat = row['max_lat']
                
                # Slice the MRMS data cube using our spatial boundary limits
                try:
                    val = ds.sel(
                        latitude=slice(max_lat, min_lat),  # MRMS grids run North to South
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
    # Layer 1: The outer operational CWA outline (Crisp Blue Perimeter)
    outline_layer = pdk.Layer(
        "GeoJsonLayer",
        cwa_layer,
        stroke_width=3,
        get_line_color=[0, 150, 255, 255], 
        get_fill_color=[0, 0, 0, 0],       
        line_width_min_pixels=2,
    )
    
    # Layer 2: The pure AWIPS-style urban footprints
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

if st.button("Refresh & Scan"):
    with st.spinner("Downloading MRMS grids and analyzing regional CWA footprints..."):
        alert_results = scan_data()
        
        # Reset colors back to clean baseline gray first
        for feature in urban_shapes_geojson["features"]:
            feature["properties"]["fill_color"] = [180, 180, 180, 50]
            feature["properties"]["line_color"] = [120, 120, 120, 100]
            
        if alert_results:
            st.error("🚨 THRESHOLDS EXCEEDED WITHIN OPERATIONAL REGIONS:")
            alerted_towns = [key.split(",")[0].strip().upper() for key in alert_results.keys()]
            
            # Loop through the polygon features and turn matching threatened shapes bright red
            for feature in urban_shapes_geojson["features"]:
                feat_name = str(feature["properties"]["name"]).upper()
                if any(town in feat_name for town in alerted_towns):
                    feature["properties"]["fill_color"] = [255, 0, 0, 180]  # Vivid Warning Red
                    feature["properties"]["line_color"] = [150, 0, 0, 255]  # Deep Red Border
            
            map_placeholder.pydeck_chart(render_map(cwa_geojson, urban_shapes_geojson))
            st.json(alert_results)
        else:
            st.success("✅ All systems normal across all 5 operational WFO domains.")
            map_placeholder.pydeck_chart(render_map(cwa_geojson, urban_shapes_geojson))
