import streamlit as st
import pandas as pd

# 1. Page Configuration
st.set_page_config(page_title="Urban FF - NGP", layout="wide")

st.title("Urban Flash Flood Decision Support (NGP")
st.write("Monitoring real-time MRMS data for urban flash flood potential.")

# 2. Placeholder for our Map
st.map(pd.DataFrame({'lat': [47.9253], 'lon': [-97.0329]})) # Starting with Grand Forks, ND

# 3. Sidebar for Logic Toggles
st.sidebar.header("Alert Logic")
use_persistence = st.sidebar.checkbox("Use 6-min Rain Rate Persistence", value=True)
threshold_qpe = st.sidebar.slider("1-hr QPE Threshold (in)", 0.0, 3.0, 1.0)

# 4. Status Display
st.subheader("Current Urban Alert Status")
st.info("System is waiting for live MRMS data ingestion.")
