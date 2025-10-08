import json
import plotly.io as pio
import streamlit as st

st.title("3D Lesion Topography")

with open("images/3d_heatmap.json", "r") as f:
    fig = pio.from_json(f.read())

st.plotly_chart(fig, use_container_width=True)




