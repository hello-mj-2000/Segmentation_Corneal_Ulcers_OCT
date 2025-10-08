import json
import streamlit as st, plotly.io as pio

@st.cache_resource
def load_fig():
    with open("images/3d_heatmap.json","r") as f:
        return pio.from_json(f.read())

st.title("3D Lesion Topography")
fig = load_fig()
st.plotly_chart(fig, use_container_width=True)




