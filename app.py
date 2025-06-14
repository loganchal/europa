import time
import matplotlib.pyplot as plt
import streamlit as st
from model import run_simulation, plot_profile

st.title("Europa Ocean Model")

st.sidebar.header("Simulation Parameters")
spatial_resolution = st.sidebar.slider(
    "Spatial resolution (m)", min_value=100, max_value=5000, value=1000, step=100
)
days = st.sidebar.slider("Days", min_value=10, max_value=5000, value=1000)
time_steps_per_week = st.sidebar.slider(
    "Time steps per week", min_value=1, max_value=20, value=7
)
convergence_threshold = st.sidebar.number_input(
    "Convergence threshold", value=1e-7, format="%e"
)

st.sidebar.subheader("Planet Parameters")
geo_flux = st.sidebar.slider(
    "Geothermal heat flux (W/m^2)", min_value=0.0, max_value=5.0, value=1.0
)
love_number = st.sidebar.slider("Love number", min_value=0.0, max_value=1.0, value=0.15)
tidal_coefficient = st.sidebar.slider(
    "Tidal heating coefficient", min_value=0.0, max_value=0.05, value=0.01
)
depth = st.sidebar.slider(
    "Hydrosphere depth (m)", min_value=1000.0, max_value=200000.0, value=127000.0, step=1000.0
)
ice_depth = st.sidebar.slider(
    "Initial ice layer depth (m)", min_value=0.0, max_value=50000.0, value=25000.0, step=1000.0
)

if st.button("Run Simulation"):
    temp, depth_pts, ice_extent, frames = run_simulation(
        spatial_resolution=spatial_resolution,
        days=int(days),
        time_steps_per_week=int(time_steps_per_week),
        convergence_threshold=convergence_threshold,
        geothermal_heat_flux=geo_flux,
        love_number=love_number,
        tidal_heating_coefficient=tidal_coefficient,
        depth=depth,
        ice_depth=ice_depth,
        record=True,
        frame_interval=10,
    )

    placeholder = st.empty()
    for frame in frames:
        fig = plot_profile(frame, depth_pts, ice_extent)
        placeholder.pyplot(fig)
        plt.close(fig)
        time.sleep(0.1)

    final_fig = plot_profile(temp, depth_pts, ice_extent)
    st.pyplot(final_fig)
