import streamlit as st
from model import run_simulation, plot_profile

st.title("Europa Ocean Model")

st.sidebar.header("Simulation Parameters")
spatial_resolution = st.sidebar.slider(
    "Spatial resolution (m)", 50, 10000, value=1000, step=50,
    help="Vertical grid spacing of the model in meters." )
days = st.sidebar.slider(
    "Days", 10, 2000, value=1000, step=10,
    help="Total length of the simulation in days.")
time_steps_per_week = st.sidebar.slider(
    "Time steps per week", 1, 21, value=7,
    help="Temporal resolution of the solver.")
convergence_threshold = st.sidebar.number_input(
    "Convergence threshold", min_value=1e-9, max_value=1e-3, value=1e-7, format="%e",
    help="Stop when temperature changes fall below this value.")

st.sidebar.subheader("Planet Parameters")
geo_flux = st.sidebar.slider(
    "Geothermal heat flux (°C/s)", 0.0, 5.0, value=1.0,
    help="Heat input from Europa's interior at the base of the ocean.")
love_number = st.sidebar.slider(
    "Love number", 0.0, 0.3, value=0.15,
    help="Controls tidal deformation and associated heating.")
tidal_coefficient = st.sidebar.slider(
    "Tidal heating coefficient", 0.0, 0.1, value=0.01,
    help="Scales how strongly tides heat the water column.")
depth = st.sidebar.slider(
    "Hydrosphere depth (m)", 10000.0, 200000.0, value=127000.0,
    help="Total depth of the ice and ocean combined.")

if st.button("Run Simulation"):
    placeholder = st.empty()
    progress_bar = st.progress(0.0)

    def draw(temp, depth_pts, ice_extent, step, total_steps):
        fig = plot_profile(temp, depth_pts, ice_extent, show=False)
        placeholder.pyplot(fig)
        progress_bar.progress(step / total_steps)

    temp, depth_pts, ice_extent = run_simulation(
        spatial_resolution=spatial_resolution,
        days=int(days),
        time_steps_per_week=int(time_steps_per_week),
        convergence_threshold=convergence_threshold,
        geothermal_heat_flux=geo_flux,
        love_number=love_number,
        tidal_heating_coefficient=tidal_coefficient,
        depth=depth,
        callback=draw,
    )
    progress_bar.empty()
    placeholder.pyplot(plot_profile(temp, depth_pts, ice_extent, show=False))
