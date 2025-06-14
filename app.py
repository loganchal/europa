import streamlit as st
from model import run_simulation, plot_profile

st.title("Europa Ocean Model")

st.sidebar.header("Simulation Parameters")
spatial_resolution = st.sidebar.number_input("Spatial resolution (m)", value=1000)
days = st.sidebar.number_input("Days", value=1000)
time_steps_per_week = st.sidebar.number_input("Time steps per week", value=7)
convergence_threshold = st.sidebar.number_input("Convergence threshold", value=1e-7, format="%e")

st.sidebar.subheader("Planet Parameters")
geo_flux = st.sidebar.number_input("Geothermal heat flux (°C/s)", value=1.0)
love_number = st.sidebar.number_input("Love number", value=0.15)
tidal_coefficient = st.sidebar.number_input("Tidal heating coefficient", value=0.01)
g = st.sidebar.number_input("Gravity (m/s^2)", value=1.315)
albedo = st.sidebar.number_input("Albedo", value=0.64)
solar_flux = st.sidebar.number_input("Solar radiation flux (W/m^2)", value=50)
depth = st.sidebar.number_input("Hydrosphere depth (m)", value=127000.0)

if st.button("Run Simulation"):
    temp, depth_pts, ice_extent = run_simulation(
        spatial_resolution=spatial_resolution,
        days=int(days),
        time_steps_per_week=int(time_steps_per_week),
        convergence_threshold=convergence_threshold,
        geothermal_heat_flux=geo_flux,
        love_number=love_number,
        tidal_heating_coefficient=tidal_coefficient,
        g=g,
        albedo=albedo,
        solar_radiation_flux=solar_flux,
        depth=depth,
    )
    st.pyplot(plot_profile(temp, depth_pts, ice_extent))
