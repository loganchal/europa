import numpy as np
import matplotlib.pyplot as plt


def run_simulation(
    spatial_resolution=1000,
    days=1000,
    time_steps_per_week=7,
    convergence_threshold=1e-7,
    geothermal_heat_flux=1,
    love_number=0.15,
    tidal_heating_coefficient=0.01,
    depth=127000.0,
    ice_depth=25000.0,
    record=False,
    frame_interval=50,
):
    """Run the Europa ocean model simulation.

    Parameters match the variables from the original script. The function
    returns the temperature profile, corresponding depth points, the
    highest extent of ice reached during the run and a list of frames for
    animation when ``record`` is ``True``.
    """
    # Physical constants
    kappa = 1e-6  # Thermal diffusivity of water (in m^2/s)
    latent_heat_fusion = 334000  # Latent heat of fusion for ice-water transition (J/kg)
    sigma = 5.67e-8  # Stefan-Boltzmann constant (W/m^2/K^4)
    g = 1.315  # Gravity (m/s^2)
    albedo = 0.64
    solar_radiation_flux = 50

    num_points = int(depth / spatial_resolution) + 1
    days_per_week = 7
    total_time_steps = days * time_steps_per_week
    dt = 1 / (time_steps_per_week * days_per_week)
    dx = depth / (num_points - 1)
    highest_ice_extent = 0

    def get_mixing_coefficient(dTdz, background_kappa):
        alpha = 2e-5  # Thermal expansion coefficient of water (1/°C)
        Ri = g / (alpha * background_kappa) * np.abs(dTdz)
        Ri_crit = 0.65  # Critical Richardson number
        return np.where(Ri < Ri_crit, background_kappa * (1 - Ri / Ri_crit) ** 2, background_kappa * 1e-3)

    # Initial conditions
    depth_points = np.linspace(0, depth, num_points)
    initial_temp = np.where(
        depth_points <= ice_depth,
        0 + -160 * (1 - depth_points[:num_points] / ice_depth),
        40,
    )

    temperature = initial_temp.copy()
    prev_temperature = temperature.copy()
    percent_complete = 0
    frames = []

    print("Simulation Progress:")
    for t in range(1, total_time_steps):
        temperature[-1] += geothermal_heat_flux * dt  # Geothermal heating at base

        # Radiative heat loss from surface to space
        surface_flux = sigma * (temperature[0] + 273.15) ** 4
        surface_flux -= (solar_radiation_flux * albedo)
        temperature[0] -= surface_flux * dt

        # Compute temperature gradient
        dTdz = np.gradient(temperature, dx)

        # Calculate tidal heating
        tidal_heating_flux = tidal_heating_coefficient * (1 / love_number) * (dTdz ** 2)

        # Update temperature with tidal heating
        temperature += tidal_heating_flux * dt

        # Compute heat diffusion
        d2Tdx2 = np.gradient(dTdz, dx)

        # Compute heat diffusion in ice layer
        ice_indices = np.where(temperature <= 0)[0]
        if len(ice_indices) > 0:
            d2Tdx2_ice = np.gradient(dTdz[ice_indices], dx)
            mixing_coeff_ice = get_mixing_coefficient(dTdz[ice_indices], kappa)
            temperature[ice_indices] += mixing_coeff_ice * dt * d2Tdx2_ice

        # Apply temperature-dependent mixing
        mixing_coeff = get_mixing_coefficient(dTdz, kappa)
        temperature += mixing_coeff * dt * d2Tdx2

        # Implement convection for unstable water columns
        if t > 1 and np.any(dTdz < 0):
            unstable_indices = np.where((dTdz < 0) & (temperature > 0))[0]
            for idx in unstable_indices:
                if idx > 0:
                    temperature[idx:] = temperature[idx]

        # Calculate melting/freezing at ice-ocean interface
        ice_surface_temp = temperature[-1]
        ice_melting_rate = latent_heat_fusion / (depth * latent_heat_fusion) * (
            ice_surface_temp - 0) * dt
        temperature[-1] = max(0, temperature[-1] - ice_melting_rate)

        # Update deepest 0°C water
        if len(ice_indices) > 0:
            highest_ice_extent = max(highest_ice_extent, depth_points[ice_indices[-1]])

        # Check for equilibrium
        change_in_temperature = np.max(np.abs(temperature - prev_temperature))
        if change_in_temperature < convergence_threshold:
            print("Equilibrium reached. Stopping simulation.")
            break

        prev_temperature = temperature.copy()

        progress = (t / total_time_steps) * 100
        if progress >= percent_complete:
            print(f"{percent_complete:.0f}% complete")
            percent_complete += 1

        if record and t % frame_interval == 0:
            frames.append(temperature.copy())
    if record:
        frames.append(temperature.copy())

    return temperature, depth_points, highest_ice_extent, frames


def plot_profile(temperature, depth_points, highest_ice_extent):
    """Plot the final water column profile."""
    fig, ax = plt.subplots()
    ax.plot(temperature, depth_points, label="Water Temperature")
    ax.set_xlabel("Temperature (°C)")
    ax.set_ylabel("Depth (m)")
    ax.set_title("Water Column at Finish")
    ax.axhline(y=highest_ice_extent, color="g", linestyle="--", label="Ice Depth")
    ax.text(temperature[0], highest_ice_extent,
            f"Ice Depth: {highest_ice_extent:.0f} m",
            verticalalignment="bottom")
    ax.invert_yaxis()
    ax.legend()
    ax.grid(True)
    return fig


if __name__ == "__main__":
    temp, depth_pts, ice_extent, _ = run_simulation()
    fig = plot_profile(temp, depth_pts, ice_extent)
    plt.show()
