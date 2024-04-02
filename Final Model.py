import numpy as np
import matplotlib.pyplot as plt

# Model Params To Change
spatial_resolution = 1000 # In meters
days = 1000  
time_steps_per_week = 7  # Temporal resolution
convergence_threshold = 0.0000001  # Higher takes longer to stop 

# Planet Parameters
geothermal_heat_flux = 1  #(in °C/s)
love_number = 0.15  # Higher is more rigid 
tidal_heating_coefficient = 0.01  #
g = 1.315 
albedo = 0.64
solar_radiation_flux = 50  #  W/m^2
depth = 127000.0  # Depth of hydrosphere

# Physical constants
kappa = 1e-6  # Thermal diffusivity of water (in m^2/s)
latent_heat_fusion = 334000  # Latent heat of fusion for ice-water transition (J/kg)
sigma = 5.67e-8  # Stefan-Boltzmann constant (W/m^2/K^4)

# Model Stuff
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
    mixing_coeff = np.where(Ri < Ri_crit, background_kappa * (1 - Ri / Ri_crit) ** 2, background_kappa * 1e-3)
    return mixing_coeff

# Initial conditions
depth_points = np.linspace(0, depth, num_points)  
initial_temp = np.where(depth_points <= 25000, 0 + -160 * (1 - depth_points[:num_points] / 25000), 40)  # Initial temperature profile

# Model
print("Simulation Progress:")
temperature = initial_temp.copy()
percent_complete = 0

# Save previous profile for equilibrium function
prev_temperature = temperature.copy()

for t in range(1, total_time_steps):
    temperature[-1] += geothermal_heat_flux * dt  # Geothermal heating at base

    # Radiative heat loss from surface to space
    surface_flux = sigma * (temperature[0] + 273.15) ** 4  # Stefan-Boltzmann law finds radiative loss
    surface_flux -= (solar_radiation_flux * albedo)  # Solar radiation
    temperature[0] -= surface_flux * dt  # Update profile surface

    # Compute temperature gradient
    dTdz = np.gradient(temperature, dx)

    # Calculate tidal heating 
    tidal_heating_flux = tidal_heating_coefficient * (1 / love_number) * (dTdz ** 2)

    # Update temperature with tidal heating
    temperature += tidal_heating_flux * dt

    # Compute heat diffusion
    d2Tdx2 = np.gradient(dTdz, dx)
    
    # Compute heat diffusion in ice layer
    ice_indices = np.where(temperature <= 0)[0]  # Find indices where temperature is at or below freezing
    if len(ice_indices) > 0:
        # Calculate heat diffusion only for the ice layer
        d2Tdx2_ice = np.gradient(dTdz[ice_indices], dx)
        
        # Apply temperature-dependent mixing specifically for ice layer
        mixing_coeff_ice = get_mixing_coefficient(dTdz[ice_indices], kappa)
        temperature[ice_indices] += mixing_coeff_ice * dt * d2Tdx2_ice

    # Apply temperature-dependent mixing
    mixing_coeff = get_mixing_coefficient(dTdz, kappa)
    temperature += mixing_coeff * dt * d2Tdx2

    # Implement convection
    if t > 1:
        # Check for stable stratification
        if np.any(dTdz < 0):
            # Find and resolve unstable sections of water column
            unstable_indices = np.where((dTdz < 0) & (temperature > 0))[0]
            for idx in unstable_indices:
                if idx > 0:
                    temperature[idx:] = temperature[idx]  

    # Calculate melting/freezing at ice-ocean interface
    ice_surface_temp = temperature[-1]
    ice_melting_rate = latent_heat_fusion / (depth * latent_heat_fusion) * (
                ice_surface_temp - 0) * dt  # Rate of melting (m/s)
    temperature[-1] = max(0, temperature[-1] - ice_melting_rate)  # Update temperature at ice-ocean interface
    
    # Update deepest 0°C water
    highest_ice_extent = max(highest_ice_extent, depth_points[ice_indices[-1]])

    # Check for equilibrium
    change_in_temperature = np.max(np.abs(temperature - prev_temperature))
    if change_in_temperature < convergence_threshold:
        print("Equilibrium reached. Stopping simulation.")
        break  # Exit the simulation loop

    prev_temperature = temperature.copy()

    # Print progress
    progress = (t / total_time_steps) * 100
    if progress >= percent_complete:
        print(f"{percent_complete:.0f}% complete")
        percent_complete += 1
        
# Plot final water column
plt.plot(temperature, depth_points, label='Water Temperature')
plt.xlabel('Temperature (°C)')
plt.ylabel('Depth (m)')
plt.title('Water Column at Finish')

# Annotate ice depth
plt.axhline(y=highest_ice_extent, color='g', linestyle='--', label='Ice Depth')
plt.text(temperature[0], highest_ice_extent, f'Ice Depth: {highest_ice_extent:.0f} m', verticalalignment='bottom')

#Plot stuff
plt.gca().invert_yaxis() 
plt.legend()
plt.grid(True)
plt.show()
