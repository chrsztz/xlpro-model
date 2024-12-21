import numpy as np
import matplotlib.pyplot as plt
import torch
from matplotlib.animation import FuncAnimation
import matplotlib.animation as animation

# Rocket parameters
diameter = 3.7  # meters
cd = 0.5  # Drag coefficient
Sref = 0.25 * np.pi * diameter ** 2  # Rocket reference area in m²
DryMassRatio1 = 0.05  # Stage 1 dry mass ratio
DryMassRatio2 = 0.03  # Stage 2 dry mass ratio
Mass_Dry1 = 20379  # Stage 1 dry mass in kg
Mass_Dry2 = 4243  # Stage 2 dry mass in kg
Rb1 = 2844  # Specific impulse of Stage 1 engine in m/s
Rb2 = 3413  # Specific impulse of Stage 2 engine in m/s
thrust1 = 903000 * 9  # Stage 1 thrust in N
thrust2 = 934100  # Stage 2 thrust in N
MassFairing = 1900  # Fairing mass in kg
MassPayload = 6000  # Payload mass in kg
H_FairingSeperate = 100000  # Altitude for fairing separation in meters
# Adjustments
t_kick = 15  # Time for pitch maneuver kick in seconds
gamma_kick = 4.377168  # Pitch angle in degrees
FillRatio1 = 0.896834  # Fuel fill ratio for Stage 1
FillRatio2 = 0.870498  # Fuel fill ratio for Stage 2
CoeThrust1 = 1.000000  # Thrust coefficient for Stage 1
CoeThrust2 = 0.999996  # Thrust coefficient for Stage 2

# Initial values
V_init = 0  # Initial velocity in m/s
x_init = 0  # Initial horizontal position in meters
H_init = 0  # Initial altitude in meters
gamma0 = 90  # Initial pitch angle in degrees
gamma_dot_init = 0  # Initial pitch angle velocity in rad/s

# Constants
miu_earth = 3.99E+14  # Gravitational parameter for Earth
R_earth = 6378 * 1000  # Average Earth radius in meters
g0 = 9.8067  # Gravitational acceleration at sea level in m/s²
rou0 = 1.225  # Atmospheric density at sea level in kg/m³
rho_H0 = 7194  # Scale height for atmospheric density in meters
t_step = 1  # Time step in seconds

# Initial conditions
thrust1 *= CoeThrust1
thrust2 *= CoeThrust2
fuel_flow1 = thrust1 / Rb1
fuel_flow2 = thrust2 / Rb2
Mass1 = Mass_Dry1 / DryMassRatio1
Mass2 = Mass_Dry2 / DryMassRatio2
Mass1 -= Mass1 * (1 - DryMassRatio1) * (1 - FillRatio1)
Mass2 -= Mass2 * (1 - DryMassRatio2) * (1 - FillRatio2)
mass_init = Mass1 + Mass2 + MassFairing + MassPayload

# Simulation arrays
t = [0]
v = [V_init]
gamma = [gamma0 * np.pi / 180]
x = [x_init]
H = [H_init]
rou = [rou0 * np.exp(-H[0] / rho_H0)]
Drag = [0.5 * rou[0] * v[0] ** 2 * Sref * cd]
g_loc = [g0 / (1 + H[0] / R_earth) ** 2]
mass = [mass_init]
accel = [(thrust1 - Drag[0]) / mass[0] - g_loc[0] * np.sin(gamma[0])]
gamma_dot = [gamma_dot_init * np.pi / 180]
x_dot = [v[0] * np.cos(gamma[0])]
H_dot = [v[0] * np.sin(gamma[0])]

# Flag for stage separation
FlagStageSeperate = 0

# Dynamic simulation loop
for i in range(1, 1000000):
    # Time step
    t.append(t[-1] + t_step)

    # Calculate velocity
    v.append(v[-1] + accel[-1] * t_step)

    # Calculate pitch angle
    if t[i] == t_kick:
        gamma.append(gamma[-1] + gamma_dot[-1] * t_step - gamma_kick * np.pi / 180)
    else:
        gamma.append(gamma[-1] + gamma_dot[-1] * t_step)

    # Switch engine
    if FlagStageSeperate == 0:
        thrust = thrust1
        fuel_flow = fuel_flow1
    else:
        thrust = thrust2
        fuel_flow = fuel_flow2

    # Calculate vertical velocity
    H_dot.append(v[i] * np.sin(gamma[i]))

    # Calculate vertical position
    H.append(H[-1] + H_dot[i] * t_step)

    # Calculate horizontal velocity
    x_dot.append(v[i] * np.cos(gamma[i]))

    # Calculate horizontal position
    x.append(x[-1] + x_dot[i] / (1 + H[i] / R_earth) * t_step)

    # Update atmospheric density
    rou.append(rou0 * np.exp(-H[i] / rho_H0))

    # Update drag force
    Drag.append(0.5 * rou[i] * v[i] ** 2 * Sref * cd)

    # Update local gravitational acceleration
    g_loc.append(g0 / (1 + H[i] / R_earth) ** 2)

    # Update rocket mass
    mass.append(mass[-1] - fuel_flow * t_step)

    # Stage separation
    if mass[i] < mass_init - Mass1 * (1 - DryMassRatio1) and FlagStageSeperate == 0:
        mass[i] -= Mass_Dry1
        FlagStageSeperate = 1

    # Stop simulation if rocket reaches horizontal flight or runs out of fuel
    if gamma[i] < 0 or mass[i] < Mass_Dry2 + MassPayload:
        break

    # Update acceleration
    accel.append((thrust - Drag[i]) / mass[i] - g_loc[i] * np.sin(gamma[i]))

    # Update pitch rate
    gamma_dot.append(-(g_loc[i] / v[i]) * np.cos(gamma[i]))

# Check and ensure lengths are equal
if len(t) != len(accel):
    accel.append(accel[-1])  # Append last value of accel if lengths don't match

# Plot the results
fig, axes = plt.subplots(3, 2, figsize=(15, 12))

# Angle vs Time
axes[0, 0].plot(t, np.array(gamma) * 180 / np.pi, 'b-', linewidth=2)
axes[0, 0].set_title("Trajectory Angle (Gamma) vs Time")
axes[0, 0].set_xlabel("Time (s)")
axes[0, 0].set_ylabel("Angle (degrees)")
axes[0, 0].grid(True)

# Height vs Time
axes[0, 1].plot(t, H, 'b-', linewidth=2)
axes[0, 1].set_title("Height vs Time")
axes[0, 1].set_xlabel("Time (s)")
axes[0, 1].set_ylabel("Height (m)")
axes[0, 1].grid(True)

# Speed vs Time
axes[1, 0].plot(t, v, 'b-', linewidth=2)
axes[1, 0].set_title("Speed vs Time")
axes[1, 0].set_xlabel("Time (s)")
axes[1, 0].set_ylabel("Speed (m/s)")
axes[1, 0].grid(True)

# Mass vs Time
axes[1, 1].plot(t, mass, 'b-', linewidth=2)
axes[1, 1].set_title("Mass vs Time")
axes[1, 1].set_xlabel("Time (s)")
axes[1, 1].set_ylabel("Mass (kg)")
axes[1, 1].grid(True)

# Range vs Height
axes[2, 0].plot(np.array(x) / 1000, np.array(H) / 1000, 'b-', linewidth=2)
axes[2, 0].set_title("Trajectory (Range vs Height)")
axes[2, 0].set_xlabel("Range (km)")
axes[2, 0].set_ylabel("Height (km)")
axes[2, 0].grid(True)

# Acceleration vs Time
axes[2, 1].plot(t, accel, 'b-', linewidth=2)
axes[2, 1].set_title("Acceleration vs Time")
axes[2, 1].set_xlabel("Time (s)")
axes[2, 1].set_ylabel("Acceleration (m/s²)")
axes[2, 1].grid(True)

plt.tight_layout()

# Helper function to generate and save animation for each graph
def create_animation(x_data, y_data, title, xlabel, ylabel, filename):
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.set_xlim(0, max(t))
    ax.set_ylim(min(y_data), max(y_data))

    line, = ax.plot([], [], lw=2)

    def init():
        line.set_data([], [])
        return line,

    def animate(i):
        line.set_data(x_data[:i], y_data[:i])
        return line,

    ani = FuncAnimation(fig, animate, frames=len(x_data), init_func=init, blit=True)

    # Save the animation to MP4
    ani.save(filename, writer='ffmpeg', fps=30)

    plt.close(fig)  # Close the figure to prevent it from displaying on the screen

# 1. Trajectory Angle vs Time
create_animation(t, np.array(gamma) * 180 / np.pi, "Trajectory Angle (Gamma) vs Time", "Time (s)", "Angle (degrees)", "trajectory_angle.mp4")

# 2. Height vs Time
create_animation(t, H, "Height vs Time", "Time (s)", "Height (m)", "height_time.mp4")

# 3. Speed vs Time
create_animation(t, v, "Speed vs Time", "Time (s)", "Speed (m/s)", "speed_time.mp4")

# 4. Mass vs Time
create_animation(t, mass, "Mass vs Time", "Time (s)", "Mass (kg)", "mass_time.mp4")

# 5. Range vs Height
create_animation(np.array(x) / 1000, np.array(H) / 1000, "Trajectory (Range vs Height)", "Range (km)", "Height (km)", "range_height.mp4")

# 6. Acceleration vs Time
create_animation(t, accel, "Acceleration vs Time", "Time (s)", "Acceleration (m/s²)", "acceleration_time.mp4")
