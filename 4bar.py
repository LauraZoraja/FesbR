import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import differential_evolution

# parameters
k_spring = 15000  # N/m
c_damper = 1500   # Ns/m
t_total = 2       # s

# motion range 
phi2_range = np.deg2rad(np.linspace(80, 105, 100))
t_vals = np.linspace(0, t_total, len(phi2_range))

# kinematics
def calculate_positions(phi2, params):
    x1, x2, x3, x4, x5, x6 = params
    A = np.array([0, 0])
    B_ = A + np.array([x1, 0])
    B  = B_ + x2 * np.array([np.cos(phi2), np.sin(phi2)])
    C  = B  + x5 * np.array([np.cos(phi2), np.sin(phi2)])
    E_ = C  + x3 * np.array([np.cos(phi2 + np.pi/2), np.sin(phi2 + np.pi/2)])
    E  = E_ + x4 * np.array([np.cos(phi2 + np.pi), np.sin(phi2 + np.pi)])
    return E

# target velocity profile
def desired_velocity_profile(t_vals, profile='quadratic'):
    if profile == 'linear':
        return 0.05 + 0.1 * (t_vals / t_vals[-1])
    elif profile == 'quadratic':
        return 0.05 + 0.1 * (t_vals / t_vals[-1])**2
    else:
        raise ValueError("Unsupported profile type")

# use a linear profile
v_target = desired_velocity_profile(t_vals, profile='linear')

# linkage fit to target velocity
def linkage_velocity_fit(all_vars):
    x1, x2, x3, x4, x5, x6, Fx, Fy = all_vars
    params = [x1, x2, x3, x4, x5, x6]
    F_point = np.array([Fx, Fy])

    try:
        EF_distances = np.array([np.linalg.norm(calculate_positions(phi, params) - F_point) for phi in phi2_range])
    except:
        return np.inf

    if np.any(np.isnan(EF_distances)) or np.any(np.isinf(EF_distances)):
        return np.inf

    velocities = np.gradient(EF_distances, t_vals)
    mse = np.mean((velocities - v_target)**2)
    return mse

# bounds
bounds = [
    (0.09, 0.145),   # x1 - frame width
    (-0.015, 0.09), # x2 - pivot offset
    (0.01, 0.045),   # x3 - swingarm
    (-0.01, 0.09),   # x4 - rocker arm length
    (0.06, 0.10),    # x5 - extension to damper
    (0.06, 0.11),    # x6 - rear chassis link
    (-0.08, 0.0),    # Fx
    (0.0, 0.025),     # Fy
]

# run optimization 
print("Optimizing geometry and F location for damper performance (linear target)...")
result = differential_evolution(linkage_velocity_fit, bounds, seed=42, maxiter=300, disp=True)
opt_vars = result.x
opt_params = opt_vars[:6]
opt_F = opt_vars[6:]
print("\n✅ Optimized geometry:", opt_params)
print("✅ Optimized F point:", opt_F)

# simulate optimized system 
EF_opt = np.array([np.linalg.norm(calculate_positions(phi, opt_params) - opt_F) for phi in phi2_range])
v_opt = np.gradient(EF_opt, t_vals)

plt.figure(figsize=(10, 5))
plt.plot(t_vals, v_target, '--', label='Target velocity (linear)', color='orange')
plt.plot(t_vals, v_opt, label='Optimized velocity', color='blue')
plt.xlabel("Time [s]")
plt.ylabel("Damper Velocity [m/s]")
plt.title("Final Optimized Damper Velocity vs Target (Linear)")
plt.grid()
plt.legend()
plt.tight_layout()
plt.show()

# forces
stroke = EF_opt - EF_opt[0]
F_spring = k_spring * stroke
F_damper = c_damper * v_opt
F_total = F_spring + F_damper

plt.figure(figsize=(10, 5))
plt.plot(t_vals, F_total, label='Total Damper Force [N]', color='green')
plt.plot(t_vals, F_spring, '--', label='Spring Force [N]', color='red')
plt.plot(t_vals, F_damper, '--', label='Damping Force [N]', color='blue')
plt.xlabel("Time [s]")
plt.ylabel("Force [N]")
plt.title("Forces Acting on the Damper Over Time")
plt.grid()
plt.legend()
plt.tight_layout()
plt.show()

# draw 4-bar
def draw_four_bar_linkage(phi2=np.deg2rad(95), params=None, F=None):
    x1, x2, x3, x4, x5, x6 = params
    A = np.array([0, 0])
    D = np.array([0.115, 0.0])
    B_ = A + np.array([x1, 0])
    B  = B_ + x2 * np.array([np.cos(phi2), np.sin(phi2)])
    C  = B  + x5 * np.array([np.cos(phi2), np.sin(phi2)])
    E_ = C  + x3 * np.array([np.cos(phi2 + np.pi/2), np.sin(phi2 + np.pi/2)])
    E  = E_ + x4 * np.array([np.cos(phi2 + np.pi), np.sin(phi2 + np.pi)])

    fig, ax = plt.subplots(figsize=(8, 6))
    def draw(p1, p2, label=None, color='k'):
        ax.plot([p1[0], p2[0]], [p1[1], p2[1]], 'o-', color=color)
        if label:
            ax.text(*(p1 + p2)/2, label, fontsize=9, color=color)

    draw(A, B_, 'x1')
    draw(B_, B, 'x2')
    draw(B, C, 'x5')
    draw(C, E_, 'x3')
    draw(E_, E, 'x4')
    draw(D, C, 'x6')
    draw(E, F, 'Damper', color='blue')

    # labels
    for label, pt in zip(['A', "B'", 'B', 'C', "E'", 'E', 'D', 'F'], [A, B_, B, C, E_, E, D, F]):
        ax.text(pt[0], pt[1], label, fontsize=10)

    ax.set_aspect('equal')
    ax.set_title('Optimized 4-Bar Rear Suspension (Uni-Trak)')
    ax.set_xlabel('x [m]')
    ax.set_ylabel('y [m]')
    ax.grid(True)
    plt.tight_layout()
    plt.show()

draw_four_bar_linkage(phi2=np.deg2rad(95), params=opt_params, F=opt_F)
