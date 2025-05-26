import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import fsolve

# Constants (from paper)
omega2 = 2.0  # rad/s
phi2_initial = np.deg2rad(80)  # initial angle in radians
phi2_final = np.deg2rad(105)   # final angle in radians
T_total = (phi2_final - phi2_initial) / omega2  # total time from eq (9)
PHI2_RANGE = np.linspace(phi2_initial, phi2_final, 100)
T_VALS = np.linspace(0, T_total, len(PHI2_RANGE))

# Parameters (quadratic case)
params_quad = [0.10008, 0.02016, 0.00463, 0.01529, 0.09035, 0.08014]
params_lin = [0.10000, 0.02612, 0.01539, 0.00596, 0.09897, 0.09315]
AD_prime = 0.115  # m
DD_prime = 0.025  # m
F = np.array([-0.04, 0.02])  # coordinates of point F
F_prime = np.array([-0.04, 0])

# Spring and damper properties
K_SPRING = 15000  # N/m
C_DAMPER = 1500   # Ns/m

def calculate_positions(phi2, params, phi_guess=None):
    x1, x2, x3, x4, x5, x6 = params
    
    A = np.array([0, 0])
    
    D_prime = np.array([AD_prime, 0])
    
    # Point D is directly above D' by DD' in y
    D = D_prime + np.array([0, DD_prime])
    
    # Point B' is x2 distance from B at a right angle
    B_prime = A + x1 * np.array([np.cos(phi2), np.sin(phi2)])
    B = B_prime + x2 * np.array([np.sin(phi2), -np.cos(phi2)])
    
    
    # Equations (2) and (3) from the paper
    def closure_eqs(vars):
        phi3, phi4 = vars
        # Equation (2)
        eq1 = x1 * np.cos(phi2) + x5 * np.sin(phi4) + x6 * np.cos(phi3) - x6 * np.cos(phi4) - AD_prime
        # Equation (3)
        eq2 = x1 * np.sin(phi2) - x2 * np.cos(phi2) - x5 * np.sin(phi3) - x6 * np.sin(phi4) - DD_prime
        return [eq1, eq2]
    
    # Initial guess for angles
    if phi_guess is None:
        phi_guess = [phi2, phi2]
    
    # Solve the nonlinear equations
    try:
        phi_sol = fsolve(closure_eqs, phi_guess)
        phi3, phi4 = phi_sol
    except:
        return None, None, None, None, None
    
    C = B + x5 * np.array([np.cos(phi3), np.sin(phi3)])
    
    E_prime = C + x3 * np.array([np.cos(phi4), np.sin(phi4)])
    
    E = E_prime + x4 * np.array([np.sin(phi4), np.cos(phi4)])
    
    return E, phi3, phi4, C, E_prime

def desired_velocity_profile(profile='linear'):
    a, b, c = 0.145, 0.275, 0.4
    
    if profile == 'linear':
        return -a - b * T_VALS
    else:
        return -a - b * T_VALS - c * T_VALS**2

def evaluate(params):
    EF_distances = []
    phi_guess = None
    best_phi2 = None
    best_phi3 = None
    best_phi4 = None
    best_E = None
    
    for phi2 in PHI2_RANGE:
        E, phi3, phi4, C, E_prime = calculate_positions(phi2, params, phi_guess)
        if E is None:
            return None, None, None, None
        phi_guess = [phi3, phi4]
        EF_distances.append(np.linalg.norm(E - F))
        
        # Store the values from the last iteration (which will be for phi2_final)
        best_phi2 = phi2
        best_phi3 = phi3
        best_phi4 = phi4
        best_E = E
    
    # Print the best optimized values
    print("Optimized values at final position:")
    print(f"phi2: {np.rad2deg(best_phi2):.4f}°")
    print(f"phi3: {np.rad2deg(best_phi3):.4f}°")
    print(f"phi4: {np.rad2deg(best_phi4):.4f}°")
    print(f"E: {best_E}")
    
    EF_distances = np.array(EF_distances)
    velocities = np.gradient(EF_distances, T_VALS)
    v_target = desired_velocity_profile('quadratic')
    
    return EF_distances, velocities, v_target, phi_guess

def plot_all(params):
    EF_distances, velocities, v_target, _ = evaluate(params)

    if EF_distances is None:
        print("Error in evaluating suspension. Aborting.")
        return

    # Calculate forces
    stroke = EF_distances - EF_distances[0]
    F_spring = K_SPRING * stroke
    F_damper = C_DAMPER * velocities
    F_total = F_spring + F_damper

    # Plot 1: Velocity profile
    plt.figure(figsize=(16, 5))
    plt.subplot(1, 3, 1)
    plt.plot(np.rad2deg(PHI2_RANGE), v_target, '--', label='Target Velocity', color='orange')
    plt.plot(np.rad2deg(PHI2_RANGE), velocities, label='Actual Velocity', color='blue')
    plt.xlabel("Swingarm Angle [deg]")
    plt.ylabel("Velocity [m/s]")
    plt.title("Damper Velocity vs Target")
    plt.grid(True)
    plt.legend()

    # Plot 2: Damping force
    plt.subplot(1, 3, 2)
    plt.plot(np.rad2deg(PHI2_RANGE), F_damper, color='blue', label='Damping Force')
    plt.xlabel("Swingarm Angle [deg]")
    plt.ylabel("Damping Force [N]")
    plt.title("Damping Force vs Angle")
    plt.grid(True)
    plt.legend()

    # Plot 3: All forces
    plt.subplot(1, 3, 3)
    plt.plot(np.rad2deg(PHI2_RANGE), F_total, label='Total Force', color='green')
    plt.plot(np.rad2deg(PHI2_RANGE), F_spring, '--', label='Spring Force', color='red')
    plt.plot(np.rad2deg(PHI2_RANGE), F_damper, '--', label='Damping Force', color='blue')
    plt.xlabel("Swingarm Angle [deg]")
    plt.ylabel("Force [N]")
    plt.title("Forces Acting on the Damper")
    plt.grid(True)
    plt.legend()
    
    plt.tight_layout()
    plt.show()

    draw_linkage(params, phi2=np.deg2rad(87))

def draw_linkage(params, phi2=np.deg2rad(87)):
    E, phi3, phi4, C, E_prime = calculate_positions(phi2, params)
    if E is None:
        print("Invalid linkage configuration")
        return

    # Get all points from the kinematic calculations
    x1, x2, x3, x4, x5, x6 = params
    A = np.array([0, 0])
    D_prime = np.array([AD_prime, 0])
    D = D_prime + np.array([0, DD_prime])
    B_prime = A + x1 * np.array([np.cos(phi2), np.sin(phi2)])
    B = B_prime + x2 * np.array([np.sin(phi2), -np.cos(phi2)])
    
    # Create figure
    fig, ax = plt.subplots(figsize=(10, 8))

    def draw_link(p1, p2, label=None, color='k', linestyle='-', offset=None):
        ax.plot([p1[0], p2[0]], [p1[1], p2[1]], linestyle, color=color, linewidth=2)
        if label:
            pos = ((p1[0]+p2[0])/2, (p1[1]+p2[1])/2)
            if offset:
                pos = (pos[0] + offset[0], pos[1] + offset[1])
            ax.text(pos[0], pos[1], label,
                    fontsize=10, color=color, ha='center', va='center')

    # Draw all links using the calculated points
    draw_link(A, B_prime, 'x1')
    draw_link(B_prime, B, 'x2')
    draw_link(B, C, 'x5')
    draw_link(C, E_prime, 'x3')
    draw_link(E_prime, E, 'x4')
    draw_link(D, C, 'x6')
    draw_link(E, F, 'Damper', color='blue')

    # Rocker arms (dashed lines)
    draw_link(D, B, '', color='gray', linestyle='--')

    # Label points
    for label, pt in zip(['A', "B'", 'B', 'C', "E'", 'E', 'D', "D'", 'F', "F'"],
                         [A, B_prime, B, C, E_prime, E, D, D_prime, F, F_prime]):
        ax.text(pt[0], pt[1], label, fontsize=12, ha='center', va='center',
                bbox=dict(facecolor='white', alpha=0.8, edgecolor='none'))

    ax.set_aspect('equal')
    ax.set_title(f'4-Bar Suspension Geometry \nφ2 = {np.rad2deg(phi2):.1f}°')
    ax.set_xlabel('x [m]')
    ax.set_ylabel('y [m]')
    ax.grid(True)
    plt.tight_layout()
    plt.show()

# Run the analysis
plot_all(params_lin)