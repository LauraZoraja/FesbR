import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import differential_evolution, minimize, fsolve

# Constants and parameters
K_SPRING = 15000  # N/m (spring rate)
C_DAMPER = 1500   # Ns/m (damping coefficient)
T_TOTAL = 2       # s (simulation time)
PHI2_RANGE = np.deg2rad(np.linspace(80, 105, 100))  # Swingarm angle range
T_VALS = np.linspace(0, T_TOTAL, len(PHI2_RANGE))  # Time values

# Target parameters 
A = 0.145  # m/s (constant term for velocity profile)
B = 0.275  # m/s² (linear term coefficient)
C = 0.4    # m/s³ (quadratic term coefficient)
AD_PRIME = 0.115  # m (fixed distance A-D')
DD_PRIME = 0.025  # m (fixed vertical offset)
F_POINT = np.array([-0.04, 0.02])  # m (fixed damper chassis mount point)

class SuspensionOptimizer:
    def __init__(self):
        self.optimal_params_linear = [0.10000, 0.02612, 0.01539, -0.00596, 0.09897, 0.09315]
        self.optimal_params_quadratic = [0.10008, 0.02016, 0.00463, -0.01529, 0.09035, 0.08014]

    def calculate_positions(self, phi2, params, phi_guess=None):
        x1, x2, x3, x4, x5, x6 = params
        A = np.array([0, 0])
        D_PRIME = np.array([0, -AD_PRIME])
        D = D_PRIME + np.array([DD_PRIME, 0])

        B_PRIME = x1 * np.array([np.cos(phi2), -np.sin(phi2)])
        B = B_PRIME + x2 * np.array([np.cos(phi2), -np.sin(phi2)])

        C = B + x5 * np.array([np.cos(phi2), np.sin(phi2)])
        def closure_eqs(vars):
            phi3, phi4 = vars
            eq1 = x1 * np.cos(phi2) + x2 * np.sin(phi2) + x5 * np.cos(phi3) - x6 * np.cos(phi4) - AD_PRIME
            eq2 = x1 * np.sin(phi2) - x2 * np.cos(phi2) - x5 * np.sin(phi3) - x6 * np.sin(phi4) - DD_PRIME
            return [eq1, eq2]

        if phi_guess is None:
            phi_guess = [phi2, phi2]

        try:
            phi_sol, info, ier, msg = fsolve(closure_eqs, phi_guess, full_output=True)
            if ier != 1 or np.any(np.isnan(phi_sol)) or np.any(np.isinf(phi_sol)):
                return None, None, None, None
            phi3, phi4 = phi_sol
        except Exception:
            return None, None, None, None

        E = np.array([
            (x6 + x3) * np.cos(phi4) - x4 * np.sin(phi4) + AD_PRIME,
            (x6 + x3) * np.sin(phi4) + x4 * np.cos(phi4) + DD_PRIME
        ])

        return E, phi3, phi4, C

    def desired_velocity_profile(self, t_vals, profile='quadratic'):
        if profile == 'linear':
            return -A - B * (t_vals / T_TOTAL)
        elif profile == 'quadratic':
            return -A - B * (t_vals / T_TOTAL) - C * (t_vals / T_TOTAL) ** 2
        else:
            raise ValueError("Unsupported profile type")

    def evaluate_suspension(self, params, profile='quadratic'):
        EF_distances = []
        valid = True
        phi_guess = None

        for phi2 in PHI2_RANGE:
            E, phi3, phi4, _ = self.calculate_positions(phi2, params, phi_guess)
            if E is None:
                valid = False
                break
            phi_guess = [phi3, phi4]
            EF_distances.append(np.linalg.norm(E - F_POINT))

        if not valid or len(EF_distances) != len(PHI2_RANGE):
            return None, None, None

        EF_distances = np.array(EF_distances)
        if np.any(np.isnan(EF_distances)) or np.any(np.isinf(EF_distances)):
            return None, None, None

        EF_distances = np.array(EF_distances)
        velocities = np.gradient(EF_distances, T_VALS)
        v_target = self.desired_velocity_profile(T_VALS, profile)

        return EF_distances, velocities, v_target

    def objective_function(self, params, profile='quadratic'):
        x1, x2, x3, x4, x5, x6 = params

        # Enforce physical constraints up front
        if x1 <= 0 or x3 <= 0 or x5 <= 0 or x6 <= 0:
            return 1e6  # large penalty instead of inf

        # Evaluate suspension
        EF_distances, velocities, v_target = self.evaluate_suspension(params, profile)

        # Fallback in case evaluation failed
        if EF_distances is None or velocities is None or v_target is None:
            return 1e6

        # Check for bad numeric values
        if (
            np.any(np.isnan(EF_distances)) or np.any(np.isinf(EF_distances)) or
            np.any(np.isnan(velocities)) or np.any(np.isinf(velocities)) or
            np.any(np.isnan(v_target)) or np.any(np.isinf(v_target))
        ):
            return 1e6

        # Grashof condition
        L1, L2 = x5, x6
        L3 = np.sqrt(AD_PRIME**2 + DD_PRIME**2)
        L4 = x1
        lengths = sorted([L1, L2, L3, L4])
        if lengths[0] + lengths[3] > lengths[1] + lengths[2]:
            return 1e6

        mse = np.mean((velocities - v_target) ** 2)
        return mse

    #except Exception:
        # Catch any unexpected errors safely
        #return 1e6

    def optimize(self, profile='quadratic'):
        bounds = [
            (0.09, 0.145),  # x1
            (-0.015, 0.045),  # x2
            (0.01, 0.045),  # x3
            (-0.01, 0.05),  # x4
            (0.06, 0.10),  # x5
            (0.06, 0.11),  # x6
        ]

        x0 = self.optimal_params_linear if profile == 'linear' else self.optimal_params_quadratic

        res = minimize(self.objective_function, x0, args=(profile,), bounds=bounds, method='L-BFGS-B')
        result = differential_evolution(self.objective_function, bounds, args=(profile,), maxiter=100, polish=False)

        return (res.x, res.fun) if res.fun < result.fun else (result.x, result.fun)


    def plot_results(self, params, profile='quadratic'):
        """Plot the optimization results"""
        EF_distances, velocities, v_target = self.evaluate_suspension(params, profile)
        
        if EF_distances is None:
            print("Invalid parameters - cannot plot")
            return
            
        # Damper velocity comparison
        plt.figure(figsize=(12, 5))
        plt.subplot(1, 2, 1)
        plt.plot(np.rad2deg(PHI2_RANGE), v_target, '--', label='Target velocity', color='orange')
        plt.plot(np.rad2deg(PHI2_RANGE), velocities, label='Optimized velocity', color='blue')
        plt.xlabel("Swingarm angle [deg]")
        plt.ylabel("Damper Velocity [m/s]")
        plt.title(f"Damper Velocity vs Target ({profile.capitalize()} Profile)")
        plt.grid()
        plt.legend()
        
        # Forces calculation
        stroke = EF_distances - EF_distances[0]
        F_spring = K_SPRING * stroke
        F_damper = C_DAMPER * velocities
        F_total = F_spring + F_damper
        
        plt.subplot(1, 2, 2)
        plt.plot(np.rad2deg(PHI2_RANGE), F_total, label='Total Force [N]', color='green')
        plt.plot(np.rad2deg(PHI2_RANGE), F_spring, '--', label='Spring Force [N]', color='red')
        plt.plot(np.rad2deg(PHI2_RANGE), F_damper, '--', label='Damping Force [N]', color='blue')
        plt.xlabel("Swingarm angle [deg]")
        plt.ylabel("Force [N]")
        plt.title("Forces Acting on the Damper")
        plt.grid()
        plt.legend()
        plt.tight_layout()
        plt.show()
        
        # Draw linkage at midpoint position
        self.draw_linkage(params, phi2=np.deg2rad(92.5))

    def draw_linkage(self, params, phi2=np.deg2rad(90)):
        """Draw the four-bar linkage at specified swingarm angle"""
        x1, x2, x3, x4, x5, x6 = params
        A = np.array([0, 0])
        D = np.array([AD_PRIME, -DD_PRIME])
        D_prime = D + np.array([0, DD_PRIME])
        
        # Calculate all points
        B_prime = A + np.array([x1, 0])
        B = B_prime + x2 * np.array([-np.sin(phi2), np.cos(phi2)])
        C = B + x5 * np.array([np.cos(phi2), np.sin(phi2)])
        
        E, phi3, phi4, _ = self.calculate_positions(phi2, params)
        if E is None:
            print("Cannot draw - invalid configuration at this angle")
            return
            
        fig, ax = plt.subplots(figsize=(10, 8))
        
        # Draw all links
        def draw_link(p1, p2, label=None, color='k', linestyle='-'):
            ax.plot([p1[0], p2[0]], [p1[1], p2[1]], linestyle, color=color, linewidth=2)
            if label:
                ax.text((p1[0]+p2[0])/2, (p1[1]+p2[1])/2, label, 
                       fontsize=10, color=color, ha='center', va='center')
        
        draw_link(A, B_prime, 'x1')
        draw_link(B_prime, B, 'x2')
        draw_link(B, C, 'x5')
        draw_link(C, E, 'x3+x4')
        draw_link(D, C, 'x6')
        draw_link(E, F_POINT, 'Damper', color='blue')
        
        # Draw rocker arm
        draw_link(D_prime, C, '', color='gray', linestyle='--')
        draw_link(D_prime, E, '', color='gray', linestyle='--')
        
        # Label points
        for label, pt in zip(['A', "B'", 'B', 'C', 'E', 'D', "D'", 'F'], 
                           [A, B_prime, B, C, E, D, D_prime, F_POINT]):
            ax.text(pt[0], pt[1], label, fontsize=12, ha='center', va='center',
                   bbox=dict(facecolor='white', alpha=0.8, edgecolor='none'))
        
        ax.set_aspect('equal')
        ax.set_title(f'Optimized 4-Bar Rear Suspension (φ={np.rad2deg(phi2):.1f}°)')
        ax.set_xlabel('x [m]')
        ax.set_ylabel('y [m]')
        ax.grid(True)
        plt.tight_layout()
        plt.show()

# Run optimization and analysis
if __name__ == "__main__":
    optimizer = SuspensionOptimizer()
    
    print("\nOptimizing for linear velocity profile...")
    opt_params_lin, _ = optimizer.optimize(profile='linear')
    print("Optimized parameters:", opt_params_lin)
    optimizer.plot_results(opt_params_lin, profile='linear')