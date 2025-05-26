import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import differential_evolution, minimize, fsolve

# Constants and parameters
K_SPRING = 15000  # N/m (spring rate)
C_DAMPER = 1500   # Ns/m (damping coefficient)
T_TOTAL = 2       # s (simulation time)
PHI2_RANGE = np.deg2rad(np.linspace(80, 105, 100))  # Swingarm angle range
T_VALS = np.linspace(0, T_TOTAL, len(PHI2_RANGE))  # Time values

# Target parameters from research papers
A = 0.145  # m/s (constant term for velocity profile)
B = 0.275  # m/s² (linear term coefficient)
C = 0.4    # m/s³ (quadratic term coefficient)
AD_PRIME = 0.115  # m (fixed distance A-D')
DD_PRIME = 0.025  # m (fixed vertical offset)
F_POINT = np.array([-0.04, 0.02])  # m (fixed damper chassis mount point)

class SuspensionOptimizer:
    def __init__(self):
        # Initialize with parameters from the research paper
        self.optimal_params_linear = [0.10000, 0.02612, 0.01539, -0.00596, 0.09897, 0.09315]
        self.optimal_params_quadratic = [0.10008, 0.02016, 0.00463, -0.01529, 0.09035, 0.08014]
        self.params2 = [1,1,1]
        
    def calculate_positions(self, phi2, params):
        """Calculate positions of all linkage points for given swingarm angle"""
        x1, x2, x3, x4, x5, x6 = params
        A = np.array([0, 0])
        D = np.array([AD_PRIME, -DD_PRIME])
        
        # Points along the linkage
        B_prime = A + np.array([x1, 0])
        B = B_prime + x2 * np.array([-np.sin(phi2), np.cos(phi2)])  # Modified for correct orientation
        C = B + x5 * np.array([np.cos(phi2), np.sin(phi2)])
        D_prime = D + np.array([0, DD_PRIME])
        
        # Solve for rocker arm angles using vector closure equations
        # Equations (2) and (3) from the paper
        def closure_eqs(vars):
            phi3, phi4 = vars
            eq1 = x1 * np.cos(phi2) + x5 * np.sin(phi4) + x6 * np.cos(phi3) - x6 * np.cos(phi4) - AD_PRIME
            eq2 = x1 * np.sin(phi2) - x2 * np.cos(phi2) - x5 * np.sin(phi3) - x6 * np.sin(phi4) - DD_PRIME
            return [eq1, eq2]
        
        # Solve nonlinear equations for phi3 and phi4
        try:
            phi3, phi4 = fsolve(closure_eqs, [phi2, phi2])
        except:
            return None, None, None, None
        
        # Calculate point E position (damper mount on rocker)
        E = D_prime + np.array([
            (x6 + x3) * np.cos(phi4) - x4 * np.sin(phi4),
            (x6 + x3) * np.sin(phi4) + x4 * np.cos(phi4)
        ])
        
        return E, phi3, phi4, C

    def desired_velocity_profile(self, t_vals, params2, profile='quadratic'):
        a, b, c = params2
        """Target damper compression velocity profile"""
        if profile == 'linear':
            return -a - b*(t_vals/T_TOTAL)
        elif profile == 'quadratic':
            return -a - b*(t_vals/T_TOTAL) - c*(t_vals/T_TOTAL)**2
        else:
            raise ValueError("Unsupported profile type")

    def evaluate_suspension(self, params, profile='quadratic'):
        """Evaluate suspension performance with given parameters"""
        EF_distances = []
        valid = True
        
        for phi2 in PHI2_RANGE:
            E, _, _, _ = self.calculate_positions(phi2, params)
            if E is None:
                valid = False
                break
            EF_distances.append(np.linalg.norm(E - F_POINT))
        
        if not valid or len(EF_distances) != len(PHI2_RANGE):
            return None, None, None
        
        EF_distances = np.array(EF_distances)
        velocities = np.gradient(EF_distances, T_VALS)
        v_target = self.desired_velocity_profile(T_VALS, profile)
        
        return EF_distances, velocities, v_target

    def objective_function(self, params, profile='quadratic'):
        """Objective function for optimization"""
        EF_distances, velocities, v_target = self.evaluate_suspension(params, profile)
        
        if EF_distances is None:
            return np.inf
        
        # Check Grashof condition for four-bar linkage
        x1, x2, x3, x4, x5, x6 = params
        L1 = x5  # Connecting rod (BC)
        L2 = x6  # Rocker (CD)
        L3 = np.sqrt(AD_PRIME**2 + DD_PRIME**2)  # Fixed link (AD)
        L4 = x1  # Swingarm (AB)
        
        # Grashof condition: shortest + longest <= sum of other two
        lengths = sorted([L1, L2, L3, L4])
        if not (lengths[0] + lengths[3] <= lengths[1] + lengths[2]):
            return np.inf
        
        # MSE between target and actual velocity profiles
        mse = np.mean((velocities - v_target)**2)
        
        # Add penalty for invalid configurations
        if np.any(np.isnan(velocities)) or np.any(np.isinf(velocities)):
            return np.inf
            
        return mse

    def optimize(self, profile='quadratic'):
        """Optimize the suspension geometry"""
        bounds = [
            (0.09, 0.145),   # x1 (AB)
            (-0.015, 0.045),  # x2 (B'B)
            (0.01, 0.045),    # x3 (CE')
            (-0.01, 0.05),    # x4 (E'E)
            (0.06, 0.10),     # x5 (BC)
            (0.06, 0.11),    # x6 (DC)
        ]
        
        # First try with paper's optimal values as initial guess
        if profile == 'linear':
            x0 = self.optimal_params_linear
        else:
            x0 = self.optimal_params_quadratic
            
        res = minimize(self.objective_function, x0, args=(profile,),
                      bounds=bounds, method='L-BFGS-B')
        
        # Then refine with differential evolution
        result = differential_evolution(self.objective_function, bounds, 
                                      args=(profile,), maxiter=100, polish=False)
        
        # Use the better of the two results
        if res.fun < result.fun:
            return res.x, res.fun
        return result.x, result.fun

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
    
    print("Optimizing for quadratic velocity profile...")
    opt_params_quad, _ = optimizer.optimize(profile='quadratic')
    print("Optimized parameters:", opt_params_quad)
    optimizer.plot_results(opt_params_quad, profile='quadratic')
    
    print("\nOptimizing for linear velocity profile...")
    opt_params_lin, _ = optimizer.optimize(profile='linear')
    print("Optimized parameters:", opt_params_lin)
    optimizer.plot_results(opt_params_lin, profile='linear')