import numpy as np
import matplotlib.pyplot as plt
from math import cos, sin, radians, degrees, sqrt, atan2

# Constants
FRONT_SUSPENSION_POINT = np.array([0, 0])
REAR_WHEEL_POSITION = np.array([1.5, 0])

def calculate_swing_arm_angle(pivot_point, rear_wheel_pos):
    swing_arm_vector = rear_wheel_pos - pivot_point
    swing_arm_angle = atan2(swing_arm_vector[1], swing_arm_vector[0])
    return swing_arm_angle

def triangle_sides(point_A, point_B, point_C):
    side_BC = np.linalg.norm(point_B - point_C)
    side_AC = np.linalg.norm(point_A - point_C)
    side_AB = np.linalg.norm(point_A - point_B)
    return side_BC, side_AC, side_AB

def damper_endpoint_and_angle_fixed_length(point_A, swing_arm_angle, damper_length):
    theta = swing_arm_angle + radians(90)
    damper_endpoint = point_A + damper_length * np.array([cos(theta), sin(theta)])
    return damper_endpoint, degrees(theta)

def damper_endpoint_and_length_fixed_angle(point_A, swing_arm_angle, damper_angle):
    theta = radians(damper_angle)
    if sin(theta) == 0:
        raise ValueError("Damper angle results in division by zero.")
    y_target = 0  # Target y-coordinate for the damper endpoint
    y_A = point_A[1]
    dy = y_target - y_A
    damper_length = dy / sin(theta)
    damper_endpoint = point_A + damper_length * np.array([cos(theta), sin(theta)])
    return damper_endpoint, damper_length

def simulate_suspension(pivot_point, variation="fixed_length"):
    if pivot_point[1] >= FRONT_SUSPENSION_POINT[1]:
        raise ValueError("Pivot point must be below the front suspension point.")
    
    swing_arm_angle = calculate_swing_arm_angle(pivot_point, REAR_WHEEL_POSITION)
    point_C = (pivot_point + REAR_WHEEL_POSITION) / 2
    point_B = np.array([point_C[0] + 0.1, point_C[1] + 0.2])
    point_A = np.array([point_C[0] - 0.1, point_C[1] + 0.2])
    
    side_BC, side_AC, side_AB = triangle_sides(point_A, point_B, point_C)
    
    if variation == "fixed_length":
        damper_length = 0.3
        damper_endpoint, damper_angle = damper_endpoint_and_angle_fixed_length(point_A, swing_arm_angle, damper_length)
    elif variation == "fixed_angle":
        damper_angle = 170
        damper_endpoint, damper_length = damper_endpoint_and_length_fixed_angle(point_A, swing_arm_angle, damper_angle)
    else:
        raise ValueError("Invalid variation. Use 'fixed_length' or 'fixed_angle'.")
    
    return swing_arm_angle, pivot_point, side_BC, side_AC, side_AB, damper_endpoint, damper_angle, damper_length, REAR_WHEEL_POSITION, point_A, point_B, point_C

def plot_suspension(pivot_point, rear_wheel_pos, point_A, point_B, point_C, damper_endpoint, variation, title_suffix=""):
    plt.figure()
    plt.plot([pivot_point[0], rear_wheel_pos[0]], [pivot_point[1], rear_wheel_pos[1]], 'b-', label="Swing Arm")
    plt.scatter(*pivot_point, color='green', label="Pivot Point")
    plt.scatter(*rear_wheel_pos, color='blue', label="Rear Wheel")
    plt.scatter(*FRONT_SUSPENSION_POINT, color='black', label="Front Suspension Point")
    plt.plot([point_A[0], point_B[0]], [point_A[1], point_B[1]], 'r-', label="Triangle Side AB")
    plt.plot([point_B[0], point_C[0]], [point_B[1], point_C[1]], 'r-', label="Triangle Side BC")
    plt.plot([point_C[0], point_A[0]], [point_C[1], point_A[1]], 'r-', label="Triangle Side CA")
    plt.plot([point_A[0], damper_endpoint[0]], [point_A[1], damper_endpoint[1]], 'm-', label="Damper")
    plt.scatter(*point_A, color='red', label="Point A")
    plt.scatter(*point_B, color='orange', label="Point B")
    plt.scatter(*point_C, color='purple', label="Point C")
    plt.scatter(*damper_endpoint, color='magenta', label="Damper Endpoint")
    plt.legend()
    plt.xlabel("X Position")
    plt.ylabel("Y Position")
    plt.title(f"Motorcycle Rear Suspension System ({variation}) {title_suffix}")
    plt.grid(True)
    plt.axis('equal')
    plt.show()

# Generate pivot points
pivot_points = [np.array([0.4 + 0.1 * i, -0.5 - 0.1 * i]) for i in range(5)]

# Simulate suspension for each variation
for variation in ["fixed_length", "fixed_angle"]:
    print(f"\nVariation: {variation}")
    max_damper_value = -float('inf')
    min_damper_value = float('inf')
    max_damper_pivot = None
    min_damper_pivot = None

    for pivot_point in pivot_points:
        try:
            swing_arm_angle, pivot_point, side_BC, side_AC, side_AB, damper_endpoint, damper_angle, damper_length, rear_wheel_pos, point_A, point_B, point_C = simulate_suspension(pivot_point, variation)
            
            print(f"Pivot Point: {pivot_point}")
            if variation == "fixed_length":
                print(f"Damper Angle (degrees): {damper_angle}")
                damper_value = damper_angle
            elif variation == "fixed_angle":
                print(f"Damper Length: {damper_length}")
                damper_value = damper_length
            print("-" * 40)
            
            if damper_value > max_damper_value:
                max_damper_value = damper_value
                max_damper_pivot = pivot_point
                max_damper_config = (swing_arm_angle, pivot_point, side_BC, side_AC, side_AB, damper_endpoint, damper_angle, damper_length, rear_wheel_pos, point_A, point_B, point_C)
            if damper_value < min_damper_value:
                min_damper_value = damper_value
                min_damper_pivot = pivot_point
                min_damper_config = (swing_arm_angle, pivot_point, side_BC, side_AC, side_AB, damper_endpoint, damper_angle, damper_length, rear_wheel_pos, point_A, point_B, point_C)
        except ValueError as e:
            print(f"Error for pivot point {pivot_point}: {e}")

    if variation == "fixed_length":
        print(f"Pivot point with the largest damper angle ({max_damper_value}°): {max_damper_pivot}")
        print(f"Pivot point with the smallest damper angle ({min_damper_value}°): {min_damper_pivot}")
    elif variation == "fixed_angle":
        print(f"Pivot point with the largest damper length ({max_damper_value}): {max_damper_pivot}")
        print(f"Pivot point with the smallest damper length ({min_damper_value}): {min_damper_pivot}")

    # Plot the configuration with the largest damper angle or longest damper length
    if max_damper_pivot is not None:
        swing_arm_angle, pivot_point, side_BC, side_AC, side_AB, damper_endpoint, damper_angle, damper_length, rear_wheel_pos, point_A, point_B, point_C = max_damper_config
        plot_suspension(pivot_point, rear_wheel_pos, point_A, point_B, point_C, damper_endpoint, variation, title_suffix="(Max Damper Value)")

    # Plot the configuration with the smallest damper angle or shortest damper length
    if min_damper_pivot is not None:
        swing_arm_angle, pivot_point, side_BC, side_AC, side_AB, damper_endpoint, damper_angle, damper_length, rear_wheel_pos, point_A, point_B, point_C = min_damper_config
        plot_suspension(pivot_point, rear_wheel_pos, point_A, point_B, point_C, damper_endpoint, variation, title_suffix="(Min Damper Value)")