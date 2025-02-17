import numpy as np
import matplotlib.pyplot as plt

# Constants
swingl = 500  # Swing arm length in mm
ovjesp = np.array([0, 0])  # Front suspension point (x, y)
kotac_radijus = 300  # Radius of the rear wheel path in mm
swing_kut = np.radians(30)  # Initial angle of the swing arm in radians
pivot_rangex = np.linspace(-20, 20, 1)  # Range of swing arm pivot points (x-coordinate)
pivot_rangey = np.linspace(-10, 10, 1)  # Range of swing arm pivot points (y-coordinate)
prigusivac_kutr = np.linspace(np.radians(30), np.radians(60), 5)  # Range of damper angles
prigusivacl = 300  # Length of the damper line for visualization

# Triangle sides (a, b, c) ranges
rangeA = np.linspace(100, 300, 1)  # Range for side a of the triangle
rangeB = np.linspace(100, 300, 1)  # Range for side b of the triangle
rangeC = np.linspace(100, 300, 1)  # Range for side c of the triangle

# Function to calculate the position of the rear wheel
def zadnji_kotac(pivot_point, kut):
    x = pivot_point[0] + swingl * np.cos(kut)
    y = pivot_point[1] + swingl * np.sin(kut)
    return np.array([x, y])

# Function to calculate the triangle vertices and damper endpoint
def prigusivac(pivot_point, kotac_polozaj, kut_prigusivac, a, b, c):
    # Point C is exactly in the middle of the line between pivot point and rear wheel
    C = (kotac_polozaj + pivot_point) / 2

    # Calculate point B using the triangle sides a, b, c
    # Using the law of cosines to find the angle at point C
    cos_kut_C = (a**2 + b**2 - c**2) / (2 * a * b)
    kut_C = np.arccos(np.clip(cos_kut_C, -1, 1))

    # Direction vector from C to B
    smjer = np.array([np.cos(kut_C), np.sin(kut_C)])
    B = C + a * smjer

    # Damper connects to point B
    # Calculate the damper direction vector by rotating the swing arm vector by the damper angle
    swing_arm_vektor = kotac_polozaj - pivot_point
    rotation_matrix = np.array([
        [np.cos(kut_prigusivac), -np.sin(kut_prigusivac)],
        [np.sin(kut_prigusivac), np.cos(kut_prigusivac)]
    ])
    prigusivac_smjer = rotation_matrix @ swing_arm_vektor

    # Normalize the damper direction and scale it to the damper length
    prigusivac_smjer = prigusivac_smjer / np.linalg.norm(prigusivac_smjer) * prigusivacl

    # Calculate the damper endpoints
    prigusivac_pocetak = B
    prigusivac_kraj = B + prigusivac_smjer

    return C, B, prigusivac_pocetak, prigusivac_kraj

# Initialize lists to store results
polozaj_prigusivaca = []

# Loop over swing arm pivot points, damper angles, and triangle sides
for pivot_x in pivot_rangex:
    for pivot_y in pivot_rangey:
        pivot_point = np.array([pivot_x, pivot_y])  # Pivot point moves in both x and y directions
        zadnji_kotac_polozaj = zadnji_kotac(pivot_point, swing_kut)  # Rear wheel position
        for kut_prigusivac in prigusivac_kutr:
            for a in rangeA:
                for b in rangeB:
                    for c in rangeC:
                        # Check if the triangle inequality is satisfied
                        if a + b > c and a + c > b and b + c > a:
                            C, B, prigusivac_pocetak, prigusivac_kraj = prigusivac(pivot_point, zadnji_kotac_polozaj, kut_prigusivac, a, b, c)
                            polozaj_prigusivaca.append((pivot_point, zadnji_kotac_polozaj, C, B, prigusivac_pocetak, prigusivac_kraj, kut_prigusivac, a, b, c))

# Plot the results
plt.figure(figsize=(10, 6))
for i, (pivot_point, zadnji_kotac_polozaj, C, B, prigusivac_pocetak, prigusivac_kraj, kut_prigusivac, a, b, c) in enumerate(polozaj_prigusivaca):
    # Plot swing arm
    plt.plot([pivot_point[0], zadnji_kotac_polozaj[0]], [pivot_point[1], zadnji_kotac_polozaj[1]], 'b-', label="Swing Arm" if i == 0 else "")
    
    # Plot rear wheel
    plt.scatter(zadnji_kotac_polozaj[0], zadnji_kotac_polozaj[1], color='green', label="Rear Wheel" if i == 0 else "")
    
    # Plot triangle
    plt.plot([ovjesp[0], C[0], B[0], ovjesp[0]], 
             [ovjesp[1], C[1], B[1], ovjesp[1]], 'm-', label="Triangle" if i == 0 else "")
    
    # Plot damper
    plt.plot([prigusivac_pocetak[0], prigusivac_kraj[0]], [prigusivac_pocetak[1], prigusivac_kraj[1]], 'r--', label=f"Damper ({np.degrees(kut_prigusivac):.1f}°)" if i == 0 else "")

# Plot front suspension point
plt.scatter(ovjesp[0], ovjesp[1], color='black', label="Front Suspension Point")

# Add labels and legend
plt.xlabel("X (mm)")
plt.ylabel("Y (mm)")
plt.title("Optimal Pivot Point, Triangle, and Damper Endpoint")
plt.legend()
plt.grid(True)
plt.axis('equal')
plt.show()