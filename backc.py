import numpy as np
import matplotlib.pyplot as plt

# Konstante
swingl = 500  # Swing arm duljina
ovjesp = (0, 0)  # Točka prednjeg ovjesa
kotac_radijus = 300  # Radijus kružnice po kojoj se giba kotac
swing_kut = np.radians(30)  # Početni kut swing arma
t = 0.01  # dt
pivot_rangex = np.linspace(-20, 20, 5) # Range swing arm pivot pointa)
pivot_rangey = np.linspace(-10, 10, 5)
prigusivac_kutr = np.linspace(np.radians(30), np.radians(60), 5)  # Range kutova prigušivača
prigusivacl = 300

# Položaj prednjeg kotača
def zadnji_kotac(pivot_point, kut):
    x = pivot_point[0] + swingl * np.cos(kut)
    y = pivot_point[1] + swingl * np.sin(kut)
    return x, y

# Položaj prigušivača
def prigusivac(pivot_point, kotac_polozaj, kut_prigusivac):
    # Swing arm vektor
    swing_arm_vektor = np.array(kotac_polozaj) - np.array(pivot_point)
    
    # Prigušivač vektor
    rotation_matrix = np.array([
        [np.cos(kut_prigusivac), -np.sin(kut_prigusivac)],
        [np.sin(kut_prigusivac), np.cos(kut_prigusivac)]
    ])
    prigusivac_smjer = rotation_matrix @ swing_arm_vektor
    
    # Normalizacija vektora prigušivača
    prigusivac_smjer = prigusivac_smjer / np.linalg.norm(prigusivac_smjer) * prigusivacl
    
    # Položaj prigušivača
    prigusivac_pocetak = pivot_point
    prigusivac_kraj = pivot_point + prigusivac_smjer
    return prigusivac_pocetak, prigusivac_kraj

polozaj_prigusivaca = []

for pivot_x in pivot_rangex:
    for pivot_y in pivot_rangey:
        pivot_point = (pivot_x, pivot_y)
        for kut_prigusivac in prigusivac_kutr:
            zadnji_kotac_polozaj = zadnji_kotac(pivot_point, swing_kut)
            prigusivac_pocetak, prigusivac_kraj = prigusivac(pivot_point, zadnji_kotac_polozaj, kut_prigusivac)
            polozaj_prigusivaca.append((pivot_point, zadnji_kotac_polozaj, prigusivac_pocetak,prigusivac_kraj, kut_prigusivac))

plt.figure(figsize=(10, 6))
for i, (pivot_point, zadnji_kotac_polozaj, prigusivac_pocetak, prigusivac_kraj, kut_prigusivac) in enumerate(polozaj_prigusivaca):
    plt.plot([pivot_point[0], zadnji_kotac_polozaj[0]], [pivot_point[1], zadnji_kotac_polozaj[1]], 'b-', label="Swing Arm" if i == 0 else "")
    
    plt.scatter(zadnji_kotac_polozaj[0], zadnji_kotac_polozaj[1], color='green', label="Rear Wheel" if i == 0 else "")
    
    plt.plot([prigusivac_pocetak[0], prigusivac_kraj[0]], [prigusivac_pocetak[1], prigusivac_kraj[1]], 'r', label=f"Damper ({np.degrees(kut_prigusivac):.1f}°)" if i == 0 else "")


plt.xlabel("X (mm)")
plt.ylabel("Y (mm)")

plt.legend()
plt.grid(True)
plt.axis('equal')
plt.show()