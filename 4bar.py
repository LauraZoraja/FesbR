import numpy as np
import matplotlib.pyplot as plt

# === Parametri sustava ===
k_spring = 15000  # N/m
c_damper = 1500   # Ns/m
t_total = 2       # trajanje simulacije [s]
A = np.array([0, 0])
D = np.array([0.115, 0.0])
F = np.array([-0.04, 0.02])  # fiksna točka na šasiji

# === Geometrijski parametri 4-bar mehanizma (Uni-Trak) ===
x1, x2, x3, x4, x5, x6 = 0.1, 0.02, 0.015, -0.01, 0.09, 0.08
params = [x1, x2, x3, x4, x5, x6]

# === Kinematika zakretanja vilice ===
phi2_range = np.deg2rad(np.linspace(80, 105, 100))
t_vals = np.linspace(0, t_total, len(phi2_range))

# === Simulacija gibanja i sila ===
EF_distances, EF_velocities, forces = [], [], []
prev_EF, prev_time = None, None

for phi2, t in zip(phi2_range, t_vals):
    B_ = A + [x1, 0]
    B  = B_ + [x2 * np.cos(phi2), x2 * np.sin(phi2)]
    C  = B  + [x5 * np.cos(phi2), x5 * np.sin(phi2)]
    E_ = C  + [x3 * np.cos(phi2 + np.pi / 2), x3 * np.sin(phi2 + np.pi / 2)]
    E  = E_ + [x4 * np.cos(phi2 + np.pi), x4 * np.sin(phi2 + np.pi)]
    
    EF = np.linalg.norm(E - F)
    EF_distances.append(EF)

    v = (EF - prev_EF) / (t - prev_time) if prev_EF is not None else 0
    EF_velocities.append(v)

    x_compression = EF - EF_distances[0]
    F_total = k_spring * x_compression + c_damper * v
    forces.append(F_total)

    prev_EF, prev_time = EF, t

# === Prikaz rezultata ===
phi_deg = np.rad2deg(phi2_range)
fig, ax = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

ax[0].plot(phi_deg, EF_distances, label='Duljina EF (opruge)', color='green')
ax[0].set_ylabel("Duljina [m]")
ax[0].legend()
ax[0].grid()

ax[1].plot(phi_deg, forces, label='Sila u opruzi + prigušivaču', color='red')
ax[1].set_xlabel("Kut stražnje vilice [°]")
ax[1].set_ylabel("Sila [N]")
ax[1].legend()
ax[1].grid()

fig.suptitle("Kompresija i sila u opruzi/prigušivaču")
plt.tight_layout()
plt.show()

# === Skica 4-bar mehanizma ===
def draw_four_bar_linkage(phi2_angle=np.deg2rad(95), params=params):
    x1, x2, x3, x4, x5, x6 = params
    A = np.array([0, 0])
    D = np.array([0.115, 0.0])
    F = np.array([-0.04, 0.02])

    B_ = A + [x1, 0]
    B  = B_ + [x2 * np.cos(phi2_angle), x2 * np.sin(phi2_angle)]
    C  = B  + [x5 * np.cos(phi2_angle), x5 * np.sin(phi2_angle)]
    E_ = C  + [x3 * np.cos(phi2_angle + np.pi/2), x3 * np.sin(phi2_angle + np.pi/2)]
    E  = E_ + [x4 * np.cos(phi2_angle + np.pi), x4 * np.sin(phi2_angle + np.pi)]

    fig, ax = plt.subplots(figsize=(8, 6))
    def draw_link(p1, p2, label=None, color='k'):
        ax.plot([p1[0], p2[0]], [p1[1], p2[1]], 'o-', color=color)
        if label:
            mid = (p1 + p2) / 2
            ax.text(mid[0], mid[1], label, fontsize=9, color=color)

    draw_link(A, B_, "x1")
    draw_link(B_, B, "x2")
    draw_link(B, C, "x5")
    draw_link(C, E_, "x3")
    draw_link(E_, E, "x4")
    draw_link(D, C, "x6")
    draw_link(E, F, "prigušivač", color='blue')

    # Točke i oznake
    ax.plot(*F, 'ro', label="Točka F (šasija)")
    ax.plot(*E, 'bo', label="Točka E (ovjes)")
    for label, pt in zip(['A', "B'", 'B', 'C', "E'", 'E', 'D', 'F'],
                         [A, B_, B, C, E_, E, D, F]):
        ax.text(pt[0], pt[1], label, fontsize=10)

    ax.set_aspect('equal')
    ax.set_title('Skica 4-bar ovjesnog mehanizma (Uni-Trak)')
    ax.set_xlabel('x [m]')
    ax.set_ylabel('y [m]')
    ax.legend()
    ax.grid(True)
    plt.tight_layout()
    plt.show()

# Prikaz skice
draw_four_bar_linkage()