import numpy as np
import matplotlib.pyplot as plt

plt.rcParams.update({
    "font.family": "serif",
    "font.size": 10,
    "figure.dpi": 300,
    "savefig.dpi": 300
})

fig, axes = plt.subplots(1, 2, figsize=(8, 3.5))

# 1. Reference Latent Domain
xi = np.linspace(-1, 1, 20)
eta = np.linspace(-1, 1, 20)
XI, ETA = np.meshgrid(xi, eta)

axes[0].plot(XI, ETA, 'k-', alpha=0.3, lw=0.8)
axes[0].plot(XI.T, ETA.T, 'k-', alpha=0.3, lw=0.8)
axes[0].set_title(r"Latent Reference Domain $\Omega_l$")
axes[0].set_xlabel(r"$\xi$")
axes[0].set_ylabel(r"$\eta$")
axes[0].set_aspect('equal')
axes[0].grid(False)

# 2. Deformed Physical Domain (Diffeomorphic mapping)
R_grid = np.sqrt(XI**2 + ETA**2) / np.sqrt(2)
THETA_grid = np.arctan2(ETA, XI)
r_mesh = 0.5 + 0.1 * np.cos(3 * THETA_grid) + 0.05 * np.sin(2 * THETA_grid)

X_phys = R_grid * r_mesh * np.cos(THETA_grid)
Y_phys = R_grid * r_mesh * np.sin(THETA_grid)

axes[1].plot(X_phys, Y_phys, 'b-', alpha=0.4, lw=0.8)
axes[1].plot(X_phys.T, Y_phys.T, 'b-', alpha=0.4, lw=0.8)
axes[1].set_title(r"Physical Domain $\Omega_p$ ($\det(J) > 0$)")
axes[1].set_xlabel(r"$x$")
axes[1].set_ylabel(r"$y$")
axes[1].set_aspect('equal')
axes[1].grid(False)

plt.tight_layout()
plt.savefig("figure1_diffeomorphism.pdf")
plt.savefig("figure1_diffeomorphism.png")
print("[OK] Figura 1 (vector PDF & 300 DPI PNG) generata con successo.")
