import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint

# constants
H0 = 1.0  # normalized Hubble constant

def friedmann_equation(a, t, Om, Or, Ol):
    """
    friedmann equation: da/dt = a * H(a)
    H(a) = H0 * sqrt(Or*a^-4 + Om*a^-3 + Ol)
    """
    # avoid division by zero
    if a <= 1e-6: return 0
    # Hubble parameter H(a) derivation
    h_a = H0 * np.sqrt(Or * a**-4 + Om * a**-3 + Ol)
    return a * h_a

def plot_cosmology_dynamics():
    # 1. plot Density Scaling
    a_vals = np.logspace(-4, 0, 100)
    rho_m = 1.0 * a_vals**-3
    rho_r = 1.0 * a_vals**-4
    rho_l = np.ones_like(a_vals)

    plt.figure(figsize=(10, 6))
    plt.loglog(a_vals, rho_m, label=r'Matter ($\rho \propto a^{-3}$)')
    plt.loglog(a_vals, rho_r, label=r'Radiation ($\rho \propto a^{-4}$)')
    plt.loglog(a_vals, rho_l, label=r'Dark Energy ($\rho = \text{const}$)')
    plt.xlabel(r'Scale Factor $a$')
    plt.ylabel(r'Density $\rho$')
    plt.title('Evolution of Energy Densities')
    plt.legend()
    plt.grid(True, which='both', linestyle='--')
    plt.savefig('density_evolution.png')
    plt.show()

    # 2. plot scale factor evolution a(t)
    t_vals = np.linspace(0.01, 3, 100)
    a0 = 0.01 # initial condition (near Big Bang)
    
    # solve ODEs for different compositions
    # matter-only (Om=1, Or=0, Ol=0)
    a_m = odeint(friedmann_equation, a0, t_vals, args=(1.0, 0.0, 0.0))
    # radiation-only (Om=0, Or=1, Ol=0)
    a_r = odeint(friedmann_equation, a0, t_vals, args=(0.0, 1.0, 0.0))
    # lambda-only (Om=0, Or=0, Ol=1)
    a_l = odeint(friedmann_equation, a0, t_vals, args=(0.0, 0.0, 1.0))

    plt.figure(figsize=(10, 6))
    plt.plot(t_vals, a_m, label='Matter-only ($a \propto t^{2/3}$)')
    plt.plot(t_vals, a_r, label='Radiation-only ($a \propto t^{1/2}$)')
    plt.plot(t_vals, a_l, label=r'$\Lambda$-only ($a \propto e^{H_{\Lambda}t}$)')
    plt.xlabel(r'Time $t$')
    plt.ylabel(r'Scale Factor $a(t)$')
    plt.title('Scale Factor Evolution')
    plt.legend()
    plt.grid(True)
    plt.savefig('scale_factor_evolution.png')
    plt.show()

if __name__ == "__main__":
    plot_cosmology_dynamics()
    print("Plots generated successfully: density_evolution.png and scale_factor_evolution.png")