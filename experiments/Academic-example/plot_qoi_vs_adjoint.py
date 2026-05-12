"""
Energy balance adjoint diagnostic plot
=======================================
Diagnostic 3-panel figure (|G_i|, ||z_i||, |G_i|*||z_i||) for the index-1 port-Hamiltonian DAE example.
Plotting style: Latin Modern Roman font, black and white lines.
"""

import os
for _var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
             "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_var, "1")

import numpy as np
import scipy.linalg as la
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ==============================================================================
# Inline style 
# ==============================================================================
plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Latin Modern Roman', 'DejaVu Serif', 'serif'],
    'mathtext.fontset': 'cm',
    'font.size': 12,
    'axes.linewidth': 0.8,
    'axes.labelsize': 13,
    'axes.titlesize': 13,
    'axes.spines.top': False,
    'axes.spines.right': False,
    'xtick.labelsize': 11,
    'ytick.labelsize': 11,
    'xtick.direction': 'in',
    'ytick.direction': 'in',
    'xtick.major.width': 0.6,
    'ytick.major.width': 0.6,
    'xtick.minor.width': 0.4,
    'ytick.minor.width': 0.4,
    'xtick.major.size': 5,
    'ytick.major.size': 5,
    'xtick.minor.size': 3,
    'ytick.minor.size': 3,
    'xtick.major.pad': 5,
    'ytick.major.pad': 5,
    'axes.grid': False,
    'legend.fontsize': 10,
    'legend.frameon': True,
    'legend.framealpha': 0.92,
    'legend.edgecolor': '#cccccc',
    'legend.fancybox': False,
    'legend.borderpad': 0.5,
    'legend.handlelength': 2.2,
    'figure.dpi': 150,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.08,
})

_W = {
    'black': '#000000',
}

def add_grid(ax, which='major'):
    ax.grid(True, which='major', color='#cccccc', lw=0.3, alpha=0.5)
    if which == 'both':
        ax.grid(True, which='minor', color='#e0e0e0', lw=0.2, alpha=0.3)


# ==============================================================================
# Index-1 port-Hamiltonian DAE (reduced system)
# ==============================================================================
class Index1_PH_DAE:
    def __init__(self, E, J, R, Q, B):
        A = (J - R) @ Q
        self.r = np.linalg.matrix_rank(E)
        self.E11 = E[:self.r, :self.r]

        A11 = A[:self.r, :self.r]
        A12 = A[:self.r, self.r:]
        A21 = A[self.r:, :self.r]
        A22 = A[self.r:, self.r:]
        A22i = la.inv(A22)

        self.S = -(A11 - A12 @ A22i @ A21)
        self.S_tilde = 0.5 * (self.S + self.S.T)
        self.F = B[:self.r] - A12 @ A22i @ B[self.r:]

        # Keep full system matrices for reconstruction
        self.Q = Q
        self.B = B
        self.M_x1 = -A22i @ A21
        self.M_u  = -A22i @ B[self.r:]

    def hamiltonian(self, x):
        return 0.5 * x @ self.E11 @ x

    def grad_hamiltonian(self, x):
        return self.E11 @ x


# ==============================================================================
# Primal solver  (dG(0))
# ==============================================================================
def solve_primal(sys, grid, x0, u):
    x = np.zeros((len(grid), sys.r))
    x[0] = x0
    for i in range(len(grid) - 1):
        dt = grid[i + 1] - grid[i]
        tmid = grid[i] + 0.5 * dt
        x[i + 1] = la.solve(sys.E11 + dt * sys.S,
                             sys.E11 @ x[i] + dt * sys.F @ u(tmid))
    return x


# ==============================================================================
# Energy balance residuals G_i  
# ==============================================================================
def compute_G(sys, grid, x, u):
    M = len(grid) - 1
    G = np.zeros(M)
    for i in range(M):
        dt = grid[i + 1] - grid[i]
        x1 = x[i + 1]
        tmid = grid[i] + 0.5 * dt
        G[i] = (sys.hamiltonian(x[i + 1]) - sys.hamiltonian(x[i])
                + dt * (x1 @ sys.S_tilde @ x1 - u(tmid) @ (sys.F.T @ x1)))
    return G


# ==============================================================================
# Adjoint solve  (backward pass)
# ==============================================================================
def solve_adjoint(sys, grid, x, u):
    M = len(grid) - 1
    G = compute_G(sys, grid, x, u)
    z = np.zeros((M, sys.r))

    for i in reversed(range(M)):
        dt = grid[i + 1] - grid[i]
        tmid = grid[i] + 0.5 * dt
        x1 = x[i + 1]

        # Derivative of G[i] w.r.t. x[i+1]
        rhs = 2 * G[i] * (
            sys.grad_hamiltonian(x1)
            + 2 * dt * sys.S_tilde @ x1
            - dt * (sys.F @ u(tmid))
        )

        # Coupling from interval i+1
        if i < M - 1:
            rhs -= 2 * G[i + 1] * sys.grad_hamiltonian(x1)
            rhs += sys.E11 @ z[i + 1]

        z[i] = la.solve(sys.E11.T + dt * sys.S.T, rhs)

    return z, G


# ==============================================================================
# Main diagnostic plot
# ==============================================================================
def main():
    # --- system matrices ---
    E = np.diag([1.0, 1.0, 0.0])
    J = np.array([[0, 1, -1], [-1, 0, 0], [1, 0, 0]], dtype=float)
    R = np.diag([0.5, 0.5, 0.1])
    Q = np.eye(3)
    B = np.array([[1.0], [0.0], [0.0]])
    x0 = np.array([1.0, 0.0])

    sys_obj = Index1_PH_DAE(E, J, R, Q, B)

    def u(t):
        return np.array([np.sin(2 * np.pi * t)]) if t < 0.5 else np.array([0.0])

    # --- grid ---
    grid = np.linspace(0, 1.0, 160)

    # --- solves ---
    x = solve_primal(sys_obj, grid, x0, u)
    z, G = solve_adjoint(sys_obj, grid, x, u)

    # --- diagnostics ---
    t_mid = 0.5 * (grid[:-1] + grid[1:])
    G_abs = np.abs(G)
    z_norm = np.linalg.norm(z, axis=1)
    contribution = G_abs * z_norm

    # ==============================================================================
    # Plot: 3 panels
    # ==============================================================================
    fig, axes = plt.subplots(1, 3, figsize=(8.5, 2.6), sharex=True)

    # Panel 1 — energy violation |G_i|
    axes[0].semilogy(t_mid, G_abs, '-', color=_W['black'],
                     linewidth=1.8, label=r'$|G_i|$')
    axes[0].set_ylabel(r'Energy violation $|G_i|$')
    axes[0].set_xlabel('Time $t$')
    add_grid(axes[0], 'both')
    axes[0].legend(loc='upper right')

    # Panel 2 — adjoint magnitude ||z_i||
    axes[1].semilogy(t_mid, z_norm, '-', color=_W['black'],
                     linewidth=1.8, label=r'$\|z_i\|$')
    axes[1].set_ylabel(r'Adjoint magnitude $\|z_i\|$')
    axes[1].set_xlabel('Time $t$')
    add_grid(axes[1], 'both')
    axes[1].legend(loc='upper right')

    # Panel 3 — DWR contribution |G_i| * ||z_i||
    axes[2].semilogy(t_mid, contribution, '-', color=_W['black'],
                     linewidth=1.8, label=r'$|G_i|\,\|z_i\|$')
    axes[2].set_ylabel(r'QoI contribution $|G_i|\,\|z_i\|$')
    axes[2].set_xlabel('Time $t$')
    add_grid(axes[2], 'both')
    axes[2].legend(loc='upper right')

    fig.tight_layout()
    fig.savefig("energy_balance_adjoint_compact.pdf", dpi=300)
    fig.savefig("energy_balance_adjoint_compact.png", dpi=300)
    plt.close(fig)
    print("Saved: energy_balance_adjoint_compact.pdf / .png")


if __name__ == "__main__":
    main()