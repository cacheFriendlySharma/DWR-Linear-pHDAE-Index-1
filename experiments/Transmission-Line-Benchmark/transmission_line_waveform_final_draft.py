"""
Waveform plots for the transmission line system loaded from rcl_ladder_system.mat.

System (MATLAB v7.3 / HDF5):
  Keys: E (152x152), J (152x152), R (152x152), G (152x152, = Q = I)

Algorithm (Algorithm 1)
-----------------------
* compute_eta: SIGNED indicators (no elementwise abs).
* Stopping    : eta_tot = |sum_i eta_i|   (signed sum, then abs).
* Dorfler     : sum_{M} |eta_i| >= theta * sum_i |eta_i|  (elementwise abs).
* Primal      : full S in system matrix.
* QoI G_i     : midpoint rule with S_tilde.
* Adjoint     : full RHS; LHS uses S.T.

Inline style: Wong (2011) colorblind-safe, Latin Modern Roman.
"""

import os
import h5py
import matplotlib
matplotlib.use('Agg')
import numpy as np
import scipy.linalg as la
import matplotlib.pyplot as plt

for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
          "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(v, "1")

# ==============================================================================
# Inline style
# ==============================================================================
plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Latin Modern Roman', 'DejaVu Serif', 'serif'],
    'mathtext.fontset': 'cm', 'font.size': 12,
    'axes.linewidth': 0.8, 'axes.labelsize': 13, 'axes.titlesize': 12,
    'axes.spines.top': False, 'axes.spines.right': False,
    'xtick.labelsize': 11, 'ytick.labelsize': 11,
    'xtick.direction': 'in', 'ytick.direction': 'in',
    'xtick.major.width': 0.6, 'ytick.major.width': 0.6,
    'xtick.minor.width': 0.4, 'ytick.minor.width': 0.4,
    'xtick.major.size': 5, 'ytick.major.size': 5,
    'xtick.minor.size': 3, 'ytick.minor.size': 3,
    'xtick.major.pad': 5, 'ytick.major.pad': 5,
    'axes.grid': False,
    'legend.fontsize': 9.5, 'legend.frameon': True, 'legend.framealpha': 0.92,
    'legend.edgecolor': '#cccccc', 'legend.fancybox': False,
    'legend.borderpad': 0.5, 'legend.handlelength': 2.0,
    'figure.dpi': 150, 'savefig.dpi': 300,
    'savefig.bbox': 'tight', 'savefig.pad_inches': 0.08,
})
_W = {
    'blue': '#0072B2', 'vermil': '#D55E00', 'skyblue': '#56B4E9',
    'green': '#009E73', 'pink': '#CC79A7', 'orange': '#E69F00',
    'purple': '#7B2D8B',
}
_NODE_COLORS = ['#000000', _W['green'], _W['vermil'], _W['orange'], _W['blue'], _W['pink']]


def add_grid(ax, which='major'):
    ax.grid(True, which='major', color='#cccccc', lw=0.3, alpha=0.5)
    if which == 'both':
        ax.grid(True, which='minor', color='#e0e0e0', lw=0.2, alpha=0.3)


# ==============================================================================
# System  --  loaded from .mat file  (MATLAB v7.3 / HDF5)
# ==============================================================================
class TransmissionLine_PH_DAE:
    """
    pH-DAE loaded from rcl_ladder_system.mat.

    File layout (HDF5 keys):
      E  -- descriptor matrix          (n x n)
      J  -- skew-symmetric structure   (n x n)
      R  -- symmetric PSD dissipation  (n x n)
      G  -- Hamiltonian weighting Q    (n x n, diagonal = I here)

    pH form:   E ẋ = (J - R) Q x + B u
    """

    def __init__(self, mat_path='rcl_ladder_system.mat'):
        with h5py.File(mat_path, 'r') as f:
            E_full = f['E'][()].T
            J_full = f['J'][()].T
            R_full = f['R'][()].T
            Q_full = f['G'][()].T   # 'G' in file == Q

        n = E_full.shape[0]
        self.n = n

        # pH axiom checks (soft)
        if not np.allclose(J_full, -J_full.T, atol=1e-10):
            print("  WARNING: J is not exactly skew-symmetric")
        if not np.allclose(R_full, R_full.T, atol=1e-10):
            print("  WARNING: R is not exactly symmetric")

        self.E = E_full
        self.J = J_full
        self.R = R_full
        self.Q = Q_full

        self.B      = np.zeros((n, 1))
        self.B[0,0] = 1.0

        # Differential / algebraic split  (differential DOFs contiguous at front)
        diag_E = np.abs(np.diag(E_full))
        self.r = int(np.sum(diag_E > 1e-10))
        r      = self.r

        if np.where(diag_E > 1e-10)[0].max() != r - 1:
            raise ValueError(
                "Differential DOFs are not contiguous in the first r rows.")

        J11 = J_full[:r, :r]
        n_q = next(
            (i for i in range(1, r) if np.any(np.abs(J11[i, :i]) > 1e-10)),
            r  # fallback: all vars are charge-like
        )
        self.n_q   = n_q
        self.n_phi = r - n_q

        # Schur complement onto differential block
        A   = (J_full - R_full) @ Q_full
        A11 = A[:r, :r];  A12 = A[:r, r:]
        A21 = A[r:, :r];  A22 = A[r:, r:]
        B1  = self.B[:r];  B2  = self.B[r:]

        A22_inv      = la.inv(A22)
        self.S       = -(A11 - A12 @ A22_inv @ A21)
        self.S_tilde =  0.5*(self.S + self.S.T)
        self.F       =  B1 - A12 @ A22_inv @ B2
        self.Kx      = -A22_inv @ A21
        self.Ku      = -A22_inv @ B2
        self.E11     = E_full[:r, :r]
        self.Q11     = Q_full[:r, :r]

        print(f"System loaded from: {mat_path}")
        print(f"  n={n}, r={r} diff vars "
              f"(auto-detected: {self.n_q} charge + {self.n_phi} flux), "
              f"{n - r} alg vars")

    def reconstruct(self, x1, u):
        """Recover algebraic variables from differential state x1 and input u."""
        x2     = self.Kx @ x1 + self.Ku @ u
        x_full = np.zeros(self.n)
        x_full[:self.r] = x1
        x_full[self.r:] = x2
        return x_full

    def hamiltonian(self, x1):
        return 0.5 * float(x1 @ (self.E11.T @ self.Q11) @ x1)

    def grad_hamiltonian(self, x1):
        return (self.E11.T @ self.Q11) @ x1


# ==============================================================================
# Input
# ==============================================================================
def make_input():
    def u(t):
        return np.array([50.0 * np.exp(-((t - 0.5)**2) / (2 * 0.05**2))])
    return u


# ==============================================================================
# Primal  dG(0)
# ==============================================================================
def solve_primal(sys, grid, x0, u_func):
    x    = np.zeros((len(grid), sys.r))
    x[0] = x0
    for i in range(len(grid) - 1):
        dt     = grid[i+1] - grid[i]
        tmid   = grid[i] + 0.5*dt
        x[i+1] = la.solve(sys.E11 + dt*sys.S,
                           sys.E11 @ x[i] + dt * sys.F @ u_func(tmid))
    return x


# ==============================================================================
# QoI residuals G_i
# ==============================================================================
def compute_G(sys, grid, x, u_func):
    M = len(grid) - 1
    G = np.zeros(M)
    for i in range(M):
        dt   = grid[i+1] - grid[i]
        tmid = grid[i] + 0.5*dt
        x1   = x[i+1]
        G[i] = (sys.hamiltonian(x[i+1]) - sys.hamiltonian(x[i])
                + dt * x1 @ sys.S_tilde @ x1
                - dt * u_func(tmid) @ (sys.F.T @ x1))
    return G


# ==============================================================================
# Adjoint
# ==============================================================================
def _adjoint_rhs(sys, grid, x, G, u_func):
    M   = len(grid) - 1
    rhs = np.zeros((M, sys.r))
    for i in range(M):
        dt     = grid[i+1] - grid[i]
        tmid   = grid[i] + 0.5*dt
        u      = u_func(tmid)
        grad_H = sys.grad_hamiltonian(x[i+1])
        rhs[i] = 2*G[i] * (grad_H
                            + 2*dt * sys.S_tilde @ x[i+1]
                            - dt   * (sys.F @ u))
        if i < M-1:
            rhs[i] -= 2*G[i+1] * grad_H
    return rhs


def solve_adjoint(sys, grid, x, u_func):
    M   = len(grid) - 1
    z   = np.zeros((M, sys.r))
    G   = compute_G(sys, grid, x, u_func)
    rhs = _adjoint_rhs(sys, grid, x, G, u_func)
    for i in reversed(range(M)):
        dt    = grid[i+1] - grid[i]
        local = rhs[i].copy()
        if i < M-1:
            local += sys.E11.T @ z[i+1]
        z[i] = la.solve(sys.E11.T + dt*sys.S.T, local)
    return z


# ==============================================================================
# DWR indicator — SIGNED
# ==============================================================================
def compute_eta(sys, grid, x, z, u_func):
    M   = len(grid) - 1
    eta = np.zeros(M)
    for i in range(M):
        dt   = grid[i+1] - grid[i]
        tmid = grid[i] + 0.5*dt
        Dz   = z[i] - (z[i-1] if i > 0 else np.zeros(sys.r))
        interior  = 0.5*dt * np.dot(sys.F @ u_func(tmid) - sys.S @ x[i+1], Dz)
        jump_term = np.dot(sys.E11 @ (x[i+1] - x[i]), Dz)
        eta[i]    = interior + jump_term   # SIGNED — no abs
    return eta


# ==============================================================================
# Dorfler marking — operates on |eta_i|
# ==============================================================================
def dorfler_marking(eta, theta=0.5):
    abs_eta = np.abs(eta)
    idx     = np.argsort(abs_eta)[::-1]
    cumsum  = np.cumsum(abs_eta[idx])
    n_mark  = np.searchsorted(cumsum, theta * cumsum[-1]) + 1
    return set(idx[:n_mark])


def refine_grid(grid, marked):
    new = [grid[0]]
    for i in range(len(grid) - 1):
        if i in marked:
            new.append(0.5*(grid[i] + grid[i+1]))
        new.append(grid[i+1])
    return np.array(new)


# ==============================================================================
# Adaptive loop  (Algorithm 1)
# ==============================================================================
def dwr_adaptive(sys, x0, u_func, T,
                 n_init=50, max_iter=20, theta=0.5, max_intervals=400):
    grid    = np.linspace(0, T, n_init)
    history = []
    for it in range(max_iter):
        x       = solve_primal(sys, grid, x0, u_func)
        z       = solve_adjoint(sys, grid, x, u_func)
        G       = compute_G(sys, grid, x, u_func)
        eta     = compute_eta(sys, grid, x, z, u_func)
        qoi     = float(np.sum(G**2))
        eta_tot = abs(float(np.sum(eta)))
        history.append({"n": len(grid)-1, "qoi": qoi,
                        "eta": eta_tot, "grid": grid.copy(), "x": x.copy()})
        print(f"  Iter {it:2d}: N={len(grid)-1:4d}, QoI={qoi:.4e}, "
              f"eta_tot={eta_tot:.4e}")
        if len(grid)-1 >= max_intervals:
            break
        marked = dorfler_marking(eta, theta=theta)
        grid   = refine_grid(grid, marked)
    return history


# ==============================================================================
# Waveform plots
# ==============================================================================
def plot_waveforms(sys, history, T, u_func):
    final_grid = history[-1]["grid"]
    x_final    = history[-1]["x"]   # shape (N+1, r)

    n_q, n_phi = sys.n_q, sys.n_phi

    node_picks = [0, 5, 10, 15, 20, 25]
    q_picks    = [k for k in node_picks if k < n_q]
    phi_picks  = [k for k in node_picks if k < n_phi]
    q_labels   = [rf"$v_{{{k+1}}}$"         for k in q_picks]
    phi_labels  = [rf"$\varphi_{{{k+1}}}$"  for k in phi_picks]

    # Extract waveforms directly from the differential state (no reconstruction needed)
    q_data   = x_final[:, :n_q]          # (N+1, n_q)
    phi_data = x_final[:, n_q:n_q+n_phi] # (N+1, n_phi)
    H_data   = np.array([sys.hamiltonian(x_final[t]) for t in range(len(final_grid))])

    # --- Figure 1:
    fig, ax = plt.subplots(figsize=(5.5, 3.5))
    for k, col, lab in zip(q_picks, _NODE_COLORS, q_labels):
        ax.plot(final_grid, q_data[:, k], color=col, lw=1.8, label=lab)
    ax.set_xlabel(r"Time $t$", fontsize=14)
    ax.set_ylabel(r"Node voltage $v_k(t)$", fontsize=14)
    ax.set_xlim(final_grid[0], final_grid[-1])
    ax.legend(loc='upper right', fontsize = 11)
    add_grid(ax)
    fig.tight_layout()
    fig.savefig("fig_waveform_charge.png", dpi=300)
    plt.close(fig)
    print("Saved: fig_waveform_charge.png")

    # --- Figure 2:
    fig, ax = plt.subplots(figsize=(6.5, 3.5))
    for k, col, lab in zip(phi_picks, _NODE_COLORS, phi_labels):
        ax.plot(final_grid, phi_data[:, k], color=col, lw=1.8, label=lab)
    ax.set_xlabel(r"Time $t$")
    ax.set_ylabel(r"Flux state $\varphi_k(t)$")
    ax.set_xlim(final_grid[0], final_grid[-1])
    ax.legend(loc='upper right')
    add_grid(ax)
    fig.tight_layout()
    fig.savefig("fig_waveform_flux.png", dpi=300)
    plt.close(fig)
    print("Saved: fig_waveform_flux.png")

    # --- Figure 3: Hamiltonian decay ---
    fig, ax = plt.subplots(figsize=(5.5, 3.5))
    ax.semilogy(final_grid, H_data, color=_W['purple'], lw=1.8)
    ax.set_xlabel(r"Time $t$")
    ax.set_ylabel(r"$H(x_1(t))$")
    add_grid(ax, 'both')
    fig.tight_layout()
    fig.savefig("fig_hamiltonian.png", dpi=300)
    plt.close(fig)
    print("Saved: fig_hamiltonian.png")

    # --- Figure 4: Adaptive step size distribution ---
    fig, ax = plt.subplots(figsize=(6.5, 3.0))
    dt_vals = np.diff(final_grid)
    ax.step(final_grid[:-1], dt_vals, where='post',
            color=_W['green'], lw=1.2)
    ax.set_yscale('log')
    ax.set_xlabel(r"Time $t$")
    ax.set_ylabel(r"Step size $k_i$")
    ax.set_xlim(final_grid[0], final_grid[-1])
    add_grid(ax, 'both')
    fig.tight_layout()
    fig.savefig("fig_dt_distribution.png", dpi=300)
    plt.close(fig)
    print("Saved: fig_dt_distribution.png")

    # --- Figure 5: Grid evolution across iterations ---
    fig, ax = plt.subplots(figsize=(7.0, 4.0))
    n_show  = min(20, len(history))
    colors  = plt.cm.viridis(np.linspace(0, 1, n_show))
    for row, (h, col) in enumerate(zip(history[-n_show:], colors)):
        g = h["grid"]
        ax.plot(g, np.full_like(g, len(history) - n_show + row),
                '|', color=col, ms=6, markeredgewidth=1.0)
    ax.set_xlabel(r"Time $t$")
    ax.set_ylabel("Refinement iteration")
    ax.set_xlim(final_grid[0], final_grid[-1])
    add_grid(ax)
    fig.tight_layout()
    fig.savefig("fig_grid_evolution.png", dpi=300)
    plt.close(fig)
    print("Saved: fig_grid_evolution.png")

    # --- Figure 6: DWR indicator structure at final iteration ---
    x_f   = history[-1]["x"]
    g_f   = history[-1]["grid"]
    z_f   = solve_adjoint(sys, g_f, x_f, u_func)
    G_f   = compute_G(sys, g_f, x_f, u_func)
    eta_f = compute_eta(sys, g_f, x_f, z_f, u_func)
    t_mid = 0.5*(g_f[:-1] + g_f[1:])

    z_norm  = np.linalg.norm(z_f, axis=1)
    G_abs   = np.abs(G_f)
    eta_abs = np.abs(eta_f)

    fig, axes = plt.subplots(1, 3, figsize=(10.0, 3.0), sharex=True)

    axes[0].semilogy(t_mid, G_abs,  color=_W['blue'],   lw=1.4)
    axes[0].set_ylabel(r"$|G_i|$")
    axes[0].set_xlabel(r"Time $t$")
    axes[0].set_title(r"Energy residual")
    add_grid(axes[0], 'both')

    axes[1].semilogy(t_mid, z_norm, color=_W['orange'], lw=1.4)
    axes[1].set_ylabel(r"$\|z_i\|$")
    axes[1].set_xlabel(r"Time $t$")
    axes[1].set_title(r"Adjoint sensitivity")
    add_grid(axes[1], 'both')

    axes[2].semilogy(t_mid, eta_abs, color=_W['green'], lw=1.4)
    axes[2].set_ylabel(r"$|\eta_i|$")
    axes[2].set_xlabel(r"Time $t$")
    axes[2].set_title(r"DWR indicator $|\eta_i|$")
    add_grid(axes[2], 'both')

    fig.tight_layout()
    fig.savefig("fig_indicator_structure.png", dpi=300)
    plt.close(fig)
    print("Saved: fig_indicator_structure.png")


# ==============================================================================
# Main
# ==============================================================================
if __name__ == "__main__":
    MAT_PATH = "rcl_ladder_system.mat"

    sys_obj = TransmissionLine_PH_DAE(mat_path=MAT_PATH)
    T       = 30.0
    u_func  = make_input()
    x0      = np.zeros(sys_obj.r)
    x0[0]   = 1.0

    print("\n=== Running DWR adaptive loop ===")
    history = dwr_adaptive(sys_obj, x0, u_func, T,
                           n_init=600, max_iter=20, theta=0.5,
                           max_intervals=1200)

    print("\n=== Generating waveform plots ===")
    plot_waveforms(sys_obj, history, T, u_func)

    print(f"\nFinal grid: N={history[-1]['n']}, QoI={history[-1]['qoi']:.4e}")