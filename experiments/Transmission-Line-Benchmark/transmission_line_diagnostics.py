"""
Additional diagnostics for the Transmission Line experiment (Section 8.6)
=========================================================================
1. Effectivity index table (analogous to Table 1 but at scale)
2. Spectral radius of G_i = M_i^{-1} E11^T  (validates Proposition 7.1)
3. Cost-to-target table: DWR vs Uniform savings
"""

import os
import matplotlib
matplotlib.use('Agg')

import numpy as np
import scipy.linalg as la
import matplotlib.pyplot as plt
from plot_style import apply_compact_style, finalize_figure

apply_compact_style()

for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
          "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(v, "1")

plt.rcParams.update({
    "font.size": 10,
    "axes.grid": True,
    "grid.alpha": 0.3
})


# ==============================================================================
# System Class (identical to fixed version)
# ==============================================================================
class TransmissionLine_PH_DAE:
    def __init__(self, n_nodes, r_base=0.02, r_jitter=0.0):
        n = 2 * n_nodes
        self.n = n
        self.J = np.zeros((n, n))
        for i in range(n_nodes):
            self.J[2*i, 2*i+1] = 1.0
            self.J[2*i+1, 2*i] = -1.0
            if 2*i + 2 < n:
                self.J[2*i+1, 2*i+2] = -1.0
                self.J[2*i+2, 2*i+1] = 1.0

        np.random.seed(42)
        self.R = np.diag(r_base + r_jitter * np.random.rand(n))
        self.Q = np.eye(n)
        self.E = np.eye(n)
        n_alg = max(1, n // 10)
        for k in range(n_alg):
            self.E[n - 1 - k, n - 1 - k] = 0.0

        self.B = np.zeros((n, 1))
        self.B[0, 0] = 1.0

        self.r = np.linalg.matrix_rank(self.E)
        A = (self.J - self.R) @ self.Q

        A11 = A[:self.r, :self.r]
        A12 = A[:self.r, self.r:]
        A21 = A[self.r:, :self.r]
        A22 = A[self.r:, self.r:]
        B1 = self.B[:self.r]
        B2 = self.B[self.r:]
        A22_inv = la.inv(A22)

        self.S = -(A11 - A12 @ A22_inv @ A21)
        self.S_tilde = 0.5 * (self.S + self.S.T)
        self.F = B1 - A12 @ A22_inv @ B2
        self.Kx = -A22_inv @ A21
        self.Ku = -A22_inv @ B2
        self.E11 = self.E[:self.r, :self.r]
        self.Q11 = self.Q[:self.r, :self.r]

    def reconstruct(self, x1, u):
        x2 = self.Kx @ x1 + self.Ku @ u
        return np.concatenate([x1, x2])

    def hamiltonian(self, x1):
        return 0.5 * x1 @ (self.E11.T @ self.Q11) @ x1

    def grad_hamiltonian(self, x1):
        return self.E11.T @ self.Q11 @ x1


# ==============================================================================
# Input
# ==============================================================================
def make_input():
    def u(t):
        center = 0.5; width = 0.05; amp = 50.0
        return np.array([amp * np.exp(-((t - center)**2) / (2 * width**2))])
    return u


# ==============================================================================
# Solvers (fixed versions)
# ==============================================================================
def solve_primal(sys, grid, x0, u_func):
    x = np.zeros((len(grid), sys.r))
    x[0] = x0
    for i in range(len(grid) - 1):
        dt = grid[i+1] - grid[i]
        tm = grid[i] + 0.5 * dt
        A = sys.E11 + dt * sys.S
        rhs = sys.E11 @ x[i] + dt * sys.F @ u_func(tm)
        x[i+1] = la.solve(A, rhs)
    return x


def compute_G(sys, grid, x, u_func):
    M = len(grid) - 1
    G = np.zeros(M)
    for i in range(M):
        dt = grid[i+1] - grid[i]
        u_left = u_func(grid[i])
        z_left = sys.reconstruct(x[i], u_left)
        e_left = sys.Q @ z_left
        y_left = sys.B.T @ e_left
        power_left = u_left @ y_left
        diss_left = e_left @ sys.R @ e_left

        u_right = u_func(grid[i+1])
        z_right = sys.reconstruct(x[i+1], u_right)
        e_right = sys.Q @ z_right
        y_right = sys.B.T @ e_right
        power_right = u_right @ y_right
        diss_right = e_right @ sys.R @ e_right

        integrand = 0.5 * dt * (
            (-power_left + diss_left) + (-power_right + diss_right)
        )
        G[i] = sys.hamiltonian(x[i+1]) - sys.hamiltonian(x[i]) + integrand
    return G


def solve_adjoint(sys, grid, x, u_func):
    M = len(grid) - 1
    z = np.zeros((M + 1, sys.r))
    G = compute_G(sys, grid, x, u_func)
    for i in reversed(range(M)):
        dt = grid[i+1] - grid[i]
        A = sys.E11.T + dt * sys.S.T
        rhs = sys.E11.T @ z[i+1]
        grad_H_ip1 = sys.grad_hamiltonian(x[i+1])
        rhs += 2 * G[i] * grad_H_ip1
        if i + 1 < M:
            rhs -= 2 * G[i+1] * grad_H_ip1
        z[i] = la.solve(A, rhs)
    return z


def compute_eta(sys, grid, x, z, u_func):
    M = len(grid) - 1
    eta = np.zeros(M)
    for i in range(M):
        dt = grid[i+1] - grid[i]
        tm = grid[i] + 0.5 * dt
        u = u_func(tm)
        Dz = z[i] if i == 0 else z[i] - z[i-1]
        jump = sys.E11 @ (x[i+1] - x[i])
        interior = 0.5 * dt * (sys.F @ u - sys.S @ x[i+1])
        eta[i] = abs(jump @ Dz + interior @ Dz)
    return eta


# ==============================================================================
# Adaptive Loop
# ==============================================================================
def dwr_adaptive(sys, x0, u_func, T, n_init=50, max_iter=30, theta=0.5,
                 max_intervals=400, verbose=False):
    grid = np.linspace(0, T, n_init)
    history = []

    for iteration in range(max_iter):
        x = solve_primal(sys, grid, x0, u_func)
        z = solve_adjoint(sys, grid, x, u_func)
        G = compute_G(sys, grid, x, u_func)
        eta = compute_eta(sys, grid, x, z, u_func)

        qoi = np.sum(G**2)
        eta_tot = np.sum(eta)

        history.append({
            "n_intervals": len(grid) - 1,
            "qoi": qoi,
            "eta": eta_tot,
            "grid": grid.copy(),
        })

        if verbose:
            print(f"  Iter {iteration:2d}: N={len(grid)-1:4d}, "
                  f"QoI={qoi:.4e}, eta={eta_tot:.4e}")

        if len(grid) - 1 >= max_intervals:
            break

        idx = np.argsort(eta)[::-1]
        cumsum = np.cumsum(eta[idx])
        marked = set(idx[:np.searchsorted(cumsum, theta * cumsum[-1]) + 1])

        new_grid = [grid[0]]
        for i in range(len(grid) - 1):
            if i in marked:
                new_grid.append(0.5 * (grid[i] + grid[i+1]))
            new_grid.append(grid[i+1])
        grid = np.array(new_grid)

    return grid, history


def run_uniform(sys, x0, u_func, T, n):
    grid = np.linspace(0, T, n + 1)
    x = solve_primal(sys, grid, x0, u_func)
    G = compute_G(sys, grid, x, u_func)
    return np.sum(G**2), grid, x


# ==============================================================================
# DIAGNOSTIC 1: Effectivity Index Table
# ==============================================================================
def effectivity_table(sys, x0, u_func, T):
    """
    Compute I_eff = eta_tot / |J(x_ref) - J(x_h)| across adaptive iterations.
    Reference computed on a very fine uniform grid.
    """
    print("=" * 72)
    print("DIAGNOSTIC 1: Effectivity Index Table (Transmission Line)")
    print("=" * 72)

    # Reference solution on very fine grid
    n_ref = 10000
    print(f"Computing reference on N={n_ref} uniform grid...")
    qoi_ref, _, _ = run_uniform(sys, x0, u_func, T, n_ref)
    print(f"  J(x_ref) = {qoi_ref:.6e}\n")

    # Run adaptive loop, recording eta and qoi at each iteration
    grid = np.linspace(0, T, 50)
    max_iter = 20
    theta = 0.5
    max_intervals = 500

    rows = []
    for iteration in range(max_iter):
        x = solve_primal(sys, grid, x0, u_func)
        z = solve_adjoint(sys, grid, x, u_func)
        G = compute_G(sys, grid, x, u_func)
        eta = compute_eta(sys, grid, x, z, u_func)

        qoi = np.sum(G**2)
        eta_tot = np.sum(eta)
        true_err = abs(qoi_ref - qoi)
        ieff = eta_tot / true_err if true_err > 1e-16 else float('inf')

        rows.append({
            "iter": iteration,
            "dofs": len(grid) - 1,
            "eta_tot": eta_tot,
            "true_err": true_err,
            "qoi": qoi,
            "ieff": ieff,
        })

        if len(grid) - 1 >= max_intervals:
            break

        idx = np.argsort(eta)[::-1]
        cumsum = np.cumsum(eta[idx])
        marked = set(idx[:np.searchsorted(cumsum, theta * cumsum[-1]) + 1])
        new_grid = [grid[0]]
        for i in range(len(grid) - 1):
            if i in marked:
                new_grid.append(0.5 * (grid[i] + grid[i+1]))
            new_grid.append(grid[i+1])
        grid = np.array(new_grid)

    # Print table
    header = f"{'Iter':>4s}  {'DOFs':>5s}  {'Est. Error (η_tot)':>18s}  " \
             f"{'True Error':>14s}  {'I_eff':>8s}"
    print(header)
    print("-" * len(header))
    for r in rows:
        print(f"{r['iter']:4d}  {r['dofs']:5d}  {r['eta_tot']:18.2e}  "
              f"{r['true_err']:14.2e}  {r['ieff']:8.2f}")
    print()

    return rows


# ==============================================================================
# DIAGNOSTIC 2: Spectral Radius of G_i (Proposition 7.1)
# ==============================================================================
def spectral_radius_plot(sys, x0, u_func, T):
    """
    Compute rho(G_i) = rho(M_i^{-1} E11^T) for each time step on both
    a uniform grid and the final adaptive grid.
    
    G_i = (E11^T + k_i S_tilde^T)^{-1} E11^T   [Eq. (69)]
    
    Proposition 7.1 guarantees rho(G_i) < 1 for all k_i > 0.
    """
    print("=" * 72)
    print("DIAGNOSTIC 2: Spectral Radius of G_i (Proposition 7.1)")
    print("=" * 72)

    E11T = sys.E11.T
    ST = sys.S_tilde.T

    def compute_rho_sequence(grid):
        """Compute spectral radius of G_i for each interval."""
        M = len(grid) - 1
        rho_vals = np.zeros(M)
        t_mid = np.zeros(M)
        for i in range(M):
            ki = grid[i+1] - grid[i]
            t_mid[i] = 0.5 * (grid[i] + grid[i+1])
            Mi = E11T + ki * ST
            Gi = la.solve(Mi, E11T)
            eigvals = la.eigvalsh(Gi)  # G_i is similar to symmetric
            rho_vals[i] = np.max(np.abs(eigvals))
        return t_mid, rho_vals

    # Uniform grid
    n_uni = 200
    grid_uni = np.linspace(0, T, n_uni + 1)
    t_uni, rho_uni = compute_rho_sequence(grid_uni)
    print(f"Uniform (N={n_uni}):  rho in [{rho_uni.min():.6f}, {rho_uni.max():.6f}]")

    # Adaptive grid (run a quick adaptive loop)
    _, hist = dwr_adaptive(sys, x0, u_func, T, n_init=50, max_iter=25,
                           max_intervals=300)
    grid_dwr = hist[-1]["grid"]
    t_dwr, rho_dwr = compute_rho_sequence(grid_dwr)
    print(f"DWR     (N={len(grid_dwr)-1}):  rho in [{rho_dwr.min():.6f}, {rho_dwr.max():.6f}]")
    print()

    # --- Updated Plot ---
    # Reduced height from 4 to 3 (or 2.5) to make it more compact
    fig, axes = plt.subplots(1, 2, figsize=(7, 2.5), sharey=True)

    # (a) rho vs time
    # Increased lw to 2.0 and ms to 5 for better visibility
    axes[0].plot(t_uni, rho_uni, 'b-', lw=2.0, alpha=0.8, label=f"Uniform (N={n_uni})")
    axes[0].plot(t_dwr, rho_dwr, 'r.', ms=6, alpha=0.7, 
             label=f"DWR (N={len(grid_dwr)-1})")
    axes[0].axhline(y=1.0, color='k', ls='--', lw=1.2, label=r"$\rho = 1$ bound")

    axes[0].set_xlabel("Time $t$")
    axes[0].set_ylabel(r"$\rho(G_i)$")
    axes[0].set_title(r"(a) Spectral radius vs time", fontsize=11)
    axes[0].legend(fontsize=5, loc='lower left') # Normalized font size
    axes[0].set_ylim(0.8, 1.05)

    # (b) rho vs step size k_i (log scale on x)
    dt_uni = np.diff(grid_uni)
    dt_dwr = np.diff(grid_dwr)

    # Increased marker size for scattered points
    axes[1].semilogx(dt_uni, rho_uni, 'b.', ms=6, alpha=0.5, label="Uniform")
    axes[1].semilogx(dt_dwr, rho_dwr, 'r.', ms=6, alpha=0.5, label="DWR")
    axes[1].axhline(y=1.0, color='k', ls='--', lw=1.2)

    axes[1].set_xlabel(r"Step size $k_i$")
    axes[1].set_ylabel(r"$\rho(G_i)$")
    axes[1].set_title(r"(b) $\rho(G_i)$ vs step size", fontsize=11)
    axes[1].legend(fontsize=4) # Normalized font size

    # Fine-tuning layout
    plt.tight_layout() 
    finalize_figure() # Assuming this is your custom style function

    plt.savefig("fig_spectral_radius.png", dpi=300, bbox_inches='tight')
    plt.close()
    print("Saved: fig_spectral_radius.png (Optimized size and thickness)\n")


# ==============================================================================
# DIAGNOSTIC 3: Cost-to-Target Table
# ==============================================================================
def cost_to_target_table(sys, x0, u_func, T):
    """
    For a set of QoI accuracy targets, find N required by uniform and DWR,
    and compute the savings ratio.
    """
    print("=" * 72)
    print("DIAGNOSTIC 3: Cost-to-Target Table (DWR vs Uniform)")
    print("=" * 72)

    # --- Uniform sweep (fine-grained for interpolation) ---
    uni_ns = [50, 75, 100, 150, 200, 300, 400, 600, 800, 1200, 1600, 2400, 3200]
    uni_qs = []
    print("Running uniform sweep...")
    for n in uni_ns:
        q, _, _ = run_uniform(sys, x0, u_func, T, n)
        uni_qs.append(q)
        print(f"  N={n:5d}: QoI = {q:.4e}")
    print()

    # --- DWR adaptive (allow enough iterations to reach tight targets) ---
    print("Running DWR adaptive...")
    _, hist = dwr_adaptive(sys, x0, u_func, T, n_init=50, max_iter=100,
                           max_intervals=2000)
    dwr_ns = [h["n_intervals"] for h in hist]
    dwr_qs = [h["qoi"] for h in hist]
    print()

    # --- Define targets ---
    targets = [1e1, 1e0, 1e-1, 1e-2, 1e-3, 1e-4, 1e-5]

    def find_n_for_target(ns, qs, target):
        """Find first N where QoI <= target via linear interpolation in log-log."""
        for j in range(len(qs)):
            if qs[j] <= target:
                if j == 0:
                    return ns[0]
                # Log-linear interpolation between j-1 and j
                log_n0, log_n1 = np.log(ns[j-1]), np.log(ns[j])
                log_q0, log_q1 = np.log(qs[j-1]), np.log(qs[j])
                log_target = np.log(target)
                frac = (log_target - log_q0) / (log_q1 - log_q0)
                return int(np.ceil(np.exp(log_n0 + frac * (log_n1 - log_n0))))
        return None  # target not reached

    # --- Build table ---
    header = f"{'Target QoI':>12s}  {'N (Uniform)':>12s}  {'N (DWR)':>10s}  " \
             f"{'Savings':>10s}"
    print(header)
    print("-" * len(header))

    table_rows = []
    for tgt in targets:
        n_uni = find_n_for_target(uni_ns, uni_qs, tgt)
        n_dwr = find_n_for_target(dwr_ns, dwr_qs, tgt)

        uni_str = f"{n_uni}" if n_uni is not None else "> 3200"
        dwr_str = f"{n_dwr}" if n_dwr is not None else f"> {dwr_ns[-1]}"

        if n_uni is not None and n_dwr is not None:
            savings = 1.0 - n_dwr / n_uni
            sav_str = f"{savings*100:.0f}%"
        else:
            sav_str = "—"

        print(f"{tgt:12.0e}  {uni_str:>12s}  {dwr_str:>10s}  {sav_str:>10s}")
        table_rows.append({
            "target": tgt, "n_uni": n_uni, "n_dwr": n_dwr,
        })

    print()
    return uni_ns, uni_qs, dwr_ns, dwr_qs, table_rows


# ==============================================================================
# Combined convergence plot with reference slopes
# ==============================================================================
def convergence_with_slopes(uni_ns, uni_qs, dwr_ns, dwr_qs):
    """Convergence plot with fitted reference slopes."""

    # Fit uniform slope in log-log (use asymptotic range, skip first 2 points)
    skip_uni = 2
    log_n = np.log(np.array(uni_ns[skip_uni:], dtype=float))
    log_q = np.log(np.array(uni_qs[skip_uni:], dtype=float))
    slope_uni, intercept_uni = np.polyfit(log_n, log_q, 1)

    # Fit DWR slope
    skip_dwr = min(8, len(dwr_ns) // 3)
    valid_dwr = [(n, q) for n, q in zip(dwr_ns[skip_dwr:], dwr_qs[skip_dwr:]) if q > 0]
    if len(valid_dwr) > 2:
        ns_d, qs_d = zip(*valid_dwr)
        log_n_d = np.log(np.array(ns_d, dtype=float))
        log_q_d = np.log(np.array(qs_d, dtype=float))
        slope_dwr, intercept_dwr = np.polyfit(log_n_d, log_q_d, 1)
    else:
        slope_dwr = slope_uni

    print(f"Fitted convergence rates:")
    print(f"  Uniform: O(N^{{{slope_uni:.2f}}})")
    print(f"  DWR:     O(N^{{{slope_dwr:.2f}}})")
    print()

    # Reference line: use nearest half-integer slope from uniform fit
    ref_slope = round(slope_uni * 2) / 2.0
    n_arr = np.array([uni_ns[skip_uni], uni_ns[-1]], dtype=float)
    # Anchor reference line to go through the middle of uniform data
    mid_idx = len(uni_ns) // 2
    anchor_n = float(uni_ns[mid_idx])
    anchor_q = float(uni_qs[mid_idx])
    q_ref_line = anchor_q * (n_arr / anchor_n) ** ref_slope * 2.5  # offset for clarity

    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    ax.loglog(uni_ns, uni_qs, "b^-", label="Uniform", ms=8, lw=1.5)
    ax.loglog(dwr_ns, dwr_qs, "ro-", label="DWR", ms=4, lw=1.5)
    ax.loglog(n_arr, q_ref_line, "k--", lw=1.0,
              label=fr"$O(N^{{{ref_slope:.1f}}})$")

    ax.set_xlabel("Number of intervals $N$")
    ax.set_ylabel(r"Energy balance violation $\sum G_i^2$")
    ax.set_title("Convergence: Uniform vs DWR-Adaptive")
    ax.legend(fontsize=9)
    finalize_figure()
    plt.savefig("fig_convergence_with_slopes.png", dpi=300)
    plt.close()
    print("Saved: fig_convergence_with_slopes.png\n")


# ==============================================================================
# Main
# ==============================================================================
def main():
    print("Initializing system...")
    sys = TransmissionLine_PH_DAE(n_nodes=50, r_base=0.35, r_jitter=0.00)
    T = 10.0
    u = make_input()
    x0 = np.zeros(sys.r)
    x0[0] = 1.0
    print()

    # --- Diagnostic 1: Effectivity table ---
    effectivity_table(sys, x0, u, T)

    # --- Diagnostic 2: Spectral radius ---
    spectral_radius_plot(sys, x0, u, T)

    # --- Diagnostic 3: Cost-to-target + convergence with slopes ---
    uni_ns, uni_qs, dwr_ns, dwr_qs, _ = cost_to_target_table(sys, x0, u, T)
    convergence_with_slopes(uni_ns, uni_qs, dwr_ns, dwr_qs)

    print("=" * 72)
    print("All diagnostics complete.")
    print("=" * 72)


if __name__ == "__main__":
    main()