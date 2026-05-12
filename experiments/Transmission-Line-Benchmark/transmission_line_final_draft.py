"""
Additional diagnostics for the Transmission Line experiment (Section 8.6)
=========================================================================
System loaded from: rcl_ladder_system.mat  (MATLAB v7.3 / HDF5)

pH-DAE form:  E ẋ = (J - R) Q x + B u,  with B = e_1 (canonical first column)

1. Effectivity index table
2. Spectral radius of G_i = (E11^T + k_i S_tilde^T)^{-1} E11^T 
3. Cost-to-target table: DWR dG(0) vs Uniform dG(0)
4. Convergence plot with fitted slopes   (dG(0))


Algorithm (Algorithm 1)
-----------------------
* compute_eta : SIGNED indicators -- no elementwise abs anywhere.
* Stopping    : eta_tot = |sum_i eta_i|   (signed sum, then single abs).
* Dorfler     : accumulate sum_{M} |eta_i| >= theta * sum_i |eta_i|
                (sort and accumulate on abs values).
* Primal      : system matrix E11 + dt*S   (full Schur complement S).
* QoI  G_i   : midpoint rule with S_tilde  (skew part vanishes -- Lemma 2.3).
* Adjoint RHS : full expression including distributed source d(G_i)/d(x_{i+1}).
* Adjoint LHS : E11.T + dt*S.T             (transpose of full S).

Effectivity index uses the signed estimator:
  I_eff = |sum_i eta_i| / |J_h - J_ref|

Inline style: Wong (2011) colorblind-safe palette, Latin Modern Roman.
"""

import os
import h5py
import matplotlib
import matplotlib.ticker
matplotlib.use('Agg')
import numpy as np
import scipy.linalg as la
import matplotlib.pyplot as plt

for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
          "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(v, "1")

# =============================================================================
# Inline style  (Wong 2011 palette, Latin Modern Roman)
# =============================================================================
plt.rcParams.update({
    'font.family'         : 'serif',
    'font.serif'          : ['Latin Modern Roman', 'DejaVu Serif', 'serif'],
    'mathtext.fontset'    : 'cm',
    'font.size'           : 12,
    'axes.linewidth'      : 0.8,
    'axes.labelsize'      : 13,
    'axes.titlesize'      : 12,
    'axes.spines.top'     : False,
    'axes.spines.right'   : False,
    'xtick.labelsize'     : 11,
    'ytick.labelsize'     : 11,
    'xtick.direction'     : 'in',
    'ytick.direction'     : 'in',
    'xtick.major.width'   : 0.6,
    'ytick.major.width'   : 0.6,
    'xtick.minor.width'   : 0.4,
    'ytick.minor.width'   : 0.4,
    'xtick.major.size'    : 5,
    'ytick.major.size'    : 5,
    'xtick.minor.size'    : 3,
    'ytick.minor.size'    : 3,
    'xtick.major.pad'     : 5,
    'ytick.major.pad'     : 5,
    'axes.grid'           : False,
    'legend.fontsize'     : 9.5,
    'legend.frameon'      : True,
    'legend.framealpha'   : 0.92,
    'legend.edgecolor'    : '#cccccc',
    'legend.fancybox'     : False,
    'legend.borderpad'    : 0.5,
    'legend.handlelength' : 2.0,
    'figure.dpi'          : 150,
    'savefig.dpi'         : 300,
    'savefig.bbox'        : 'tight',
    'savefig.pad_inches'  : 0.08,
})

_W = {
    'blue'    : '#0072B2',
    'vermil'  : '#D55E00',
    'skyblue' : '#56B4E9',
    'green'   : '#009E73',
    'pink'    : '#CC79A7',
    'orange'  : '#E69F00',
    'black'   : '#000000',
}

def _grid(ax, which='major'):
    ax.grid(True, which='major', color='#cccccc', lw=0.3, alpha=0.5)
    if which == 'both':
        ax.grid(True, which='minor', color='#e0e0e0', lw=0.2, alpha=0.3)


MSTYLE = {
    'Uniform': dict(
        color=_W['blue'], marker='o', ls='--', lw=1.4, ms=5.5,
        mec='white', mew=0.6, zorder=2, label='Uniform dG(0)',
    ),
    'DWR': dict(
        color=_W['pink'], marker='^', ls='-', lw=2.2, ms=6,
        mec='white', mew=0.7, zorder=5, label='DWR dG(0)',
    ),
}


# =============================================================================
# System  --  loaded from .mat file  (MATLAB v7.3 / HDF5)
# =============================================================================
class TransmissionLine_PH_DAE:
    """
    pH-DAE loaded from rcl_ladder_system.mat.

    File layout (HDF5 keys):
      E  -- descriptor matrix          (n x n)
      J  -- skew-symmetric structure   (n x n)
      R  -- symmetric PSD dissipation  (n x n)
      G  -- Hamiltonian weighting Q    (n x n, diagonal = I here)

    pH form:   E ẋ = (J - R) Q x + B u
    Input B  : canonical first column e_1   (not stored in file)
    Partition: first r rows/cols are differential (diag(E)>0),
               remaining n-r are algebraic.
    """

    def __init__(self, mat_path='rcl_ladder_system.mat'):
        # ---- load from HDF5 / MATLAB v7.3 ----
        with h5py.File(mat_path, 'r') as f:
            # MATLAB stores arrays in column-major order; transpose to row-major.
            E_full = f['E'][()].T
            J_full = f['J'][()].T
            R_full = f['R'][()].T
            Q_full = f['G'][()].T   # 'G' in file == Q (Hamiltonian weight)

        n = E_full.shape[0]
        self.n = n

        # Verify pH axioms (soft checks -- warn only)
        if not np.allclose(J_full, -J_full.T, atol=1e-10):
            print("  WARNING: J is not exactly skew-symmetric")
        if not np.allclose(R_full, R_full.T, atol=1e-10):
            print("  WARNING: R is not exactly symmetric")
        r_eigs = la.eigvalsh(R_full)
        if r_eigs.min() < -1e-10:
            print(f"  WARNING: R has negative eigenvalue {r_eigs.min():.3e}")

        self.E = E_full
        self.J = J_full
        self.R = R_full
        self.Q = Q_full

        # Input matrix: single port at node 0  
        self.B      = np.zeros((n, 1))
        self.B[0,0] = 1.0

        # Rank and differential/algebraic split
        # Differential DOFs are those where diag(E) > 0  (already in first r rows)
        diag_E     = np.abs(np.diag(E_full))
        self.r     = int(np.sum(diag_E > 1e-10))
        r          = self.r

        # Sanity: check that differential DOFs are contiguous at the front
        diff_idx = np.where(diag_E > 1e-10)[0]
        if diff_idx.max() != r - 1:
            raise ValueError(
                f"Differential DOFs are not contiguous in the first {r} rows. "
                f"Reorder the system before calling this class.")

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

    def hamiltonian(self, x1):
        return 0.5 * float(x1 @ (self.E11.T @ self.Q11) @ x1)

    def grad_hamiltonian(self, x1):
        return (self.E11.T @ self.Q11) @ x1

    def print_summary(self):
        print(f"  System loaded from .mat file")
        print(f"  n = {self.n}  (total DOFs)")
        print(f"  r = {self.r}  (differential vars)")
        print(f"  n - r = {self.n - self.r}  (algebraic vars)")
        lam_S = la.eigvalsh(self.S_tilde)
        print(f"  lambda(S_tilde): min={lam_S.min():.4e}, max={lam_S.max():.4e}")
        print(f"  J skew-sym:  {np.allclose(self.J, -self.J.T, atol=1e-10)}")
        print(f"  R sym PSD:   {np.allclose(self.R, self.R.T, atol=1e-10)}, "
              f"min_eig={la.eigvalsh(self.R).min():.3e}")
        print()


# =============================================================================
# Input
# =============================================================================
def make_input():
    def u(t):
        return np.array([50.0 * np.exp(-((t - 0.5)**2) / (2 * 0.05**2))])
    return u


# =============================================================================
# dG(0) primal
# =============================================================================
def solve_primal(sys, grid, x0, u_func):
    x    = np.zeros((len(grid), sys.r))
    x[0] = x0
    for i in range(len(grid) - 1):
        dt     = grid[i+1] - grid[i]
        tmid   = grid[i] + 0.5*dt
        x[i+1] = la.solve(sys.E11 + dt*sys.S,
                           sys.E11 @ x[i] + dt * sys.F @ u_func(tmid))
    return x


# =============================================================================
# QoI residuals G_i  
# =============================================================================
def compute_G_sym(sys, grid, x, u_func):
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


# =============================================================================
# dG(0) adjoint backward solve
# =============================================================================
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
    G   = compute_G_sym(sys, grid, x, u_func)
    rhs = _adjoint_rhs(sys, grid, x, G, u_func)
    for i in reversed(range(M)):
        dt    = grid[i+1] - grid[i]
        local = rhs[i].copy()
        if i < M-1:
            local += sys.E11.T @ z[i+1]
        z[i] = la.solve(sys.E11.T + dt*sys.S.T, local)
    return z


# =============================================================================
# DWR indicator -- SIGNED
# =============================================================================
def compute_eta(sys, grid, x, z, u_func):
    M   = len(grid) - 1
    eta = np.zeros(M)
    for i in range(M):
        dt   = grid[i+1] - grid[i]
        tmid = grid[i] + 0.5*dt
        # Boundary convention: Delta z_1 = 0 (i.e., z_k^0 := z_k^1)
        # In code: z[i] corresponds to z_k^{i+1} in paper Convention A
        # At i=0: Dz = z[0] - z[0] = 0  (boundary rule)
        # At i>0: Dz = z[i] - z[i-1]
        if i == 0:
            Dz = np.zeros(sys.r)
        else:
            Dz = z[i] - z[i-1]
        interior  = 0.5*dt * np.dot(sys.F @ u_func(tmid) - sys.S @ x[i+1], Dz)
        jump_term = np.dot(sys.E11 @ (x[i+1] - x[i]), Dz)
        eta[i]    = interior + jump_term
    return eta


# =============================================================================
# Dorfler marking and grid refinement
# =============================================================================
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


# =============================================================================
# dG(0) adaptive loop  (Algorithm 1)
# =============================================================================
def dwr_adaptive(sys, x0, u_func, T,
                 n_init=50, max_iter=30, theta=0.5,
                 max_intervals=400, verbose=False):
    grid    = np.linspace(0, T, n_init + 1)
    history = []
    for it in range(max_iter):
        x       = solve_primal(sys, grid, x0, u_func)
        z       = solve_adjoint(sys, grid, x, u_func)
        G       = compute_G_sym(sys, grid, x, u_func)
        eta     = compute_eta(sys, grid, x, z, u_func)
        qoi     = float(np.sum(G**2))
        eta_tot = abs(float(np.sum(eta)))
        history.append({"n_intervals": len(grid)-1, "qoi": qoi,
                        "eta": eta_tot, "grid": grid.copy()})
        if verbose:
            print(f"  Iter {it:2d}: N={len(grid)-1:4d}, "
                  f"QoI={qoi:.4e}, eta_tot={eta_tot:.4e}")
        if len(grid)-1 >= max_intervals:
            break
        marked = dorfler_marking(eta, theta=theta)
        grid   = refine_grid(grid, marked)
    return grid, history


def run_uniform(sys, x0, u_func, T, n):
    grid = np.linspace(0, T, n+1)
    x    = solve_primal(sys, grid, x0, u_func)
    G    = compute_G_sym(sys, grid, x, u_func)
    return float(np.sum(G**2)), grid, x


# =============================================================================
# DIAGNOSTIC 1: Effectivity Index Table
# =============================================================================
def effectivity_table(sys, x0, u_func, T):
    print("="*72)
    print("DIAGNOSTIC 1: Effectivity Index")
    print("="*72)

    n_ref = 50000
    print(f"Reference on N={n_ref}...")
    qoi_ref, _, _ = run_uniform(sys, x0, u_func, T, n_ref)
    print(f"  J_ref = {qoi_ref:.6e}\n")

    grid = np.linspace(0, T, 50)
    rows = []
    for it in range(25):
        x        = solve_primal(sys, grid, x0, u_func)
        z        = solve_adjoint(sys, grid, x, u_func)
        G        = compute_G_sym(sys, grid, x, u_func)
        eta      = compute_eta(sys, grid, x, z, u_func)
        qoi      = float(np.sum(G**2))
        eta_est  = float(np.sum(eta))
        eta_tot  = abs(eta_est)
        true_err = abs(qoi_ref - qoi)
        ieff     = eta_tot / true_err if true_err > 1e-16 else float('inf')
        rows.append({"iter": it, "dofs": len(grid)-1,
                     "eta_est": eta_est, "eta_tot": eta_tot,
                     "true_err": true_err, "qoi": qoi, "ieff": ieff})
        if len(grid)-1 >= 500:
            break
        marked = dorfler_marking(eta, theta=0.5)
        grid   = refine_grid(grid, marked)

    hdr = (f"{'Iter':>4}  {'DOFs':>5}  {'eta_est (sgn)':>15}  "
           f"{'True error':>14}  {'I_eff':>8}")
    print(hdr)
    print("-" * len(hdr))
    for r in rows:
        print(f"{r['iter']:4d}  {r['dofs']:5d}  {r['eta_est']:15.4e}  "
              f"{r['true_err']:14.4e}  {r['ieff']:8.3f}")
    print()
    return rows


# =============================================================================
# DIAGNOSTIC 2: Spectral Radius  
# =============================================================================
def spectral_radius_plot(sys, x0, u_func, T):
    print("="*72)
    print("DIAGNOSTIC 2: Spectral radius rho(G_i)  (Proposition 7.1)")
    print("="*72)

    E11T = sys.E11.T
    ST   = sys.S_tilde.T

    def rho_sequence(grid):
        M     = len(grid) - 1
        rho_v = np.zeros(M)
        for i in range(M):
            ki       = grid[i+1] - grid[i]
            Mi       = E11T + ki*ST
            Gi       = la.solve(Mi, E11T)
            # Gi is symmetric iff E11 and S_tilde commute; use eigvals for safety
            rho_v[i] = np.max(np.abs(la.eigvals(Gi)))
        return 0.5*(grid[:-1] + grid[1:]), rho_v

    n_uni    = 200
    grid_uni = np.linspace(0, T, n_uni+1)
    t_uni, rho_uni = rho_sequence(grid_uni)
    print(f"Uniform N={n_uni}:  rho in [{rho_uni.min():.6f}, {rho_uni.max():.6f}]")

    _, hist = dwr_adaptive(sys, x0, u_func, T, n_init=50,
                           max_iter=25, max_intervals=300)
    grid_dwr       = hist[-1]["grid"]
    t_dwr, rho_dwr = rho_sequence(grid_dwr)
    print(f"DWR     N={len(grid_dwr)-1}:  rho in [{rho_dwr.min():.6f}, {rho_dwr.max():.6f}]")

    eigvals_S    = la.eigvalsh(sys.S_tilde)
    lam_min      = eigvals_S.min()           # true minimum — governs rho
    lam_min_pos  = eigvals_S[eigvals_S > 1e-12].min() if np.any(eigvals_S > 1e-12) else 0.0
    n_zero       = int(np.sum(np.abs(eigvals_S) < 1e-10))
    # Boundary case: zero eigenvalues of S_tilde => rho = 1 identically
    boundary_case = (n_zero > 0)
    k_theory   = np.logspace(np.log10(np.diff(grid_uni).min()),
                              np.log10(np.diff(grid_uni).max()), 200)
    # Theory curve: use lam_min_pos to show what rho *would* be without zero modes
    rho_theory = 1.0 / (1.0 + k_theory * lam_min_pos)
    print(f"lambda_min(S_tilde) = {lam_min:.6e}  (pos: {lam_min_pos:.6f}, zeros: {n_zero})")
    print(f"Boundary case (rho=1): {boundary_case}")
    print(f"Theoretical rho range (pos modes): [{rho_theory.min():.6f}, {rho_theory.max():.6f}]")
    print()

    # Detect boundary case: lambda_min = 0 => rho = 1 identically
    boundary_case = (lam_min < 1e-10)

    fig, axes = plt.subplots(1, 2, figsize=(9.5, 3.5))

    # --- Left panel: rho vs time ---
    if boundary_case:
        # All values are 1.0 — plot as constant line with annotation
        axes[0].axhline(1.0, color=_W['blue'], lw=1.4,
                        label=f"Uniform (N={n_uni})")
        axes[0].axhline(1.0, color=_W['vermil'], lw=1.0, ls=':',
                        label=f"DWR (N={len(grid_dwr)-1})")
        axes[0].set_ylim(0.95, 1.05)
        axes[0].yaxis.set_major_formatter(
            matplotlib.ticker.FormatStrFormatter('%.2f'))
        axes[0].text(0.5, 0.5,
                     r"$\lambda_{\min}(\widetilde{S})=0$:" + "\n"
                     r"$\rho(\Gamma_i)=1$ (boundary of Assumption~2)",
                     transform=axes[0].transAxes,
                     ha='center', va='center', fontsize=9,
                     color='#555555',
                     bbox=dict(boxstyle='round,pad=0.3', fc='white',
                               ec='#cccccc', alpha=0.9))
    else:
        axes[0].plot(t_uni, rho_uni, color=_W['blue'], lw=1.2,
                     label=f"Uniform (N={n_uni})")
        axes[0].plot(t_dwr, rho_dwr, '.', color=_W['vermil'], ms=3,
                     label=f"DWR (N={len(grid_dwr)-1})")
        all_rho_t = np.concatenate([rho_uni, rho_dwr])
        rho_lo_t  = max(0.0, all_rho_t.min() - 0.02*(1.0 - all_rho_t.min()))
        axes[0].set_ylim(rho_lo_t, 1.001)
        axes[0].yaxis.set_major_formatter(
            matplotlib.ticker.FormatStrFormatter('%.4f'))

    axes[0].axhline(1.0, color=_W['black'], ls='--', lw=0.8, label=r"$\rho=1$")
    axes[0].set_xlabel(r"Time $t$")
    axes[0].set_ylabel(r"$\rho(\Gamma_i)$")
    axes[0].legend(loc='lower left', fontsize=12)
    _grid(axes[0])

    # --- Right panel: rho vs step size ---
    k_all      = np.concatenate([np.diff(grid_uni), np.diff(grid_dwr)])
    k_theory   = np.logspace(np.log10(k_all.min()), np.log10(k_all.max()), 400)
    rho_theory = 1.0 / (1.0 + k_theory * lam_min_pos)

    axes[1].semilogx(k_theory, rho_theory, color=_W['black'], ls='--', lw=1.2,
                     label=r"$1/(1+k_i\lambda_{\min}^+)$")
    axes[1].semilogx(np.diff(grid_uni), rho_uni, '.', color=_W['blue'],
                     ms=4, alpha=0.6, label="Uniform")
    axes[1].semilogx(np.diff(grid_dwr), rho_dwr, '.', color=_W['vermil'],
                     ms=4, alpha=0.6, label="DWR")
    axes[1].axhline(1.0, color=_W['black'], ls=':', lw=0.8)
    axes[1].set_xlabel(r"Step size $k_i$")
    axes[1].set_ylabel(r"$\rho(\Gamma_i)$")

    if boundary_case:
        axes[1].set_ylim(0.97, 1.03)
        axes[1].yaxis.set_major_formatter(
            matplotlib.ticker.FormatStrFormatter('%.2f'))
        axes[1].text(0.5, 0.25,
                     rf"Data: $\rho=1$ ({n_zero} zero modes of $\widetilde{{S}}$,"
                     "\nboundary of Assumption~2)",
                     transform=axes[1].transAxes,
                     ha='center', va='center', fontsize=8.5,
                     color='#444444',
                     bbox=dict(boxstyle='round,pad=0.35', fc='white',
                               ec='#cccccc', alpha=0.92))
    else:
        all_rho  = np.concatenate([rho_uni, rho_dwr, rho_theory])
        rho_lo   = max(0.0, all_rho.min() - 0.02*(1.0 - all_rho.min()))
        axes[1].set_ylim(rho_lo, 1.001)
        axes[1].yaxis.set_major_formatter(
            matplotlib.ticker.FormatStrFormatter('%.4f'))

    axes[1].legend(fontsize=14)
    _grid(axes[1])

    fig.tight_layout()
    fig.savefig("fig_spectral_radius.png", dpi=300)
    plt.close(fig)
    print("Saved: fig_spectral_radius.png\n")


# =============================================================================
# DIAGNOSTIC 3: Cost-to-target table  (dG(0))
# =============================================================================
def _find_n_for_target(ns, qs, target):
    for j, q in enumerate(qs):
        if q <= target:
            if j == 0:
                return ns[0]
            ln0, ln1 = np.log(ns[j-1]), np.log(ns[j])
            lq0, lq1 = np.log(qs[j-1]), np.log(qs[j])
            frac = (np.log(target) - lq0) / (lq1 - lq0)
            return int(np.ceil(np.exp(ln0 + frac*(ln1 - ln0))))
    return None


def cost_to_target_table(sys, x0, u_func, T, max_dofs=2000):
    print("="*72)
    print("DIAGNOSTIC 3: Cost-to-Target -- Uniform dG(0) vs DWR dG(0)")
    print("="*72)

    uni_grid_ns = [50, 75, 100, 150, 200, 300, 400, 600, 800,
                   1200, 1600, 2400, 3200]
    print("Uniform dG(0) sweep...")
    uni_ns, uni_qs = [], []
    for n in uni_grid_ns:
        q, _, _ = run_uniform(sys, x0, u_func, T, n)
        uni_ns.append(n); uni_qs.append(q)
        print(f"  N={n:5d}: QoI={q:.4e}")
    print()

    print("DWR dG(0) adaptive...")
    _, hist = dwr_adaptive(sys, x0, u_func, T, n_init=50,
                           max_iter=100, max_intervals=max_dofs, verbose=True)
    dwr_ns = [h["n_intervals"] for h in hist]
    dwr_qs = [h["qoi"]         for h in hist]
    print()

    data = {'Uniform': (uni_ns, uni_qs), 'DWR': (dwr_ns, dwr_qs)}

    col_w    = 14
    sep      = "  "
    hdr_line = sep.join([f"{'Target':>9}",
                          f"{'Uniform dG(0)':>{col_w}}",
                          f"{'DWR dG(0)':>{col_w}}"])
    print(hdr_line)
    print("-" * len(hdr_line))

    conv_min_q = max(uni_qs[-1], dwr_qs[-1])
    all_max_q  = min(uni_qs[0],  dwr_qs[0])
    targets    = [t for t in [1e4, 1e3, 1e2, 1e1, 1e0, 1e-1, 1e-2, 1e-3]
                  if conv_min_q * 1.1 < t < all_max_q * 0.9]

    table_rows = []
    for tgt in targets:
        n_uni = _find_n_for_target(uni_ns, uni_qs, tgt)
        n_dwr = _find_n_for_target(dwr_ns, dwr_qs, tgt)
        table_rows.append({"target": tgt, "Uniform": n_uni, "DWR": n_dwr})
        uni_s = str(n_uni) if n_uni is not None else f">{uni_grid_ns[-1]}"
        if n_dwr is None:
            dwr_s = f">{dwr_ns[-1]}"
        elif n_uni is not None:
            sav   = (1.0 - n_dwr / n_uni) * 100.0
            dwr_s = f"{n_dwr}({sav:+.0f}%)"
        else:
            dwr_s = str(n_dwr)
        print(sep.join([f"{tgt:9.0e}", f"{uni_s:>{col_w}}", f"{dwr_s:>{col_w}}"]))

    print()
    return data, table_rows


# =============================================================================
# DIAGNOSTIC 4: Convergence plot with fitted slopes  (dG(0))
# =============================================================================
def convergence_with_slopes(data):
    uni_ns, uni_qs = data['Uniform']
    dwr_ns, dwr_qs = data['DWR']

    mono_start = 0
    for k in range(len(uni_qs)-1, 0, -1):
        if uni_qs[k-1] > uni_qs[k]:
            mono_start = k - 1
            break
    fit_start  = min(mono_start, len(uni_ns) - 5)
    fit_ns     = np.array(uni_ns[fit_start:], dtype=float)
    fit_qs     = np.array(uni_qs[fit_start:], dtype=float)
    slope, icpt = np.polyfit(np.log(fit_ns), np.log(fit_qs), 1)
    ref_slope  = round(slope * 2) / 2.0
    n_ref      = np.array([fit_ns[0], fit_ns[-1]])
    q_ref      = np.exp(icpt) * n_ref**slope * 1.5
    print(f"  Uniform dG(0) fitted slope: O(N^{slope:.2f})")

    fig, ax = plt.subplots(figsize=(5.5, 3.5))
    ax.loglog(np.array(uni_ns, float), np.array(uni_qs, float), **MSTYLE['Uniform'])
    ax.loglog(np.array(dwr_ns, float), np.array(dwr_qs, float), **MSTYLE['DWR'])
    ax.loglog(n_ref, q_ref, color=_W['black'], ls=':', lw=1.0,
              label=fr"$O(N^{{{ref_slope:.1f}}})$")
    ax.set_xlabel(r"Number of intervals $N$", fontsize=14)
    ax.set_ylabel(r"Energy bal. viol.", fontsize=14)
    _grid(ax, 'both')
    ax.legend(loc='upper right', fontsize=11.0)
    fig.tight_layout()
    fig.savefig("fig_convergence_slopes.png", dpi=300)
    plt.close(fig)
    print("Saved: fig_convergence_slopes.png")


# =============================================================================
# PGFPlots data export
# Writes one CSV per plot curve so LaTeX can render plots directly.
# All files go into a pgfdata/ subdirectory.
# =============================================================================
def export_pgfplots_data(sys_obj, x0, u_func, T,
                         effectivity_rows, spectral_data,
                         convergence_data):
    import os, csv
    outdir = "pgfdata"
    os.makedirs(outdir, exist_ok=True)

    def write_csv(fname, header, rows):
        path = os.path.join(outdir, fname)
        with open(path, 'w', newline='') as f:
            w = csv.writer(f, delimiter='\t')
            w.writerow(header)
            w.writerows(rows)
        print(f"  Wrote {path}")

    # ------------------------------------------------------------------
    # 1. Effectivity index
    # ------------------------------------------------------------------
    write_csv("effectivity.csv",
              ["iter", "dofs", "true_err", "eta_est", "ieff"],
              [(r["iter"], r["dofs"], r["true_err"],
                r["eta_est"], r["ieff"])
               for r in effectivity_rows])

    # ------------------------------------------------------------------
    # 2. Spectral radius
    # ------------------------------------------------------------------
    t_uni,   rho_uni   = spectral_data["t_uni"],    spectral_data["rho_uni"]
    t_dwr,   rho_dwr   = spectral_data["t_dwr"],    spectral_data["rho_dwr"]
    k_uni,   k_dwr     = spectral_data["k_uni"],    spectral_data["k_dwr"]
    k_th,    rho_th    = spectral_data["k_theory"], spectral_data["rho_theory"]
    write_csv("spectral_vs_time_uniform.csv",
              ["t", "rho"],
              zip(t_uni, rho_uni))
    write_csv("spectral_vs_time_dwr.csv",
              ["t", "rho"],
              zip(t_dwr, rho_dwr))
    write_csv("spectral_vs_stepsize_uniform.csv",
              ["k", "rho"],
              zip(k_uni, rho_uni))
    write_csv("spectral_vs_stepsize_dwr.csv",
              ["k", "rho"],
              zip(k_dwr, rho_dwr))
    write_csv("spectral_theory.csv",
              ["k", "rho"],
              zip(k_th, rho_th))

    # ------------------------------------------------------------------
    # 3+4. Convergence (uniform + DWR)
    # ------------------------------------------------------------------
    uni_ns, uni_qs = convergence_data["Uniform"]
    dwr_ns, dwr_qs = convergence_data["DWR"]
    write_csv("convergence_uniform.csv",
              ["N", "J"],
              zip(uni_ns, uni_qs))
    write_csv("convergence_dwr.csv",
              ["N", "J"],
              zip(dwr_ns, dwr_qs))

    # Fitted reference slope (uniform tail)
    mono_start = 0
    for k in range(len(uni_qs)-1, 0, -1):
        if uni_qs[k-1] > uni_qs[k]:
            mono_start = k - 1; break
    fit_ns = np.array(uni_ns[mono_start:], float)
    fit_qs = np.array(uni_qs[mono_start:], float)
    slope, icpt = np.polyfit(np.log(fit_ns), np.log(fit_qs), 1)
    ref_slope   = round(slope * 2) / 2.0
    ref_ns = np.array([fit_ns[0], fit_ns[-1]])
    ref_qs = np.exp(icpt) * ref_ns**slope * 1.5
    write_csv("convergence_slope.csv",
              ["N", "J"],
              zip(ref_ns, ref_qs))


# =============================================================================
# Main
# =============================================================================
if __name__ == "__main__":
    MAT_PATH = "rcl_ladder_system.mat"

    print("Loading system from:", MAT_PATH)
    sys_obj = TransmissionLine_PH_DAE(mat_path=MAT_PATH)
    sys_obj.print_summary()

    T      = 10.0
    u_func = make_input()
    x0     = np.zeros(sys_obj.r)
    x0[0]  = 1.0

    eff_rows = effectivity_table(sys_obj, x0, u_func, T)

    # Run spectral radius and collect data for export
    print("="*72)
    print("DIAGNOSTIC 2: Spectral radius rho(G_i)")
    print("="*72)
    E11T = sys_obj.E11.T; ST = sys_obj.S_tilde.T
    def _rho_seq(grid):
        M = len(grid)-1; rv = np.zeros(M)
        for i in range(M):
            ki = grid[i+1]-grid[i]
            Gi = la.solve(E11T+ki*ST, E11T)
            rv[i] = np.max(np.abs(la.eigvals(Gi)))
        return 0.5*(grid[:-1]+grid[1:]), rv
    n_uni = 200; grid_uni = np.linspace(0, T, n_uni+1)
    t_uni, rho_uni = _rho_seq(grid_uni)
    _, hist = dwr_adaptive(sys_obj, x0, u_func, T, n_init=50,
                           max_iter=25, max_intervals=300)
    grid_dwr = hist[-1]["grid"]
    t_dwr, rho_dwr = _rho_seq(grid_dwr)
    eigvals_S = la.eigvalsh(sys_obj.S_tilde)
    lam_min_pos = eigvals_S[eigvals_S > 1e-12].min() if np.any(eigvals_S > 1e-12) else 0.0
    k_all = np.concatenate([np.diff(grid_uni), np.diff(grid_dwr)])
    k_th  = np.logspace(np.log10(k_all.min()), np.log10(k_all.max()), 400)
    rho_th = 1.0/(1.0 + k_th * lam_min_pos)
    spectral_data = dict(t_uni=t_uni, rho_uni=rho_uni,
                         t_dwr=t_dwr, rho_dwr=rho_dwr,
                         k_uni=np.diff(grid_uni), k_dwr=np.diff(grid_dwr),
                         k_theory=k_th, rho_theory=rho_th)
    spectral_radius_plot(sys_obj, x0, u_func, T)

    print("=== DIAGNOSTICS 3+4: dG(0) benchmark ===")
    data, _ = cost_to_target_table(sys_obj, x0, u_func, T, max_dofs=2000)
    print("=== DIAGNOSTIC 4: dG(0) convergence plot ===")
    convergence_with_slopes(data)

    print("All diagnostics complete.")