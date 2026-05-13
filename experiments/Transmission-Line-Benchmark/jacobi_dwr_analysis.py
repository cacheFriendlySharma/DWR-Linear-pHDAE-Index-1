"""
jacobi_dwr_analysis_v2.py
=========================
Updated Jacobi adjoint sweep analysis.


Produces:
  fig_J1_error_initial.png   — error vs sweeps k, initial grid
  fig_J2_error_late.png      — error vs sweeps k, late grid (14 DWR iters)
  fig_J3_ranked_initial.png  — ranked indicators, initial grid
  fig_J4_ranked_late.png     — ranked indicators, late grid
  fig_J5_scaling_study.png   — k* vs DWR iteration (multi-grid study)

System from rcl_ladder_system.mat (HDF5).
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

# =============================================================================
# Style  (Wong 2011 palette, Latin Modern Roman)
# =============================================================================
plt.rcParams.update({
    'font.family'        : 'serif',
    'font.serif'         : ['Latin Modern Roman', 'DejaVu Serif', 'serif'],
    'mathtext.fontset'   : 'cm',
    'font.size'          : 16,
    'axes.linewidth'     : 1.0,
    'axes.labelsize'     : 17,
    'axes.titlesize'     : 16,
    'axes.spines.top'    : False,
    'axes.spines.right'  : False,
    'xtick.labelsize'    : 14,
    'ytick.labelsize'    : 14,
    'xtick.direction'    : 'in',
    'ytick.direction'    : 'in',
    'xtick.major.width'  : 0.8,
    'ytick.major.width'  : 0.8,
    'xtick.major.size'   : 6,
    'ytick.major.size'   : 6,
    'xtick.major.pad'    : 7,
    'ytick.major.pad'    : 7,
    'axes.grid'          : False,
    'legend.fontsize'    : 13,
    'legend.frameon'     : True,
    'legend.framealpha'  : 0.93,
    'legend.edgecolor'   : '#cccccc',
    'legend.fancybox'    : False,
    'legend.borderpad'   : 0.6,
    'legend.handlelength': 2.2,
    'figure.dpi'         : 150,
    'savefig.dpi'        : 600,
    'savefig.bbox'       : 'tight',
    'savefig.pad_inches' : 0.12,
})

_W = {
    'blue'   : '#0072B2',
    'vermil' : '#D55E00',
    'green'  : '#009E73',
    'orange' : '#E69F00',
    'pink'   : '#CC79A7',
    'skyblue': '#56B4E9',
    'black'  : '#000000',
}

C_ETAERR      = _W['blue']
C_ZERR        = _W['skyblue']
C_AGREE       = _W['green']
C_EXACT       = _W['black']
RANKED_COLORS = [_W['vermil'], _W['orange'], _W['pink']]

OUTDIR = "."


def _add_grid(ax, which='major'):
    ax.grid(True, which='major', color='#cccccc', lw=0.35, alpha=0.55)
    if which == 'both':
        ax.grid(True, which='minor', color='#e0e0e0', lw=0.2, alpha=0.35)


# =============================================================================
# System
# =============================================================================
class TransmissionLine_PH_DAE:
    def __init__(self, mat_path='rcl_ladder_system.mat'):
        with h5py.File(mat_path, 'r') as f:
            E_full = f['E'][()].T
            J_full = f['J'][()].T
            R_full = f['R'][()].T
            Q_full = f['G'][()].T

        n = E_full.shape[0]
        self.n = n
        self.E = E_full
        self.J = J_full
        self.R = R_full
        self.Q = Q_full

        self.B      = np.zeros((n, 1))
        self.B[0,0] = 1.0

        diag_E = np.abs(np.diag(E_full))
        self.r = int(np.sum(diag_E > 1e-10))
        r      = self.r

        if np.where(diag_E > 1e-10)[0].max() != r - 1:
            raise ValueError("Differential DOFs not contiguous at front.")

        A   = (J_full - R_full) @ Q_full
        A11 = A[:r, :r];  A12 = A[:r, r:]
        A21 = A[r:, :r];  A22 = A[r:, r:]
        B1  = self.B[:r];  B2  = self.B[r:]

        A22_inv      = la.inv(A22)
        self.S       = -(A11 - A12 @ A22_inv @ A21)
        self.S_tilde =  0.5*(self.S + self.S.T)
        self.F       =  B1 - A12 @ A22_inv @ B2
        self.E11     = E_full[:r, :r]
        self.Q11     = Q_full[:r, :r]

        eigs = la.eigvalsh(self.S_tilde)
        pos_eigs = eigs[eigs > 1e-12]
        self.lam_min_pos = pos_eigs.min() if len(pos_eigs) else 0.0
        self.n_zero_eigs = int(np.sum(np.abs(eigs) < 1e-10))

        print(f"System loaded from: {mat_path}")
        print(f"  n={n}, r={r}, alg={n-r}")
        print(f"  lambda_min(S_tilde) = {eigs.min():.4e}  "
              f"(positive: {self.lam_min_pos:.4e}, "
              f"zero modes: {self.n_zero_eigs})")

    def H(self, x1):
        return 0.5 * float(x1 @ (self.E11.T @ self.Q11) @ x1)

    def grad_H(self, x1):
        return (self.E11.T @ self.Q11) @ x1


# =============================================================================
# Input
# =============================================================================
PULSE_T0    = 0.5
PULSE_SIGMA = 0.05
PULSE_AMP   = 50.0

def u_func(t):
    return np.array([PULSE_AMP * np.exp(-((t - PULSE_T0)**2) /
                                        (2 * PULSE_SIGMA**2))])


# =============================================================================
# Primal dG(0)
# =============================================================================
def solve_primal(sys, grid, x0):
    x = np.zeros((len(grid), sys.r))
    x[0] = x0
    for i in range(len(grid) - 1):
        dt = grid[i+1] - grid[i]
        tm = grid[i] + 0.5*dt
        x[i+1] = la.solve(sys.E11 + dt*sys.S,
                           sys.E11 @ x[i] + dt*sys.F @ u_func(tm))
    return x


# =============================================================================
# QoI residuals G_i
# =============================================================================
def compute_G(sys, grid, x):
    M    = len(grid) - 1
    Href = max(max(sys.H(x[i]) for i in range(len(x))), 1e-14)
    G    = np.zeros(M)
    for i in range(M):
        dt  = grid[i+1] - grid[i]
        tm  = grid[i] + 0.5*dt
        x1  = x[i+1]
        G[i] = (sys.H(x[i+1]) - sys.H(x[i])
                + dt * x1 @ sys.S_tilde @ x1
                - dt * u_func(tm) @ (sys.F.T @ x1)) / Href
    return G


# =============================================================================
# Adjoint RHS
# =============================================================================
def _adjoint_rhs(sys, grid, x, G):
    M    = len(grid) - 1
    Href = max(max(sys.H(x[i]) for i in range(len(x))), 1e-14)
    rhs  = np.zeros((M, sys.r))
    for i in range(M):
        dt  = grid[i+1] - grid[i]
        tm  = grid[i] + 0.5*dt
        gh  = sys.grad_H(x[i+1]) / Href
        rhs[i] = 2*G[i] * (gh
                            + 2*dt * sys.S_tilde @ x[i+1] / Href
                            - dt   * sys.F @ u_func(tm) / Href)
        if i < M-1:
            rhs[i] -= 2*G[i+1] * gh
    return rhs


# =============================================================================
# Exact sequential backward adjoint
# =============================================================================
def solve_adjoint_exact(sys, grid, x):
    M   = len(grid) - 1
    z   = np.zeros((M, sys.r))
    G   = compute_G(sys, grid, x)
    rhs = _adjoint_rhs(sys, grid, x, G)
    for i in reversed(range(M)):
        dt    = grid[i+1] - grid[i]
        local = rhs[i].copy()
        if i < M-1:
            local += sys.E11.T @ z[i+1]
        z[i] = la.solve(sys.E11.T + dt*sys.S.T, local)
    return z


# =============================================================================
# Jacobi (parallel) adjoint — k sweeps
# =============================================================================
def solve_adjoint_jacobi(sys, grid, x, k):
    M   = len(grid) - 1
    G   = compute_G(sys, grid, x)
    rhs = _adjoint_rhs(sys, grid, x, G)

    LUs = [la.lu_factor(sys.E11.T + (grid[i+1]-grid[i]) * sys.S.T)
           for i in range(M)]

    z = np.zeros((M, sys.r))
    for _ in range(k):
        z_new = np.zeros((M, sys.r))
        for i in range(M):
            local = rhs[i].copy()
            if i < M-1:
                local += sys.E11.T @ z[i+1]
            z_new[i] = la.lu_solve(LUs[i], local)
        z = z_new
    return z


# =============================================================================
# DWR indicator
# =============================================================================
def compute_eta(sys, grid, x, z):
    M   = len(grid) - 1
    eta = np.zeros(M)
    for i in range(M):
        dt  = grid[i+1] - grid[i]
        tm  = grid[i] + 0.5*dt
        Dz  = z[i] - (z[i-1] if i > 0 else np.zeros(sys.r))
        eta[i] = abs(
            0.5*dt * np.dot(sys.F @ u_func(tm) - sys.S @ x[i+1], Dz)
            + np.dot(sys.E11 @ (x[i+1] - x[i]), Dz)
        )
    return eta


# =============================================================================
# Dörfler marking and refinement
# =============================================================================
def dorfler_set(eta, theta):
    idx = np.argsort(eta)[::-1]
    cs  = np.cumsum(eta[idx])
    nm  = np.searchsorted(cs, theta * cs[-1]) + 1
    return frozenset(idx[:nm])


def refine_grid(grid, marked):
    new = [grid[0]]
    for i in range(len(grid) - 1):
        if i in marked:
            new.append(0.5*(grid[i] + grid[i+1]))
        new.append(grid[i+1])
    return np.array(new)


# =============================================================================
# Parameters — FIXED: N_INIT = 51 gives M = 50 intervals (paper says N = 50)
# =============================================================================
T             = 10.0
N_INIT        = 51       # <<< FIXED: 50 intervals, matching paper
THETA         = 0.5
N_LATE_ITERS  = 14
MAT_PATH      = 'rcl_ladder_system.mat'

sys_obj = TransmissionLine_PH_DAE(mat_path=MAT_PATH)
x0 = np.zeros(sys_obj.r); x0[0] = 1.0

k_mean      = T / (N_INIT - 1)
rho_theory  = 1.0 / (1.0 + k_mean * sys_obj.lam_min_pos)
print(f"\nT={T}, N_init={N_INIT}, M_init={N_INIT-1}, k_mean={k_mean:.4f}")
print(f"rho_theory (lam_min_pos={sys_obj.lam_min_pos:.4f}) = {rho_theory:.6f}")
print(f"Note: {sys_obj.n_zero_eigs} zero modes of S_tilde\n")


# =============================================================================
# Build grids at multiple DWR iterations for the scaling study
# =============================================================================
grid_init = np.linspace(0, T, N_INIT)

# Store grids at selected DWR iteration counts
STUDY_ITERS = [0, 2, 4, 6, 8, 10, 12, 14]
grids_at_iter = {0: grid_init.copy()}

print("Building grids via DWR refinement...")
grid = grid_init.copy()
for it in range(1, max(STUDY_ITERS) + 1):
    x      = solve_primal(sys_obj, grid, x0)
    z      = solve_adjoint_exact(sys_obj, grid, x)
    eta    = compute_eta(sys_obj, grid, x, z)
    marked = dorfler_set(eta, THETA)
    grid   = refine_grid(grid, marked)
    if it in STUDY_ITERS:
        grids_at_iter[it] = grid.copy()
        print(f"  DWR iter {it:>2}: M = {len(grid)-1}")

grid_late = grids_at_iter[N_LATE_ITERS]
print(f"\nInitial grid: M = {len(grid_init)-1}")
print(f"Late grid:    M = {len(grid_late)-1}\n")


# =============================================================================
# Analysis function
# =============================================================================
def analyze(g, label, verbose=True):
    x        = solve_primal(sys_obj, g, x0)
    z_ex     = solve_adjoint_exact(sys_obj, g, x)
    eta_ex   = compute_eta(sys_obj, g, x, z_ex)
    marked_ex = dorfler_set(eta_ex, THETA)
    nz       = np.linalg.norm(z_ex)
    neta     = np.linalg.norm(eta_ex)
    M        = len(g) - 1

    k_range = sorted(set(
        list(range(1, 21)) +
        list(range(25, 76, 5)) +
        list(range(80, 201, 10)) +
        list(range(220, 401, 20))
    ))
    k_range = [k for k in k_range if k <= M]

    z_errs, eta_errs, agrees = [], [], []

    if verbose:
        print(f"{label}  (M={M}, {len(marked_ex)} marked):")
        print(f"  {'k':>5}  {'z_err':>10}  {'η_err':>10}  {'agree%':>8}")

    for k in k_range:
        z_k   = solve_adjoint_jacobi(sys_obj, g, x, k)
        eta_k = compute_eta(sys_obj, g, x, z_k)
        mk    = dorfler_set(eta_k, THETA)

        ze = np.linalg.norm(z_k   - z_ex) / nz   if nz   > 0 else 0.
        ee = np.linalg.norm(eta_k - eta_ex) / neta if neta > 0 else 0.
        ag = 100 * len(mk & marked_ex) / len(marked_ex) if marked_ex else 100.

        z_errs.append(ze); eta_errs.append(ee); agrees.append(ag)

        if verbose and k in [1, 2, 3, 5, 10, 20, 30, 40, 50, 75, 100, 150, 200]:
            print(f"  {k:>5}  {ze:>10.4e}  {ee:>10.4e}  {ag:>8.1f}")

    # First k where agreement reaches 100% and holds for 4 consecutive
    k_agree = None
    for i in range(len(k_range) - 3):
        if all(a >= 99.9 for a in agrees[i:i+4]):
            k_agree = k_range[i]; break
    if k_agree is None and max(agrees) >= 99.9:
        k_agree = k_range[int(np.argmax(agrees))]

    speedup = M / k_agree if k_agree else None
    if verbose:
        print(f"  → k_agree={k_agree},  speedup="
              + (f"{speedup:.0f}×" if speedup else "—") + "\n")

    # Compute furthest marked interval index (propagation distance)
    if marked_ex:
        furthest_marked = min(marked_ex)  # smallest index = furthest from T
        prop_distance = M - 1 - furthest_marked
    else:
        furthest_marked = None
        prop_distance = None

    # Compute min step size (related to worst-case contraction)
    dts = np.diff(g)
    k_min_dt = dts.min()
    rho_worst = 1.0 / (1.0 + k_min_dt * sys_obj.lam_min_pos)

    return dict(
        g=g, x=x, z_ex=z_ex, eta_ex=eta_ex,
        marked_ex=marked_ex, M=M,
        k_range=k_range, z_errs=z_errs,
        eta_errs=eta_errs, agrees=agrees,
        k_agree=k_agree, speedup=speedup,
        sort_idx=np.argsort(eta_ex)[::-1],
        n_show=min(100, M), label=label,
        p=len(marked_ex),
        prop_distance=prop_distance,
        furthest_marked=furthest_marked,
        k_min_dt=k_min_dt,
        rho_worst=rho_worst,
    )


# =============================================================================
# Run main analyses (initial + late grids, verbose)
# =============================================================================
d_init = analyze(grid_init, "Initial grid")
d_late = analyze(grid_late, f"Late grid ({N_LATE_ITERS} DWR iters)")


# =============================================================================
# Systematic multi-grid study
# =============================================================================
print("=" * 70)
print("Systematic scaling study: k* vs DWR iteration")
print("=" * 70)
print(f"  {'iter':>4}  {'M':>5}  {'|M_ex|':>6}  {'k*':>5}  {'speedup':>8}  "
      f"{'k*/p':>6}  {'prop_d':>6}  {'k_min':>8}  {'rho_worst':>9}")
print("-" * 70)

scaling_data = []
for it in STUDY_ITERS:
    g = grids_at_iter[it]
    lbl = f"DWR iter {it}"
    d = analyze(g, lbl, verbose=False)
    scaling_data.append(d)

    kp = d['k_agree'] / d['p'] if (d['k_agree'] and d['p']) else None
    sp_str = f"{d['speedup']:.0f}×" if d['speedup'] else "—"
    kp_str = f"{kp:.1f}" if kp else "—"
    ka_str = str(d['k_agree']) if d['k_agree'] else "—"
    pd_str = str(d['prop_distance']) if d['prop_distance'] is not None else "—"

    print(f"  {it:>4}  {d['M']:>5}  {d['p']:>6}  {ka_str:>5}  {sp_str:>8}  "
          f"{kp_str:>6}  {pd_str:>6}  {d['k_min_dt']:>8.4e}  "
          f"{d['rho_worst']:>9.4f}")

print()


# =============================================================================
# Figure helpers
# =============================================================================
def _savefig(fig, fname):
    path = os.path.join(OUTDIR, fname)
    fig.savefig(path, dpi=300)
    plt.close(fig)
    print(f"Saved: {path}")


def fig_error(d, fname):
    fig, ax1 = plt.subplots(figsize=(6.0, 5.0))
    ax2 = ax1.twinx()

    l1, = ax1.semilogy(
        d['k_range'], d['eta_errs'],
        color=C_ETAERR, lw=2.5, ls='-', marker='o', ms=7, mew=0.7,
        mec='white', markevery=3,
        label=r'$\|\eta^{(k)}-\eta_{\rm ex}\|/\|\eta_{\rm ex}\|$')
    l2, = ax1.semilogy(
        d['k_range'], d['z_errs'],
        color=C_ZERR, lw=2.0, ls='--', marker='s', ms=6, mew=0.7,
        mec='white', markevery=3,
        label=r'$\|z^{(k)}-z_{\rm ex}\|/\|z_{\rm ex}\|$')

    ax1.set_xlabel(r'Jacobi sweeps $k$', labelpad=7, fontsize=14)
    ax1.set_ylabel('Relative error', labelpad=7, fontsize=14)
    ax1.tick_params(axis='both', labelsize=12)
    _add_grid(ax1, 'both')

    l3, = ax2.plot(
        d['k_range'], d['agrees'],
        color=C_AGREE, lw=2.8, ls='-', marker='^', ms=7, mew=0.7,
        mec='white', markevery=3,
        label='Agreement (%)')

    ax2.axhline(100, color=C_AGREE, lw=1.0, ls=':', alpha=0.45)
    ax2.set_ylim(-5, 115)
    ax2.set_ylabel('Marked-set agreement (%)', color=C_AGREE, labelpad=15, fontsize=14)
    ax2.tick_params(axis='y', labelcolor=C_AGREE, labelsize=12)
    ax2.spines['top'].set_visible(False)

    if d['k_agree']:
        ax1.axvline(d['k_agree'], color='#888888', lw=1.5, ls='--', alpha=0.7, zorder=1)
        ax1.text(d['k_agree'] + 2.0, 0.97, rf"$k^*\!=\!{d['k_agree']}$",
                 transform=ax1.get_xaxis_transform(),
                 fontsize=14, color='#444444', va='top')

    lines_leg = [l1, l2, l3]
    labels    = [l.get_label() for l in lines_leg]
    ax1.legend(lines_leg, labels,
               loc='lower center',
               fontsize=11,
               handlelength=2.0,
               labelspacing=2.5,
               borderpad=0.8,
               handleheight=1.2,
               framealpha=0.93,
               edgecolor='#cccccc')

    fig.tight_layout(rect=[0, 0, 0.95, 1])
    _savefig(fig, fname)


# =============================================================================
# Figures 3 & 4: Ranked indicator η_i
# =============================================================================
def fig_ranked(d, k_vals, fname):
    n      = d['n_show']
    idx    = d['sort_idx']
    eta_ex = d['eta_ex']
    rank   = np.arange(1, n + 1)

    cf  = np.cumsum(eta_ex[idx]) / eta_ex[idx].sum()
    cut = min(int(np.searchsorted(cf[:n], THETA)) + 1, n - 1)

    fig, ax = plt.subplots(figsize=(6.0, 5.0))

    ax.axvspan(0.5, cut + 0.5, alpha=0.07, color='#aaaaaa', zorder=0)
    ax.axvline(cut, color='#aaaaaa', lw=1.0, ls=':', alpha=0.9, zorder=1)

    ax.semilogy(rank, eta_ex[idx[:n]], color=C_EXACT, lw=2.5, ls='-',
                label=r'Exact', zorder=5)

    for k, col in zip(k_vals, RANKED_COLORS):
        z_k   = solve_adjoint_jacobi(sys_obj, d['g'], d['x'], k)
        eta_k = compute_eta(sys_obj, d['g'], d['x'], z_k)
        ax.semilogy(rank, eta_k[idx[:n]], color=col, lw=1.8, ls='--', alpha=0.88,
                    label=rf'$k = {k}$', zorder=4)

    ylo, yhi = ax.get_ylim()
    ax.text(cut + 1.5, np.exp(0.82*np.log(yhi) + 0.18*np.log(ylo)),
            r'$\theta$-cut', fontsize=12, color='#555555', va='top')

    ax.set_xlabel(r'Interval rank (sorted by exact $|\eta_i|$)', fontsize=14, labelpad=6)
    ax.set_ylabel(r'DWR indicator $|\eta_i|$', fontsize=14, labelpad=6)
    ax.tick_params(axis='both', labelsize=12)
    _add_grid(ax, 'both')

    ax.legend(loc='lower left', fontsize=13,
              handlelength=2.2, borderpad=0.6,
              labelspacing=0.4, framealpha=0.93,
              edgecolor='#cccccc')

    fig.tight_layout(pad=1.0)
    _savefig(fig, fname)


# =============================================================================
# Figure 5: Scaling study — k* and speedup vs DWR iteration
# =============================================================================
def fig_scaling(scaling_data, fname):
    iters    = [STUDY_ITERS[i] for i in range(len(scaling_data))]
    Ms       = [d['M'] for d in scaling_data]
    ps       = [d['p'] for d in scaling_data]
    ks       = [d['k_agree'] if d['k_agree'] else np.nan for d in scaling_data]
    speedups = [d['speedup'] if d['speedup'] else np.nan for d in scaling_data]

    fig, ax1 = plt.subplots(figsize=(7.0, 5.0))
    ax2 = ax1.twinx()

    l1, = ax1.plot(iters, ks, color=_W['blue'], lw=2.5, marker='o', ms=8,
                   mew=0.7, mec='white', label=r'$k^*$')
    l2, = ax1.plot(iters, ps, color=_W['vermil'], lw=2.0, ls='--', marker='s',
                   ms=7, mew=0.7, mec='white', label=r'$|\mathcal{M}_{\rm ex}|$')
    l3, = ax1.plot(iters, Ms, color=_W['orange'], lw=1.5, ls=':', marker='^',
                   ms=6, mew=0.7, mec='white', label=r'$M$')

    l4, = ax2.plot(iters, speedups, color=_W['green'], lw=2.5, marker='D',
                   ms=7, mew=0.7, mec='white', label='Speedup')

    ax1.set_xlabel('DWR refinement iteration', fontsize=14, labelpad=7)
    ax1.set_ylabel(r'$k^*$, $|\mathcal{M}_{\rm ex}|$, $M$', fontsize=14, labelpad=7)
    ax2.set_ylabel('Speedup ($M/k^*$)', color=_W['green'], fontsize=14, labelpad=9)
    ax2.tick_params(axis='y', labelcolor=_W['green'])
    ax2.spines['top'].set_visible(False)
    _add_grid(ax1)

    lines  = [l1, l2, l3, l4]
    labels = [l.get_label() for l in lines]
    fig.legend(lines, labels, loc='upper center',
               bbox_to_anchor=(0.5, 1.02), ncol=4, fontsize=12,
               handlelength=2.0, columnspacing=1.2, borderpad=0.4,
               framealpha=0.95, edgecolor='#cccccc')

    fig.tight_layout(pad=1.4)
    fig.subplots_adjust(top=0.88)
    _savefig(fig, fname)


# =============================================================================
# Generate all figures
# =============================================================================
fig_error(d_init, 'fig_J1_error_initial.png')
fig_error(d_late, 'fig_J2_error_late.png')
fig_ranked(d_init, k_vals=[1, 3, 5],   fname='fig_J3_ranked_initial.png')
fig_ranked(d_late, k_vals=[5, 20, 40], fname='fig_J4_ranked_late.png')
fig_scaling(scaling_data, 'fig_J5_scaling_study.png')


# =============================================================================
# Console summary
# =============================================================================
print("\n" + "=" * 70)
print("Summary")
print("=" * 70)
for d in [d_init, d_late]:
    sp = f"{d['speedup']:.0f}×" if d['speedup'] else "—"
    print(f"  {d['label']:<36}  M={d['M']:>5},  "
          f"|M_ex|={d['p']:>4},  "
          f"k*={str(d['k_agree']):>4},  speedup={sp}")
    print(f"    prop_distance={d['prop_distance']},  "
          f"k_min_dt={d['k_min_dt']:.4e},  "
          f"rho_worst={d['rho_worst']:.4f}")
print()
print(f"rho_theory = {rho_theory:.6f}  "
      f"(lam_min_pos={sys_obj.lam_min_pos:.4f})")
if d_init['k_agree']:
    print(f"rho^k* (initial) = {rho_theory**d_init['k_agree']:.4f}")
if d_late['k_agree']:
    print(f"rho^k* (late)    = {rho_theory**d_late['k_agree']:.4f}")

print("\nDone. Check outputs:")
print("  Figures: fig_J1..J5_*.png")