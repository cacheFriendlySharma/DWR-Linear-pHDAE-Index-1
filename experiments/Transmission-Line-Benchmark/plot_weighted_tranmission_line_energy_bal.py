"""
Energy Balance Violation: weighted vs unweighted adaptive QoI
=============================================================
System loaded from: rcl_ladder_system.mat  (MATLAB v7.3 / HDF5)
  Same benchmark as diagnostics_matfile.py.

Algorithm notes
---------------
* Primal dG(0): full S in system matrix (E11 + dt*S).
* QoI / G_i: uses S_tilde only. Justified by Lemma: for any skew-symmetric
  matrix K and any vector x, x^T K x = 0 identically (transpose argument).
* Adjoint RHS: d/dx1[x1^T S_tilde x1] = 2 S_tilde x1.
* Weighted adjoint (rho > 0): adds rho*dt*grad_H to bias refinement toward
  high-energy intervals.
* Indicators eta_i are SIGNED (no elementwise abs).
  - Stopping criterion : eta_tot = |sum_i eta_i|   (signed sum, then abs)
  - Dorfler marking    : sum_{M} |eta_i| >= theta * sum_i |eta_i|  (elementwise abs)
  This faithfully implements Algorithm 1 of the paper.

Plotting style: Wong (2011) colorblind-safe, Latin Modern Roman, all inline.
"""

import os
import h5py
for _var in ("OMP_NUM_THREADS","OPENBLAS_NUM_THREADS","MKL_NUM_THREADS",
             "VECLIB_MAXIMUM_THREADS","NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_var,"1")

import numpy as np
import scipy.linalg as la
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ==============================================================================
# Inline style  (Wong 2011 palette, Latin Modern Roman)
# ==============================================================================
plt.rcParams.update({
    'font.family':'serif',
    'font.serif':['Latin Modern Roman','DejaVu Serif','serif'],
    'mathtext.fontset':'cm','font.size':12,
    'axes.linewidth':0.8,'axes.labelsize':13,'axes.titlesize':13,
    'axes.spines.top':False,'axes.spines.right':False,
    'xtick.labelsize':11,'ytick.labelsize':11,
    'xtick.direction':'in','ytick.direction':'in',
    'xtick.major.width':0.6,'ytick.major.width':0.6,
    'xtick.minor.width':0.4,'ytick.minor.width':0.4,
    'xtick.major.size':5,'ytick.major.size':5,
    'xtick.minor.size':3,'ytick.minor.size':3,
    'xtick.major.pad':5,'ytick.major.pad':5,
    'axes.grid':False,
    'legend.fontsize':9.5,'legend.frameon':True,'legend.framealpha':0.92,
    'legend.edgecolor':'#cccccc','legend.fancybox':False,
    'legend.borderpad':0.5,'legend.handlelength':2.0,
    'figure.dpi':150,'savefig.dpi':300,
    'savefig.bbox':'tight','savefig.pad_inches':0.08,
})
_W = {
    'blue'   :'#0072B2', 'vermil' :'#D55E00',
    'skyblue':'#56B4E9', 'green'  :'#009E73',
    'pink'   :'#CC79A7',
}

def add_grid(ax, which='major'):
    ax.grid(True, which='major', color='#cccccc', lw=0.3, alpha=0.5)
    if which == 'both':
        ax.grid(True, which='minor', color='#e0e0e0', lw=0.2, alpha=0.3)

# ==============================================================================
# System -- loaded from .mat file (MATLAB v7.3 / HDF5)
# ==============================================================================
class TransmissionLine_PH_DAE:
    """
    pH-DAE loaded from rcl_ladder_system.mat.

    File layout (HDF5 keys):
      E  -- descriptor matrix          (n x n)
      J  -- skew-symmetric structure   (n x n)
      R  -- symmetric PSD dissipation  (n x n)
      G  -- Hamiltonian weighting Q    (n x n, = I here)

    pH form:   E x' = (J - R) Q x + B u
    Input B  : canonical first column e_1 (not stored in file)
    Partition: first r rows/cols are differential (diag(E) > 0),
               remaining n-r are algebraic.
    """

    def __init__(self, mat_path='rcl_ladder_system.mat'):
        with h5py.File(mat_path, 'r') as f:
            # MATLAB stores in column-major order; transpose to row-major
            E_full = f['E'][()].T
            J_full = f['J'][()].T
            R_full = f['R'][()].T
            Q_full = f['G'][()].T   # key 'G' in file = Q (Hamiltonian weight matrix)

        n = E_full.shape[0]
        self.n = n

        # pH structure checks (warn only, do not abort)
        if not np.allclose(J_full, -J_full.T, atol=1e-10):
            print("  WARNING: J is not exactly skew-symmetric")
        if not np.allclose(R_full, R_full.T, atol=1e-10):
            print("  WARNING: R is not exactly symmetric")

        self.E = E_full; self.J = J_full
        self.R = R_full; self.Q = Q_full

        # Input: single voltage source at node 0
        self.B      = np.zeros((n, 1))
        self.B[0,0] = 1.0

        # Identify differential variables: those with diag(E) > 0
        diag_E = np.abs(np.diag(E_full))
        self.r = int(np.sum(diag_E > 1e-10))
        r      = self.r

        # Schur complement onto differential block
        A   = (J_full - R_full) @ Q_full
        A11 = A[:r, :r];  A12 = A[:r, r:]
        A21 = A[r:, :r];  A22 = A[r:, r:]
        B1  = self.B[:r];  B2  = self.B[r:]

        A22_inv      = la.inv(A22)
        self.S       = -(A11 - A12 @ A22_inv @ A21)  # full Schur complement
        self.S_tilde =  0.5*(self.S + self.S.T)       # symmetric part for QoI
        self.F       =  B1 - A12 @ A22_inv @ B2       # reduced input vector
        self.E11     = E_full[:r, :r]
        self.Q11     = Q_full[:r, :r]

        print(f"System loaded: n={n}, r={r} differential, {n-r} algebraic")
        lam_S = la.eigvalsh(self.S_tilde)
        print(f"  lambda(S_tilde): min={lam_S.min():.4e}, max={lam_S.max():.4e}")

    def hamiltonian(self, x1):
        return 0.5 * float(x1 @ (self.E11.T @ self.Q11) @ x1)

    def grad_hamiltonian(self, x1):
        return (self.E11.T @ self.Q11) @ x1


# ==============================================================================
# Input: Gaussian voltage pulse centred at t=0.5
# ==============================================================================
def make_input():
    def u(t):
        return np.array([50.0 * np.exp(-((t - 0.5)**2) / (2 * 0.05**2))])
    return u


# ==============================================================================
# Primal dG(0)
# ==============================================================================
def solve_primal(sys, grid, x0, u_func):
    x = np.zeros((len(grid), sys.r)); x[0] = x0
    for i in range(len(grid)-1):
        dt   = grid[i+1]-grid[i]; tmid = grid[i]+0.5*dt
        x[i+1] = la.solve(sys.E11 + dt*sys.S,
                           sys.E11 @ x[i] + dt*sys.F @ u_func(tmid))
    return x


# ==============================================================================
# QoI residuals G_i  (midpoint rule, S_tilde only -- skew part vanishes)
# ==============================================================================
def compute_G(sys, grid, x, u_func):
    M = len(grid)-1; G = np.zeros(M)
    for i in range(M):
        dt = grid[i+1]-grid[i]; x1 = x[i+1]; tmid = grid[i]+0.5*dt
        G[i] = (sys.hamiltonian(x[i+1]) - sys.hamiltonian(x[i])
                + dt*(x1 @ sys.S_tilde @ x1
                      - u_func(tmid) @ (sys.F.T @ x1)))
    return G

def energy_balance_violation(sys, grid, x, u_func):
    return float(np.sum(compute_G(sys, grid, x, u_func)**2))


# ==============================================================================
# Adjoint backward solve
# rho > 0: weighted variant -- adds rho*dt*grad_H to bias toward high-energy
# ==============================================================================
def solve_adjoint(sys, grid, x, u_func, rho=0.0):
    M = len(grid)-1; G = compute_G(sys, grid, x, u_func)
    z = np.zeros((M, sys.r))
    for i in reversed(range(M)):
        dt = grid[i+1]-grid[i]; tmid = grid[i]+0.5*dt; x1 = x[i+1]
        gH  = sys.grad_hamiltonian(x1)
        rhs = 2*G[i]*(gH + 2*dt*sys.S_tilde @ x1 - dt*(sys.F @ u_func(tmid)))
        if rho > 0.0:
            rhs += rho * dt * gH          # weighted: bias toward high-H intervals
        if i < M-1:
            rhs -= 2*G[i+1]*gH
            rhs += sys.E11.T @ z[i+1]
        z[i] = la.solve(sys.E11.T + dt*sys.S.T, rhs)
    return z


# ==============================================================================
# DWR error indicator -- SIGNED, no elementwise abs
# eta_tot = |sum_i eta_i|  (signed sum, then single abs) for stopping
# Dorfler uses |eta_i| elementwise for marking
# ==============================================================================
def compute_eta(sys, grid, x, z, u_func):
    eta = np.zeros(len(grid)-1)
    for i in range(len(grid)-1):
        dt = grid[i+1]-grid[i]; tmid = grid[i]+0.5*dt
        res = sys.F @ u_func(tmid) - sys.S @ x[i+1]
        dz  = z[i] if i == 0 else z[i] - z[i-1]
        eta[i] = (dt/2)*res @ dz + (sys.E11 @ (x[i+1]-x[i])) @ dz
    return eta


# ==============================================================================
# Dorfler marking -- operates on |eta_i|
# min_frac: always mark at least this fraction of intervals
# ==============================================================================
def dorfler_marking(eta, theta=0.5, min_frac=0.05):
    abs_eta   = np.abs(eta)
    idx       = np.argsort(abs_eta)[::-1]
    cumsum    = np.cumsum(abs_eta[idx])
    n_dorfler = int(np.searchsorted(cumsum, theta*cumsum[-1])) + 1
    n_min     = max(int(np.ceil(min_frac * len(eta))), 1)
    return set(idx[:max(n_dorfler, n_min)])

def refine(grid, marked):
    new = [grid[0]]
    for i in range(len(grid)-1):
        if i in marked: new.append(0.5*(grid[i]+grid[i+1]))
        new.append(grid[i+1])
    return np.array(new)


# ==============================================================================
# Adaptive refinement loop
# ==============================================================================
def adaptive_curve(sys, x0, u_func, T, max_dofs, rho=0.0,
                   n_init=50, theta=0.5, min_frac=0.05, max_iter=300):
    grid = np.linspace(0, T, n_init+1)
    n_list, ev_list = [], []
    for it in range(max_iter):
        x       = solve_primal(sys, grid, x0, u_func)
        z       = solve_adjoint(sys, grid, x, u_func, rho=rho)
        eta     = compute_eta(sys, grid, x, z, u_func)
        n_dofs  = len(grid)-1
        ev      = energy_balance_violation(sys, grid, x, u_func)
        eta_tot = abs(np.sum(eta))
        n_list.append(n_dofs); ev_list.append(ev)
        print(f"  iter {it:3d}: N={n_dofs:5d}, EBV={ev:.3e}, eta_tot={eta_tot:.3e}")
        if n_dofs >= max_dofs: break
        marked = dorfler_marking(eta, theta=theta, min_frac=min_frac)
        grid   = refine(grid, marked)
    return np.array(n_list), np.array(ev_list)


# ==============================================================================
# Main
# ==============================================================================
def main():
    MAT_PATH = 'rcl_ladder_system.mat'
    T        = 10.0
    n_init   = 50     # initial uniform grid size
    max_dofs = 2000   # stop adaptive loop here
    rho      = 10.0   # weight for the weighted adjoint

    # Load system and set initial condition
    sys_obj  = TransmissionLine_PH_DAE(mat_path=MAT_PATH)
    u_func   = make_input()
    x0       = np.zeros(sys_obj.r)
    x0[0]    = 1.0    # unit initial voltage at the first capacitor node

    # Uniform refinement sweep
    uni_grid_ns = [50, 75, 100, 150, 200, 300, 400, 600, 800, 1200, 1600, 2000]
    print("\n--- Uniform refinement ---")
    n_uni, ev_uni = [], []
    for n in uni_grid_ns:
        grid = np.linspace(0, T, n+1)
        x    = solve_primal(sys_obj, grid, x0, u_func)
        ev   = energy_balance_violation(sys_obj, grid, x, u_func)
        n_uni.append(n); ev_uni.append(ev)
        print(f"  N={n:5d}: EBV={ev:.3e}")
    n_uni  = np.array(n_uni)
    ev_uni = np.array(ev_uni)

    # Adaptive: unweighted (rho=0)
    print("\n--- Adaptive unweighted (rho=0) ---")
    n_unw, ev_unw = adaptive_curve(
        sys_obj, x0, u_func, T, max_dofs,
        rho=0.0, n_init=n_init)

    # Adaptive: weighted (rho > 0)
    print(f"\n--- Adaptive weighted (rho={rho}) ---")
    n_w, ev_w = adaptive_curve(
        sys_obj, x0, u_func, T, max_dofs,
        rho=rho, n_init=n_init)

    # Plot
    fig, ax = plt.subplots(figsize=(5.5, 3.5))
    ax.loglog(n_uni, ev_uni,
              color=_W['skyblue'], marker='o', ls='--', lw=1.6, ms=5,
              mec='white', mew=0.6, label='Uniform dG(0)')
    ax.loglog(n_unw, ev_unw,
              color=_W['blue'], marker='s', ls='-', lw=1.8, ms=5,
              mec='white', mew=0.6,
              label=r'DWR adaptive ($\rho=0$)')
    ax.loglog(n_w, ev_w,
              color=_W['vermil'], marker='^', ls='-', lw=1.8, ms=6,
              mec='white', mew=0.7,
              label=rf'DWR adaptive ($\rho={rho}$)')
    ax.set_xlabel(r'Number of intervals $N$')
    ax.set_ylabel(r'Energy bal. viol.')
    add_grid(ax, 'both')
    ax.legend(loc='lower left', fontsize=10)
    fig.tight_layout()
    fname = 'fig_weighted_vs_unweighted.png'
    fig.savefig(fname, dpi=300)
    plt.close(fig)
    print(f"\nSaved: {fname}")


if __name__ == "__main__":
    main()