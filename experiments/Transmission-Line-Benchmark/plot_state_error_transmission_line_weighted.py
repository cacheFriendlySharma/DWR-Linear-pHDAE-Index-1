"""
State error in energy norm: weighted vs unweighted adaptive QoI
===============================================================
System loaded from: rcl_ladder_system.mat  (MATLAB v7.3 / HDF5)
Same benchmark as diagnostics_matfile.py.

Algorithm notes
---------------
* Primal dG(0): uses full S in the system matrix (E11 + dt*S).
* QoI / G_i: uses S_tilde only. Justified by Lemma: for any skew-symmetric
  K and any vector x, x^T K x = 0 identically (transpose argument).
  Therefore G_full = G_sym exactly for dG(0), independent of mesh size.
* Adjoint RHS: d/dx1[x1^T S_tilde x1] = 2 S_tilde x1.
* Weighted adjoint (rho > 0): adds rho*dt*grad_H to the adjoint RHS,
  biasing refinement toward intervals where the Hamiltonian gradient is large
  (i.e. high-energy regions), which improves state-error convergence.

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
    'blue'   : '#0072B2', 'vermil' : '#D55E00',
    'skyblue': '#56B4E9', 'green'  : '#009E73',
    'pink'   : '#CC79A7',
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
            Q_full = f['G'][()].T   # key 'G' in file = Q (Hamiltonian weight, = I)

        n = E_full.shape[0]
        self.n = n

        # pH structure checks (warn only)
        if not np.allclose(J_full, -J_full.T, atol=1e-10):
            print("  WARNING: J is not exactly skew-symmetric")
        if not np.allclose(R_full, R_full.T, atol=1e-10):
            print("  WARNING: R is not exactly symmetric")

        self.E = E_full; self.J = J_full
        self.R = R_full; self.Q = Q_full

        # Input: single voltage source at node 0
        self.B      = np.zeros((n, 1))
        self.B[0,0] = 1.0

        # Identify differential variables: diag(E) > 0
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
        # H(x1) = 0.5 * x1^T E11^T Q11 x1
        # For this benchmark E11 = Q11 = I, so H = 0.5 * ||x1||^2
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


# ==============================================================================
# Adjoint backward solve
# rho > 0: weighted -- adds rho*dt*grad_H to bias toward high-energy intervals
# ==============================================================================
def solve_adjoint(sys, grid, x, u_func, rho=0.0):
    M = len(grid)-1; G = compute_G(sys, grid, x, u_func)
    z = np.zeros((M, sys.r))
    for i in reversed(range(M)):
        dt = grid[i+1]-grid[i]; tmid = grid[i]+0.5*dt; x1 = x[i+1]
        gH  = sys.grad_hamiltonian(x1)
        rhs = 2*G[i]*(gH + 2*dt*sys.S_tilde @ x1 - dt*(sys.F @ u_func(tmid)))
        if rho > 0.0:
            rhs += rho * dt * gH    # weighted: bias toward high-H intervals
        if i < M-1:
            rhs -= 2*G[i+1]*gH
            rhs += sys.E11.T @ z[i+1]
        z[i] = la.solve(sys.E11.T + dt*sys.S.T, rhs)
    return z


# ==============================================================================
# DWR error indicator -- SIGNED, no elementwise abs
# eta_tot = |sum_i eta_i|  for stopping
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
# State error in energy norm
# ||e||_E = sqrt( sum_i dt * H(x_i - x_ref(t_i)) )
# x_ref is evaluated by nearest-grid-point lookup on the reference grid
# ==============================================================================
def state_error_energy(sys, grid, x, grid_ref, x_ref):
    err = 0.0
    for i in range(len(grid)-1):
        dt = grid[i+1] - grid[i]
        # Find the index in grid_ref closest to grid[i]
        j  = min(np.searchsorted(grid_ref, grid[i]), len(grid_ref)-1)
        err += dt * sys.hamiltonian(x[i] - x_ref[j])
    return np.sqrt(err)


# ==============================================================================
# Dorfler marking -- operates on |eta_i|
# ==============================================================================
def dorfler_marking(eta, theta=0.5):
    abs_eta = np.abs(eta)
    idx     = np.argsort(abs_eta)[::-1]
    cumsum  = np.cumsum(abs_eta[idx])
    n_mark  = np.searchsorted(cumsum, theta*cumsum[-1]) + 1
    return set(idx[:n_mark])

def refine(grid, marked):
    new = [grid[0]]
    for i in range(len(grid)-1):
        if i in marked: new.append(0.5*(grid[i]+grid[i+1]))
        new.append(grid[i+1])
    return np.array(new)


# ==============================================================================
# Adaptive refinement loop
# ==============================================================================
def adaptive_curve(sys, x0, u_func, T, grid_ref, x_ref,
                   rho=0.0, n_init=50, max_dofs=2000,
                   theta=0.5, max_iter=300):
    grid = np.linspace(0, T, n_init+1)
    n_list, err_list = [], []
    for it in range(max_iter):
        x       = solve_primal(sys, grid, x0, u_func)
        z       = solve_adjoint(sys, grid, x, u_func, rho=rho)
        eta     = compute_eta(sys, grid, x, z, u_func)
        n_dofs  = len(grid)-1
        err     = state_error_energy(sys, grid, x, grid_ref, x_ref)
        eta_tot = abs(np.sum(eta))
        n_list.append(n_dofs); err_list.append(err)
        print(f"  iter {it:3d}: N={n_dofs:5d}, err={err:.3e}, eta_tot={eta_tot:.3e}")
        if n_dofs >= max_dofs: break
        marked = dorfler_marking(eta, theta=theta)
        grid   = refine(grid, marked)
    return np.array(n_list), np.array(err_list)


# ==============================================================================
# Main
# ==============================================================================
def main():
    MAT_PATH = 'rcl_ladder_system.mat'
    T        = 10.0
    n_init   = 50       # initial uniform grid
    n_ref    = 10000    # reference solution grid (fine enough to be "exact")
    max_dofs = 2000     # stop adaptive loop here
    rho      = 10.0     # weight for the weighted adjoint

    # Load system
    sys_obj = TransmissionLine_PH_DAE(mat_path=MAT_PATH)
    u_func  = make_input()
    x0      = np.zeros(sys_obj.r)
    x0[0]   = 1.0    # unit initial voltage at first capacitor node

    # Compute reference solution on a very fine grid
    print(f"\nComputing reference solution on N={n_ref} uniform grid...")
    grid_ref = np.linspace(0, T, n_ref+1)
    x_ref    = solve_primal(sys_obj, grid_ref, x0, u_func)
    print("  Done.")

    # Adaptive: unweighted (rho=0)
    print("\n--- Adaptive unweighted (rho=0) ---")
    n_unw, err_unw = adaptive_curve(
        sys_obj, x0, u_func, T, grid_ref, x_ref,
        rho=0.0, n_init=n_init, max_dofs=max_dofs)

    # Adaptive: weighted (rho > 0)
    print(f"\n--- Adaptive weighted (rho={rho}) ---")
    n_w, err_w = adaptive_curve(
        sys_obj, x0, u_func, T, grid_ref, x_ref,
        rho=rho, n_init=n_init, max_dofs=max_dofs)

    # Savings table
    print(f"\n{'Target error':<16} | {'N unweighted':<14} | {'N weighted':<12} | Savings")
    print("-"*65)
    for tgt in [1e0, 1e-1, 1e-2, 1e-3]:
        nu = next((n_unw[i] for i in range(len(err_unw)) if err_unw[i] <= tgt), None)
        nw = next((n_w[i]   for i in range(len(err_w))   if err_w[i]   <= tgt), None)
        tag = lambda v: str(v) if v else 'not reached'
        if nu and nw:
            print(f"{tgt:<16.0e} | {nu:<14} | {nw:<12} | {(1-nw/nu)*100:.1f}%")
        else:
            print(f"{tgt:<16.0e} | {tag(nu):<14} | {tag(nw):<12} |")

    # Plot
    fig, ax = plt.subplots(figsize=(5.5, 3.5))
    ax.loglog(n_unw, err_unw,
              color=_W['blue'],   marker='o', ls='--', lw=1.8, ms=5,
              mec='white', mew=0.6,
              label=r'DWR adaptive ($\rho=0$)')
    ax.loglog(n_w, err_w,
              color=_W['vermil'], marker='s', ls='-',  lw=1.8, ms=5,
              mec='white', mew=0.6,
              label=rf'DWR adaptive ($\rho={rho}$)')
    ax.set_xlabel(r'Number of intervals $N$')
    ax.set_ylabel(r'State error $\|e\|_{\mathcal{E}}$')
    add_grid(ax, 'both')
    ax.legend(loc='upper right', fontsize = 7)
    fig.tight_layout()
    fname = 'fig_state_error_weighted.png'
    fig.savefig(fname, dpi=300)
    plt.close(fig)
    print(f"\nSaved: {fname}")


if __name__ == "__main__":
    main()