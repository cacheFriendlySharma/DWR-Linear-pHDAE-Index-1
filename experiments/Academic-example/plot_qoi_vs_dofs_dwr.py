"""
QoI error vs DOFs: Uniform dG(0) vs DWR adaptive dG(0).
=========================================================
Self-contained inline style.
Palette: Wong (2011) colorblind-safe.  Font: Latin Modern Roman.
"""
import os
for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
          "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(v, "1")

import numpy as np
import scipy.linalg as la
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# =============================================================================
# Style
# =============================================================================
plt.rcParams.update({
    'font.family':        'serif',
    'font.serif':         ['Latin Modern Roman', 'DejaVu Serif', 'serif'],
    'mathtext.fontset':   'cm',
    'font.size':          12,
    'axes.linewidth':     0.8,
    'axes.labelsize':     15,
    'axes.spines.top':    False,
    'axes.spines.right':  False,
    'xtick.labelsize':    13,
    'ytick.labelsize':    13,
    'xtick.direction':    'in',
    'ytick.direction':    'in',
    'xtick.major.width':  0.6,
    'ytick.major.width':  0.6,
    'xtick.major.size':   5,
    'ytick.major.size':   5,
    'xtick.minor.size':   3,
    'ytick.minor.size':   3,
    'xtick.major.pad':    5,
    'ytick.major.pad':    5,
    'axes.grid':          False,
    'legend.fontsize':    9.0,
    'legend.frameon':     True,
    'legend.framealpha':  0.92,
    'legend.edgecolor':   '#cccccc',
    'legend.fancybox':    False,
    'legend.borderpad':   0.5,
    'legend.handlelength': 2.2,
    'savefig.dpi':        300,
    'savefig.bbox':       'tight',
    'savefig.pad_inches': 0.08,
})

_W   = {'blue': '#0072B2', 'pink': '#CC79A7', 'black': '#000000'}
_mew = 0.6
MSTYLE = {
    'Uniform': dict(color=_W['blue'], marker='o', ls='--', lw=1.6, ms=7,
                    mec='white', mew=_mew, zorder=2, label='Uniform dG(0)'),
    'DWR':     dict(color=_W['pink'], marker='^', ls='-',  lw=2.5, ms=8,
                    mec='white', mew=0.8,  zorder=5, label='DWR adaptive dG(0)'),
}

def _grid(ax):
    ax.grid(True, which='major', color='#cccccc', lw=0.3, alpha=0.5)
    ax.grid(True, which='minor', color='#e0e0e0', lw=0.2, alpha=0.3)

# =============================================================================
# System
# =============================================================================
class Index1_PH_DAE:
    def __init__(self, E, J, R, Q, B):
        self.r   = np.linalg.matrix_rank(E)
        self.E11 = E[:self.r, :self.r]
        A        = (J - R) @ Q
        A11, A12 = A[:self.r, :self.r], A[:self.r, self.r:]
        A21, A22 = A[self.r:, :self.r], A[self.r:, self.r:]
        B1, B2   = B[:self.r], B[self.r:]
        A22i     = la.inv(A22)
        self.S       = -(A11 - A12 @ A22i @ A21)
        self.S_tilde =  0.5*(self.S + self.S.T)
        self.F       =  B1 - A12 @ A22i @ B2

    def hamiltonian(self, x): return 0.5 * x @ self.E11 @ x
    def grad_hamiltonian(self, x): return self.E11 @ x

# =============================================================================
# Solvers
# =============================================================================
def solve_primal(sys, grid, x0, u):
    x = np.zeros((len(grid), sys.r)); x[0] = x0
    for i in range(len(grid) - 1):
        dt     = grid[i+1] - grid[i]
        x[i+1] = la.solve(sys.E11 + dt*sys.S,
                           sys.E11 @ x[i] + dt*sys.F @ u(grid[i] + 0.5*dt))
    return x

def compute_G(sys, grid, x, u):
    M = len(grid) - 1; G = np.zeros(M)
    for i in range(M):
        dt = grid[i+1] - grid[i]; x1 = x[i+1]
        G[i] = (sys.hamiltonian(x[i+1]) - sys.hamiltonian(x[i])
                + dt*(x1 @ sys.S_tilde @ x1
                      - u(grid[i] + 0.5*dt) @ (sys.F.T @ x1)))
    return G

def solve_adjoint(sys, grid, x, u):
    M = len(grid) - 1; z = np.zeros((M, sys.r))
    G = compute_G(sys, grid, x, u)
    for i in reversed(range(M)):
        dt  = grid[i+1] - grid[i]
        rhs = 2*G[i]*(sys.grad_hamiltonian(x[i+1])
                       + 2*dt*sys.S_tilde @ x[i+1]
                       - dt*(sys.F @ u(grid[i] + 0.5*dt)))
        if i < M - 1:
            rhs -= 2*G[i+1]*sys.grad_hamiltonian(x[i+1])
            rhs += sys.E11 @ z[i+1]
        z[i] = la.solve(sys.E11.T + dt*sys.S.T, rhs)
    return z

def compute_eta(sys, grid, x, z, u):
    M = len(grid) - 1; eta = np.zeros(M)
    for i in range(M):
        dt  = grid[i+1] - grid[i]
        res = sys.F @ u(grid[i] + 0.5*dt) - sys.S @ x[i+1]
        dz  = z[i] if i == 0 else z[i] - z[i-1]
        eta[i] = (dt/2)*(res @ dz) + (sys.E11 @ (x[i+1] - x[i])) @ dz
    return eta

def dorfler_refine(grid, indicators, theta=0.5):
    em  = np.abs(indicators)
    idx = np.argsort(em)[::-1]
    cs  = np.cumsum(em[idx])
    marked = set(idx[:np.searchsorted(cs, theta*cs[-1]) + 1])
    new = [grid[0]]
    for i in range(len(grid) - 1):
        if i in marked:
            new.append(0.5*(grid[i] + grid[i+1]))
        new.append(grid[i+1])
    return np.array(new)

# =============================================================================
# Main
# =============================================================================
def main():
    E  = np.diag([1.0, 1.0, 0.0])
    J  = np.array([[0, 1, -1], [-1, 0, 0], [1, 0, 0]], dtype=float)
    R  = np.diag([0.5, 0.5, 0.1])
    Q  = np.eye(3)
    B  = np.array([[1.0], [0.0], [0.0]])
    x0 = np.array([1.0, 0.0])
    sys_obj = Index1_PH_DAE(E, J, R, Q, B)
    T       = 1.0
    u       = lambda t: (np.array([np.sin(2*np.pi*t)]) if t < 0.5
                         else np.array([0.0]))

    # Reference QoI on fine uniform grid
    q_ref, _, _ = (lambda g: (float(np.sum(compute_G(sys_obj, g,
                               solve_primal(sys_obj, g, x0, u), u)**2)),
                               None, None))(np.linspace(0, T, 20001))
    # recompute properly
    g_ref  = np.linspace(0, T, 20001)
    x_ref  = solve_primal(sys_obj, g_ref, x0, u)
    G_ref  = compute_G(sys_obj, g_ref, x_ref, u)
    q_ref  = float(np.sum(G_ref**2))
    print(f"q_ref = {q_ref:.6e}")

    # 1. Uniform
    n_uni   = [10, 20, 40, 80, 160, 320, 640, 1280]
    err_uni = []
    for n in n_uni:
        g  = np.linspace(0, T, n + 1)
        xg = solve_primal(sys_obj, g, x0, u)
        q  = float(np.sum(compute_G(sys_obj, g, xg, u)**2))
        err_uni.append(abs(q - q_ref))

    # 2. DWR adaptive loop
    grid     = np.linspace(0, T, 5)
    dwr_ns, dwr_errs = [], []
    for _ in range(30):
        x   = solve_primal(sys_obj, grid, x0, u)
        G   = compute_G(sys_obj, grid, x, u)
        q   = float(np.sum(G**2))
        dwr_ns.append(len(grid) - 1)
        dwr_errs.append(abs(q - q_ref))
        if len(grid) - 1 >= 1280 or abs(q - q_ref) < 1e-12:
            break
        z   = solve_adjoint(sys_obj, grid, x, u)
        eta = compute_eta(sys_obj, grid, x, z, u)
        grid = dorfler_refine(grid, eta, theta=0.5)

    # 3. Fit asymptotic slope on uniform data
    fit_ns = np.log(np.array(n_uni[-5:], float))
    fit_qs = np.log(np.array(err_uni[-5:]))
    slope, intercept = np.polyfit(fit_ns, fit_qs, 1)
    ref_slope = round(slope * 2) / 2.0
    n_ref_line = np.array([n_uni[0], n_uni[-1]], dtype=float)
    q_ref_line = np.exp(intercept) * n_ref_line**slope * 1.5
    print(f"Uniform fitted slope: O(N^{slope:.2f})")

    # 4. Plot
    fig, ax = plt.subplots(figsize=(5.0, 3.5))
    ax.loglog(np.array(n_uni),   err_uni,   **MSTYLE['Uniform'])
    ax.loglog(np.array(dwr_ns),  dwr_errs,  **MSTYLE['DWR'])
    ax.loglog(n_ref_line, q_ref_line, color=_W['black'], ls='--', lw=0.9,
              label=fr"$O(N^{{{ref_slope:.1f}}})$")

    ax.set_xlabel(r'Number of time steps $N$', fontsize = 14)
    ax.set_ylabel(r'QoI error $|\mathcal{J}_{\mathrm{ref}} - \mathcal{J}_k|$', fontsize = 14)
    ax.tick_params(which = 'both', labelsize=14)
    
    # Synchronize y-axis limits
    ax.set_ylim([1e-7, 2e-1])
    _grid(ax)
    
    ax.legend(loc='upper right', fontsize=10.0,
              handlelength=1.5, handletextpad=0.5, borderpad=0.4)
              
    # Enforce identical static margins
    fig.subplots_adjust(left=0.18, right=0.95, bottom=0.18, top=0.92)
    fig.savefig("plot_qoi_vs_dofs_dwr_original.png", dpi=600, bbox_inches=None)
    plt.close(fig)
    print("Saved: plot_qoi_vs_dofs_dwr_original.png")

if __name__ == "__main__":
    main()
