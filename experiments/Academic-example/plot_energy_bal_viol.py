"""
Energy balance violation vs DOFs: Uniform dG(0) vs DWR adaptive dG(0).
=======================================================================
Self-contained inline style (no external dependency).
Palette: Wong (2011) colorblind-safe.  Font: Latin Modern Roman.
"""
import os
import numpy as np
import scipy.linalg as la
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
          "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[v] = "1"

# =============================================================================
# Style
# =============================================================================
plt.rcParams.update({
    'font.family':        'serif',
    'font.serif':         ['Latin Modern Roman', 'DejaVu Serif', 'serif'],
    'mathtext.fontset':   'cm',
    'font.size':          10,
    'axes.linewidth':     0.6,
    'axes.labelsize':     11,
    'axes.spines.top':    False,
    'axes.spines.right':  False,
    'xtick.labelsize':    15,
    'ytick.labelsize':    15,
    'xtick.direction':    'in',
    'ytick.direction':    'in',
    'xtick.major.width':  0.5,
    'ytick.major.width':  0.5,
    'xtick.major.size':   3.5,
    'ytick.major.size':   3.5,
    'xtick.minor.size':   2,
    'ytick.minor.size':   2,
    'xtick.major.pad':    4,
    'ytick.major.pad':    4,
    'axes.grid':          False,
    'legend.fontsize':    9.5,
    'legend.frameon':     True,
    'legend.framealpha':  0.92,
    'legend.edgecolor':   '#cccccc',
    'legend.fancybox':    False,
    'legend.borderpad':   0.5,
    'legend.handlelength': 2.2,
    'savefig.dpi':        300,
    'savefig.bbox':       'tight',
    'savefig.pad_inches': 0.06,
})

_W   = {'blue': '#0072B2', 'pink': '#CC79A7', 'black': '#000000'}
_mew = 0.6
MSTYLE = {
    'Uniform': dict(color=_W['blue'], marker='o', ls='--', lw=1.4, ms=5.5,
                    mec='white', mew=_mew, zorder=2, label='Uniform dG(0)'),
    'DWR':     dict(color=_W['pink'], marker='^', ls='-',  lw=2.2, ms=6.0,
                    mec='white', mew=0.7,  zorder=5, label='DWR adaptive dG(0)'),
}

def _grid(ax):
    ax.grid(True, which='major', color='#cccccc', lw=0.3, alpha=0.5)
    ax.grid(True, which='minor', color='#e0e0e0', lw=0.2, alpha=0.3)

# =============================================================================
# System
# =============================================================================
class Index1_PH_DAE:
    def __init__(self, E, J, R, Q, B):
        self.E_full, self.J_full = E, J
        self.R_full, self.Q_full, self.B_full = R, Q, B
        self.r   = np.linalg.matrix_rank(E)
        self.E11 = E[:self.r, :self.r]
        self.Q11 = Q[:self.r, :self.r]
        A        = (J - R) @ Q
        A11, A12 = A[:self.r, :self.r], A[:self.r, self.r:]
        A21, A22 = A[self.r:, :self.r], A[self.r:, self.r:]
        B1, B2   = B[:self.r], B[self.r:]
        A22i     = la.inv(A22)
        self.S       = -(A11 - A12 @ A22i @ A21)
        self.S_tilde =  0.5*(self.S + self.S.T)
        self.F       =  B1 - A12 @ A22i @ B2
        self.Kx      = -A22i @ A21
        self.Ku      = -A22i @ B2

    def reconstruct(self, x1, u):
        return np.concatenate([x1, self.Kx @ x1 + self.Ku @ u])

    def hamiltonian(self, z):
        return 0.5 * z @ (self.E_full.T @ self.Q_full) @ z

    def grad_hamiltonian(self, z):
        return (self.E_full.T @ self.Q_full) @ z

# =============================================================================
# Solvers
# =============================================================================
def solve_primal(sys, grid, x0, u):
    x = np.zeros((len(grid), sys.r)); x[0] = x0
    for i in range(len(grid) - 1):
        k      = grid[i+1] - grid[i]
        x[i+1] = la.solve(sys.E11 + k*sys.S,
                           sys.E11 @ x[i] + k*sys.F @ u(grid[i] + 0.5*k))
    return x

def compute_residuals(sys, grid, x, u):
    M = len(grid) - 1; G = np.zeros(M)
    for i in range(M):
        k  = grid[i+1] - grid[i]; ui = u(grid[i] + 0.5*k)
        zl = sys.reconstruct(x[i],   ui)
        zr = sys.reconstruct(x[i+1], ui)
        e  = sys.Q_full @ zr
        G[i] = ((sys.hamiltonian(zr) - sys.hamiltonian(zl))
                - k * ((sys.B_full.T @ e) @ ui - e @ sys.R_full @ e))
    return G

def solve_adjoint(sys, grid, x, u):
    M   = len(grid) - 1
    z   = np.zeros((M + 1, sys.r))
    G   = compute_residuals(sys, grid, x, u)
    for i in reversed(range(M)):
        k     = grid[i+1] - grid[i]; ui = u(grid[i] + 0.5*k)
        rhs   = sys.E11.T @ z[i+1]
        gH    = sys.grad_hamiltonian(sys.reconstruct(x[i+1], ui))[:sys.r]
        rhs  += 2*G[i] * (gH + 2*k*sys.S_tilde @ x[i+1] - k*(sys.F @ ui))
        if i < M - 1:
            rhs -= 2*G[i+1] * gH
        z[i] = la.solve(sys.E11.T + k*sys.S.T, rhs)
    return z

def compute_dwr_indicators(sys, grid, x, z, u):
    M   = len(grid) - 1; eta = np.zeros(M)
    for i in range(M):
        k     = grid[i+1] - grid[i]
        dz    = z[i] if i == 0 else z[i] - z[i-1]
        R_int  = sys.F @ u(grid[i] + 0.5*k) - sys.S @ x[i+1]
        R_jump = sys.E11 @ x[i] if i == 0 else sys.E11 @ (x[i] - x[i-1])
        eta[i] = 0.5*k*(R_int @ dz) + R_jump @ dz
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
    u       = lambda t: (np.array([np.sin(2*np.pi*t)]) if t < 0.5
                         else np.array([0.0]))

    # 1. Uniform sweep
    n_uni    = np.array([10, 20, 40, 80, 160, 320, 640])
    viol_uni = []
    for n in n_uni:
        g  = np.linspace(0, 1, n + 1)
        xg = solve_primal(sys_obj, g, x0, u)
        viol_uni.append(float(np.sum(compute_residuals(sys_obj, g, xg, u)**2)))

    # 2. DWR adaptive loop
    grid     = np.linspace(0, 1, 11)
    dwr_ns, dwr_viol = [], []
    for _ in range(30):
        x   = solve_primal(sys_obj, grid, x0, u)
        G   = compute_residuals(sys_obj, grid, x, u)
        dwr_ns.append(len(grid) - 1)
        dwr_viol.append(float(np.sum(G**2)))
        if len(grid) - 1 >= 640:
            break
        z   = solve_adjoint(sys_obj, grid, x, u)
        eta = compute_dwr_indicators(sys_obj, grid, x, z, u)
        grid = dorfler_refine(grid, eta, theta=0.5)

    # 3. Fit asymptotic slope on uniform data
    fit_ns = np.log(n_uni[-5:].astype(float))
    fit_qs = np.log(np.array(viol_uni[-5:]))
    slope, intercept = np.polyfit(fit_ns, fit_qs, 1)
    ref_slope = round(slope * 2) / 2.0
    n_ref = np.array([n_uni[0], n_uni[-1]], dtype=float)
    q_ref = np.exp(intercept) * n_ref**slope * 1.5
    print(f"Uniform fitted slope: O(N^{slope:.2f})")

    # 4. Plot
    fig, ax = plt.subplots(figsize=(5.0, 3.5))
    ax.loglog(n_uni,              viol_uni,  **MSTYLE['Uniform'])
    ax.loglog(np.array(dwr_ns),   dwr_viol,  **MSTYLE['DWR'])
    ax.loglog(n_ref, q_ref, color=_W['black'], ls='--', lw=0.9,
              label=fr"$O(N^{{{ref_slope:.1f}}})$")

    ax.set_xlabel(r'Number of time steps $N$', fontsize = 14)
    ax.set_ylabel(r'Energy bal. viol.', fontsize = 14)
    
    # Synchronize y-axis limits
    ax.set_ylim([1e-7, 2e-1])
    _grid(ax)
    
    ax.legend(loc='upper right', fontsize=10.0,
              handlelength=1.5, handletextpad=0.5, borderpad=0.4)
              
    ax.tick_params(which = 'both', labelsize=14)
    # Enforce identical static margins
    fig.subplots_adjust(left=0.18, right=0.95, bottom=0.18, top=0.92)
    fig.savefig("energy_balance_violation.png", dpi=600, bbox_inches=None)
    plt.close(fig)
    print("Saved: energy_balance_violation.png")

if __name__ == "__main__":
    main()