import os
import numpy as np
import scipy.linalg as la
import matplotlib
matplotlib.use('Agg')
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
    'figure.dpi'         : 150,
    'savefig.dpi'        : 600,
    'savefig.bbox'       : 'tight',
    'savefig.pad_inches' : 0.12,
})

_W = {
    'black'  : '#000000',
}

def _add_grid(ax, which='major'):
    ax.grid(True, which='major', color='#cccccc', lw=0.35, alpha=0.55)
    if which == 'both':
        ax.grid(True, which='minor', color='#e0e0e0', lw=0.2, alpha=0.35)

# ==============================================================================
# System Definition 
# ==============================================================================
class Index1_PH_DAE:
    def __init__(self, E, J, R, Q, B):
        self.n = len(E)
        self.E = E
        self.Q = Q
        self.R = R
        self.J = J
        self.B = B
        self.A_sys = (J - R) @ Q
        self.r = np.linalg.matrix_rank(E)
        self.E11 = E[:self.r, :self.r]
        self.A11 = self.A_sys[:self.r, :self.r]
        self.A12 = self.A_sys[:self.r, self.r:]
        self.A21 = self.A_sys[self.r:, :self.r]
        self.A22 = self.A_sys[self.r:, self.r:]
        self.B1 = B[:self.r]
        self.B2 = B[self.r:]
        try:
            self.A22_inv = la.inv(self.A22)
        except la.LinAlgError:
            raise ValueError("System is not Index-1 (A22 is singular).")
        self.S_mat = -(self.A11 - self.A12 @ self.A22_inv @ self.A21)
        self.S_tilde = 0.5 * (self.S_mat + self.S_mat.T)
        self.F_mat = (self.B1 - self.A12 @ self.A22_inv @ self.B2)
        self.M_x1 = -self.A22_inv @ self.A21
        self.M_u = -self.A22_inv @ self.B2
        self.E11_T = self.E11.T
        self.S_tilde_T = self.S_tilde.T
        self.eigenvalues = la.eigvals(self.S_mat)

    def get_F(self, u_val):
        return self.F_mat @ u_val

    def reconstruct_z(self, x1, u_val):
        x2 = self.M_x1 @ x1 + self.M_u @ u_val
        return np.concatenate([x1, x2])

    def get_hamiltonian(self, x1):
        return 0.5 * np.dot(x1, self.E11 @ x1)

    def get_grad_hamiltonian(self, x1):
        return self.E11 @ x1

    def get_power_imbalance(self, x1, u_val):
        z = self.reconstruct_z(x1, u_val)
        Qz = self.Q @ z
        y = self.B.T @ Qz
        P_in = np.dot(y, u_val)
        P_diss = np.dot(Qz, self.R @ Qz)
        return P_in - P_diss

    def get_grad_power_imbalance(self, x1, u_val):
        dz_dx1 = np.vstack([np.eye(self.r), self.M_x1])
        z = self.reconstruct_z(x1, u_val)
        Qz = self.Q @ z
        term1 = (self.Q @ self.B @ u_val).T @ dz_dx1
        term2 = 2 * (Qz.T @ self.R @ self.Q) @ dz_dx1
        return term1 - term2


class EnergyBalanceGoal:
    def __init__(self, sys):
        self.sys = sys
        self.residuals = []

    def compute_residuals(self, grid, x_hist, u_func):
        self.residuals = []
        M = len(grid) - 1
        for i in range(M):
            dt = grid[i+1] - grid[i]
            t_mid = grid[i] + dt/2
            u_mid = u_func(t_mid)
            x_curr = x_hist[i+1]
            x_prev = x_hist[i]
            delta_H = self.sys.get_hamiltonian(x_curr) - self.sys.get_hamiltonian(x_prev)
            g_val = self.sys.get_power_imbalance(x_curr, u_mid)
            integral_g = g_val * dt
            G_i = -integral_g + delta_H
            self.residuals.append(G_i)
        return np.array(self.residuals)

    def get_adjoint_rhs(self, i, grid, x_hist, u_func):
        M = len(grid) - 1
        dt = grid[i+1] - grid[i]
        t_mid = grid[i] + dt/2
        u_mid = u_func(t_mid)
        x_curr = x_hist[i+1]
        G_i = self.residuals[i]
        grad_g = self.sys.get_grad_power_imbalance(x_curr, u_mid)
        grad_H = self.sys.get_grad_hamiltonian(x_curr)
        deriv_1 = 2 * G_i * (-dt * grad_g + grad_H)
        deriv_2 = np.zeros_like(deriv_1)
        if i < M - 1:
            G_next = self.residuals[i+1]
            deriv_2 = 2 * G_next * (-grad_H)
        return deriv_1 + deriv_2


def solve_primal_dg0(sys, grid, x1_0, u_func):
    M = len(grid) - 1
    x_hist = np.zeros((M + 1, sys.r))
    x_hist[0] = x1_0
    for m in range(M):
        dt = grid[m+1] - grid[m]
        t_mid = grid[m] + dt/2
        Mat = sys.E11 + dt * sys.S_tilde
        F_val = sys.get_F(u_func(t_mid))
        RHS = sys.E11 @ x_hist[m] + dt * F_val
        try:
            x_hist[m+1] = la.solve(Mat, RHS)
        except la.LinAlgError:
            x_hist[m+1] = la.lstsq(Mat, RHS)[0]
    return x_hist


def solve_adjoint_exact(sys, grid, x_hist, u_func, goal_obj):
    M = len(grid) - 1
    z_hist = np.zeros((M, sys.r))
    for i in range(M - 1, -1, -1):
        dt = grid[i+1] - grid[i]
        Mat_adj = sys.E11_T + dt * sys.S_tilde_T
        R_i = goal_obj.get_adjoint_rhs(i, grid, x_hist, u_func)
        rhs = R_i.copy()
        if i < M - 1:
            rhs += sys.E11_T @ z_hist[i+1]
        try:
            z_hist[i] = la.solve(Mat_adj, rhs)
        except la.LinAlgError:
            z_hist[i] = la.lstsq(Mat_adj, rhs)[0]
    return z_hist


def compute_error_indicators(sys, grid, x_hist, z_hist, u_func):
    M = len(grid) - 1
    eta = np.zeros(M)
    for i in range(M):
        dt = grid[i+1] - grid[i]
        t_mid = grid[i] + dt/2
        x_curr = x_hist[i+1]
        x_jump = x_hist[i+1] - x_hist[i]
        if i == 0:
            dz = z_hist[i]
        else:
            dz = z_hist[i] - z_hist[i-1]
        F_val = sys.get_F(u_func(t_mid))
        Res_int = F_val - sys.S_tilde @ x_curr
        term_int = (dt / 2.0) * np.dot(Res_int, dz)
        term_jump = np.dot(sys.E11 @ x_jump, dz)
        eta[i] = 0.5 * abs(term_int + term_jump)
    return eta


def get_system_config(config_name):
    if config_name == "original":
        E = np.diag([1.0, 1.0, 0.0])
        J = np.array([[0, 1, -1], [-1, 0, 0], [1, 0, 0]])
        R = np.diag([2.0, 2.0, 1.0])
        Q = np.eye(3)
        B = np.array([[1.0], [0.0], [0.0]])
        x1_0 = np.array([1.0, 0.0])
        desc = "Original: Stable with coupling, negative eigenvalues"
    elif config_name == "positive_eigs":
        E = np.diag([1.0, 1.0, 0.0])
        J = np.array([[0, 2, -1], [-2, 0, -1], [1, 1, 0]])
        R = np.diag([-1.5, -1.5, 0.5])
        Q = np.eye(3)
        B = np.array([[0.5], [0.5], [0.0]])
        x1_0 = np.array([0.5, 0.5])
        desc = "Positive Eigenvalues: Anti-dissipative with strong coupling"
    elif config_name == "negative_eigs":
        E = np.diag([1.0, 1.0, 0.0])
        J = np.array([[0, 0.5, -1], [-0.5, 0, -0.5], [1, 0.5, 0]])
        R = np.diag([0.5, 0.5, 0.1])
        Q = np.eye(3)
        B = np.array([[1.0], [0.5], [0.0]])
        x1_0 = np.array([1.0, 0.5])
        desc = "Negative Eigenvalues: Strongly dissipative with coupling"
    elif config_name == "mixed_eigs":
        E = np.diag([1.0, 1.0, 0.0])
        J = np.array([[0, 3, -2], [-3, 0, -1], [2, 1, 0]])
        R = np.diag([1.0, -0.5, 1.0])
        Q = np.eye(3)
        B = np.array([[0.8], [0.6], [0.0]])
        x1_0 = np.array([1.0, -0.5])
        desc = "Mixed Eigenvalues: Oscillatory with partial damping"
    elif config_name == "unstable":
        E = np.diag([1.0, 1.0, 0.0])
        J = np.array([[0, 1.5, -1], [-1.5, 0, -1.5], [1, 1.5, 0]])
        R = np.diag([-2.0, -2.0, 0.5])
        Q = np.eye(3)
        B = np.array([[0.5], [0.5], [0.0]])
        x1_0 = np.array([0.3, 0.3])
        desc = "Unstable: Strong anti-dissipation with coupling"
    else:
        raise ValueError(f"Unknown configuration: {config_name}")
    return E, J, R, Q, B, x1_0, desc


def run_adaptive_simulation_with_history(config_name="original", max_iter=15, tol=1e-12, T_final=1.0):
    """Run adaptive simulation and return mesh history"""
    E, J, R, Q, B, x1_0, description = get_system_config(config_name)
    
    try:
        sys = Index1_PH_DAE(E, J, R, Q, B)
    except ValueError as e:
        print(f"ERROR: {e}")
        return None, None
    
    goal = EnergyBalanceGoal(sys)
    
    def u_func(t):
        return np.array([np.sin(2 * np.pi * t)]) if t < 0.5 else np.array([0.0])
    
    fine_grid = np.linspace(0, T_final, 20001)
    x_ref_hist = solve_primal_dg0(sys, fine_grid, x1_0, u_func)
    
    if np.any(np.isnan(x_ref_hist)) or np.any(np.abs(x_ref_hist) > 1e10):
        T_final = 0.5
        x1_0 = x1_0 * 0.5
        fine_grid = np.linspace(0, T_final, 20001)
        x_ref_hist = solve_primal_dg0(sys, fine_grid, x1_0, u_func)
    
    ref_residuals = goal.compute_residuals(fine_grid, x_ref_hist, u_func)
    
    theta = 0.5
    grid = np.linspace(0, T_final, 11)
    mesh_history = []
    
    for iteration in range(max_iter):
        mesh_history.append(grid.copy())
        
        x_hist = solve_primal_dg0(sys, grid, x1_0, u_func)
        if np.any(np.isnan(x_hist)) or np.any(np.abs(x_hist) > 1e10):
            break
        
        current_residuals = goal.compute_residuals(grid, x_hist, u_func)
        z_hist = solve_adjoint_exact(sys, grid, x_hist, u_func, goal)
        eta = compute_error_indicators(sys, grid, x_hist, z_hist, u_func)
        total_est_error = np.sum(eta)
        
        if total_est_error < tol:
            break
        
        indices = np.argsort(eta)[::-1]
        cumulative_err = 0.0
        marked_indices = []
        for idx in indices:
            cumulative_err += eta[idx]
            marked_indices.append(idx)
            if cumulative_err >= theta * total_est_error:
                break
        
        marked_indices.sort()
        new_grid = [grid[0]]
        for i in range(len(grid)-1):
            if i in marked_indices:
                new_grid.append((grid[i] + grid[i+1]) / 2.0)
            new_grid.append(grid[i+1])
        grid = np.array(new_grid)
    
    return mesh_history, description


def plot_mesh_evolution(config_name="original"):
    """Plot 2: Mesh Evolution"""
    mesh_history, description = run_adaptive_simulation_with_history(config_name)
    
    if mesh_history is None:
        print("Simulation failed")
        return
    
    fig, ax = plt.subplots(figsize=(3.0, 3.0)) 
    
    for iter_num, mesh in enumerate(mesh_history):
        y_pos = np.ones(len(mesh)) * iter_num
        
        # Draw a faint horizontal line for the iteration level
        ax.plot([mesh[0], mesh[-1]], [iter_num, iter_num], 
                color='#e0e0e0', lw=0.8, zorder=1)
        
        # Use small dots instead of vertical bars
        ax.plot(mesh, y_pos, marker='.', linestyle='None', 
                color=_W['black'], markersize=4, zorder=2)
        
    ax.set_xlabel(r'Time $t$', fontsize=11)
    ax.set_ylabel('Iteration', fontsize=11)
    
    
    
    ax.set_yticks(range(0, len(mesh_history), 2)) # Show every other tick to avoid crowding
    ax.set_ylim(-0.5, len(mesh_history) - 0.5)
    ax.tick_params(labelsize=9) 
    
    # Only use horizontal grid lines to help track iterations
    ax.grid(True, axis='y', color='#f0f0f0', lw=0.5)
    
    fig.tight_layout()
    filename = f'plot2_mesh_evolution_{config_name}.png'
    fig.savefig(filename)
    plt.close(fig)
    print(f"Saved: {filename}")


if __name__ == "__main__":
    configs = ["original", "positive_eigs", "negative_eigs", "mixed_eigs", "unstable"]
    
    for config in configs:
        print(f"\nGenerating Mesh Evolution plot for: {config}")
        plot_mesh_evolution(config)