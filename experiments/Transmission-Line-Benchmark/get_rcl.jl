# =============================================================================
# generate_rcl_matfile.jl
#
# Purpose:
#   Build the RCL ladder transmission line benchmark in port-Hamiltonian
#   DAE form and save it as a .mat file for use in the Python diagnostics.
#
# Circuit topology (ns = number of building blocks):
#
#   u(t) --[node 0]--R1--[node 1]--L1--[node 2]--R2--[node 3]--L2--[node 4]-- ...
#              |                            |                        |
#             R0                           C1                       C2
#              |                            |                        |
#             GND                          GND                      GND
#
#              ... --R_ns--[node 2ns-1]--L_ns--[node 2ns]
#                                                   |
#                                               R_{ns+1}
#                                                   |
#                                                  GND
#
#   Node numbering (2*ns + 1 nodes total):
#   - Node 0          : input node, driven by voltage source u(t)
#                       shunt resistor R0 to ground (sets DAE index to 1)
#   - Node 2k-1       : intermediate node between Rk and Lk (no shunt element)
#   - Node 2k         : output node of building block k
#                       shunt capacitor Ck to ground  (k = 1,...,ns-1)
#   - Node 2*ns       : terminal node, shunt resistor R_{ns+1} to ground (no capacitor)
#
#   Building block k (k = 1,...,ns):
#   - Series resistor Rk  : from node 2k-2 to node 2k-1
#   - Series inductor Lk  : from node 2k-1 to node 2k
#   - Shunt capacitor Ck  : from node 2k to ground  (except last node which has R_{ns+1})
#
# State vector x = [v; i_L; i_v] where:
#   v    : voltages at nodes with capacitors (nodes 2,4,...,2(ns-1))  (differential, size ns-1)
#   i_L  : currents through the ns inductors L1,...,Lns               (differential, size ns)
#   i_v  : current through the voltage source + resistor currents     (algebraic,    size ns+3)
#
# Note: the intermediate nodes 2k-1 (between Rk and Lk) have no capacitor,
# so their voltages are algebraic variables -- they are eliminated via Schur complement.
# The terminal nodes 0 and 2*ns also have no capacitors (only resistors), so
# they too contribute only algebraic variables.
#
# Total state size: n = 3*ns + 2
# Differential variables: r_diff = 2*ns - 1  (ns-1 capacitor voltages + ns inductor currents)
# Algebraic variables:    ns + 3
#
# The system is written in port-Hamiltonian DAE form:
#   E * x'(t) = (J - R) * x(t) + G * u(t)
#   y(t)      = G' * x(t)
#
# where:
#   E  : descriptor matrix (diagonal, 1 for differential vars, 0 for algebraic)
#   J  : skew-symmetric structure matrix (encodes lossless energy exchange)
#   R  : symmetric PSD dissipation matrix (encodes energy loss via resistors)
#   G  : input matrix (port coupling to voltage source)
#
# Reference: Moser et al., PortHamiltonianBenchmarkSystems.jl (2021)
#            Freund, "The SPRIM Algorithm for General RCL Circuits" (2011)
# =============================================================================

using PortHamiltonianBenchmarkSystems
using MAT
using LinearAlgebra

# =============================================================================
# STEP 1: Choose parameters
# =============================================================================

# Number of building blocks (ladder sections)
# Each block adds one series RL branch and one shunt capacitor at the output node
ns = 100

# Resistance value for ALL resistors in the circuit:
#   R0, R1, ..., R_ns, R_{ns+1} -- all set to the same value here
# This is a series resistance (not a shunt), i.e. it sits in the top rail
# between two consecutive nodes.
# The vector has length ns+2 because there are ns+2 resistors total:
#   R0 (shunt at input node) + R1...R_ns (series) + R_{ns+1} (shunt at terminal)
series_resistance = 0.35
r_values = series_resistance * ones(ns + 2)

# Capacitance and inductance values (all set to 1 here, i.e. C=L=1)
# These are passed implicitly as defaults by the benchmark function

# =============================================================================
# STEP 2: Build the benchmark system
# =============================================================================

# This function constructs the pH-DAE matrices (E, J, R, G) from the circuit
# topology using the incidence-matrix formulation described in Freund (2011).
# The 'r' keyword sets the resistance values for all resistors.
E_sparse, J_sparse, R_sparse, G_sparse = setup_DAE1_RCL_LadderNetwork_sparse(
    ns = ns,
    r  = r_values
)

# Convert to dense matrices for manipulation and saving
E = Matrix(E_sparse)
J = Matrix(J_sparse)
R = Matrix(R_sparse)
G = Matrix(G_sparse)

n = size(E, 1)
println("System built successfully.")
println("  Total state dimension n = $n  (should be 3*ns+2 = $(3*ns+2))")

# Diagnostic: check what G actually is
println("  G is identity matrix: $(G ≈ I(n))")
println("  G shape: $(size(G))")

# =============================================================================
# STEP 3: Identify the differential and algebraic variable blocks
# =============================================================================

# The descriptor matrix E is diagonal.
# Entries equal to 1 correspond to differential variables (capacitor voltages
# and inductor currents). Entries equal to 0 are algebraic variables (resistor
# currents, voltage source current -- these have no dynamics of their own).
diag_E = abs.(diag(E))
n_diff = sum(diag_E .> 1e-10)   # number of differential variables
n_alg  = n - n_diff              # number of algebraic variables

println("  Differential variables: $n_diff  (should be 2*ns-1 = $(2*ns-1))")
println("  Algebraic variables:    $n_alg   (should be ns+3   = $(ns+3))")

# =============================================================================
# STEP 4: Find the split between capacitor-voltage and inductor-current variables
#
# Within the differential block (rows 1:n_diff), the variables are ordered as:
#   [capacitor voltages v at nodes 2,4,...,2(ns-1),  inductor currents i_L1,...,i_Lns]
#
# The intermediate nodes 2k-1 (between Rk and Lk) have no capacitor and appear
# only in the algebraic block -- they are not part of the differential state.
#
# We detect where the inductor block starts by looking at the J sparsity pattern.
# In J, the capacitor-voltage rows only connect FORWARD (to higher-index variables).
# The inductor-current rows connect BACKWARD (to lower-index capacitor-voltage vars).
# So the first row with a backward connection marks the start of the inductor block.
# =============================================================================

J_diff = J[1:n_diff, 1:n_diff]   # J restricted to the differential block

# Find the first row index i such that J_diff[i, j] != 0 for some j < i
# (i.e., the first row that "looks left" = first inductor row)
#
# Julia loop scoping note: to avoid the soft-scope ambiguity warning when
# assigning to a variable inside a for-loop at module level, we use findfirst
# instead of a for-loop with break.
backward_rows = [i for i in 2:n_diff if any(abs.(J_diff[i, 1:i-1]) .> 1e-10)]
if isempty(backward_rows)
    error("Could not detect capacitor/inductor boundary in J_diff. Check sparsity pattern.")
end
n_cap_voltages = backward_rows[1] - 1   # rows 1..n_cap_voltages are capacitor voltages
n_ind_currents = n_diff - n_cap_voltages

println("  Capacitor-voltage variables: $n_cap_voltages  (= ns-1 = $(ns-1))")
println("  Inductor-current variables:  $n_ind_currents  (= ns   = $ns)")

# =============================================================================
# STEP 5: Check whether the dissipation matrix R needs regularisation
#
# The dissipation matrix R has the block structure:
#
#   R = [ A_r * diag(1/r) * A_r'    0    0 ]
#       [           0                0    0 ]
#       [           0                0    0 ]
#
# where A_r is the resistor incidence matrix. The (1,1) block lives in the
# capacitor-voltage rows/columns (rows 1:n_cap_voltages).
#
# Problem: A_r * diag(1/r) * A_r' can be SINGULAR (have zero eigenvalues).
# This happens when some interior capacitor nodes have no direct resistive
# path to ground -- their only connection to resistors is through the series
# branches, and the Schur complement S_tilde ends up only positive SEMIdefinite.
#
# Why does this matter? Our DWR method requires S_tilde to be strictly positive
# definite (Assumption 2.1 in the paper). If S_tilde has zero eigenvalues,
# the Block-Jacobi iteration may not converge.
# =============================================================================


# =============================================================================
# STEP 5 & 6: Apply regularisation unconditionally
#
# From the proof (see paper Proposition on regularisation):
# For ANY resistance value, the Schur complement S_tilde has exactly
# n_cap_voltages = ns-1 zero eigenvalues before regularisation.
# This is topological -- it holds regardless of r.
#
# We therefore always add epsilon*I to the capacitor-voltage block.
# Verification of S_tilde > 0 is done in Python after saving, where
# the Schur computation is numerically stable.
# =============================================================================

epsilon = 1.0   # leakage conductance (1/ohm) at each interior capacitor node
R[1:n_cap_voltages, 1:n_cap_voltages] .+= epsilon * I(n_cap_voltages)
println("\nRegularisation applied: added epsilon=$epsilon to cap-voltage block of R")
println("  (rows/cols 1:$n_cap_voltages)")
println("  Verification of S_tilde > 0 will be done in Python.")

# =============================================================================
# STEP 7: Save to .mat file
#
# We save E, J, R (regularised), G as dense matrices.
# The key 'G' stores the Hamiltonian weighting matrix Q, which equals I here.
# Python reads this file via h5py (MATLAB v7.3 HDF5 format).
# =============================================================================

matwrite("rcl_ladder_system.mat", Dict(
    "E" => E,                  # descriptor matrix
    "J" => J,                  # skew-symmetric structure matrix
    "R" => R,                  # dissipation matrix (regularised)
    "G" => Matrix(I, n, n)     # Q = identity (Hamiltonian weight), NOT the benchmark G
))

println("\nSaved: rcl_ladder_system.mat")
println("Run verify_matfile.py to confirm S_tilde > 0.")