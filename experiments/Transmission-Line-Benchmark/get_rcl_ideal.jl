using PortHamiltonianBenchmarkSystems
using LinearAlgebra
using MAT

ns = 100
r_val = 0.35
c_val = 1.0
l_val = 1.0

# r must be a vector of length ns+2 (one resistance per resistor branch)
# ns+2 = ns series resistors + R0 (input shunt) + R_{ns+1} (terminal shunt)
E, J, R, G = setup_DAE1_RCL_LadderNetwork_sparse(ns = ns, r = r_val * ones(ns + 2), c = c_val * ones(ns -1), l = l_val * ones(ns))

Em = Matrix(E)
Jm = Matrix(J)
Rm = Matrix(R)
n  = size(Em, 1)
println("n = $n  (expected 3*ns+2 = $(3*ns+2))")

println("\n--- diag(E) (first 10) ---")
println(round.(diag(Em)[1:10], digits=4))

println("\n--- R diagonal (first 10) ---")
println(round.(diag(Rm)[1:10], digits=4))

# Save to mat file
# G from the benchmark is the port matrix (not Q).
# We save Q = I explicitly since that is what Python expects under key 'G'.
matwrite("rcl_ladder_system.mat", Dict(
    "E" => Em,
    "J" => Jm,
    "R" => Rm,
    "G" => Matrix(I, n, n)   # Q = identity (Hamiltonian weight)
))
println("\nSaved: rcl_ladder_system.mat")