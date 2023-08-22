
using DRSOM


using LinearAlgebra
using SparseArrays
using Random

Random.seed!(1)


H = Symmetric(sprand(20, 20, 0.75), :U)
g = rand(20)

Δ = 5.0

x, l, k = DRSOM.LanczosTrustRegionBisect(H, -g, Δ)
lc = DRSOM.Lanczos(20, 30, g)
x1, l1, info = DRSOM.InexactLanczosTrustRegionBisect(H, -g, Δ, lc; k=21)