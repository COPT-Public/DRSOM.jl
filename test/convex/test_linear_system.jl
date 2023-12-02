@doc raw"""
    We test the eigensolve is stable at an ill-conditioned matrix from LIBSVM
"""



include("../lp.jl")
include("../tools.jl")

using ArgParse
using DRSOM
using Distributions
using LineSearches
using Optim
using ProximalOperators
using ProximalAlgorithms
using Random
using Plots
using Printf
using LazyStack
using KrylovKit
using HTTP
using LaTeXStrings
using LinearAlgebra
using Statistics
using LinearOperators
using Optim
using SparseArrays
using DRSOM
using .LP
using LoopVectorization
using LIBSVMFileIO
using DataFrames
using CSV
Base.@kwdef mutable struct KrylovInfo
    normres::Float64
    numops::Float64
end
table = []
name = "a4a"
# name = "a9a"
# name = "w4a"
# name = "covtype"
# name = "news20"
# name = "rcv1"
# names = ["a4a"] #, "a9a", "w4a", "covtype", "rcv1", "news20"]
names = ["covtype"] #, "a9a", "w4a", "covtype", "rcv1", "news20"]
# names = ["a4a", "a9a", "w4a", "covtype", "rcv1"]
@warn "news20 is very big..., consider run on a server"
# use the data matrix of libsvm
f1(A, d=2) = sqrt.(sum(abs2.(A), dims=d))
Random.seed!(1)

# for name in names
@info "run $name"
X, y = libsvmread("test/instances/libsvm/$name.libsvm"; dense=false)
Xv = hcat(X...)'
Rc = 1 ./ f1(Xv)[:]
Xv = (Rc |> Diagonal) * Xv
if name in ["covtype"]
    y = convert(Vector{Float64}, (y .- 1.5) * 2)
else
end

γ = 1e-10
n = Xv[1, :] |> length
Random.seed!(1)
N = y |> length

Q = Xv' * Xv
function gfs(w)
    return (Q * w - Xv' * y) / N + γ * w
end
r = Dict()
# 
r["GHM-Lanczos"] = KrylovInfo(normres=0.0, numops=0)
r["Newton-CG"] = KrylovInfo(normres=0.0, numops=0)
r["Newton-GMRES"] = KrylovInfo(normres=0.0, numops=0)
r["Newton-rGMRES"] = KrylovInfo(normres=0.0, numops=0)

samples = 5
# for idx in 1:samples
w₀ = rand(Float64, n)
g = gfs(w₀)
δ = -0.1
ϵᵧ = 1e-5
function hvp(w)
    gn = gfs(w₀ + ϵᵧ .* w)
    gf = g
    return (gn - gf) / ϵᵧ
end
function hvp!(y, w)
    gn = gfs(w₀ + ϵᵧ .* w)
    gf = g
    return copyto!(y, (gn - gf) / ϵᵧ)
end
Fw(w) = [hvp(w[1:end-1]) + g .* w[end]; w[1:end-1]' * g + δ * w[end]]
Fc = DRSOM.Counting(Fw)
Hc = DRSOM.Counting(hvp)
@info "data reading finished"

max_iteration = 200
ε = 1e-6

rl = KrylovKit.eigsolve(
    Fc, [w₀; 1], 1, :SR, Lanczos(tol=ε / 10, maxiter=max_iteration, verbosity=3, eager=true);
)
λ₁ = rl[1]
ξ₁ = rl[2][1]

r["GHM-Lanczos"].normres += (Fc(ξ₁) - λ₁ .* ξ₁) |> norm
r["GHM-Lanczos"].numops += rl[end].numops

if false:
    rl = KrylovKit.linsolve(
        Hc, -g, w₀, CG(; tol=ε, maxiter=max_iteration, verbosity=3);
    )
    r["Newton-CG"].normres += ((hvp(rl[1]) + g) |> norm)
    r["Newton-CG"].numops += rl[end].numops
    rl = KrylovKit.linsolve(
        Hc, -g, w₀, GMRES(; tol=ε, maxiter=1, krylovdim=max_iteration, verbosity=3);
    )
    r["Newton-GMRES"].normres += ((hvp(rl[1]) + g) |> norm)
    r["Newton-GMRES"].numops += rl[end].numops
    rl = KrylovKit.linsolve(
        Hc, -g, w₀, GMRES(; tol=ε, maxiter=4, krylovdim=div(max_iteration, 4), verbosity=3);
    )
    r["Newton-rGMRES"].normres += ((hvp(rl[1]) + g) |> norm)
    r["Newton-rGMRES"].numops += rl[end].numops

################################
# ALTERNATIVELY, USE Krylov.jl
################################
# opH = LinearOperator(
#     Float64, n, n, true, true,
#     (y, v) -> hvp!(y, v)
# )
# d, __unused_info = cg(
#     opH, -g; verbose=1, itmax=200
# )

# end
for (k, v) in r
    push!(table, [name, n, k, v.numops / samples, v.normres / samples])
end
# end

tmat = hcat(table...)
df = DataFrame(
    name=tmat[1, :],
    method=tmat[3, :],
    k=string.(tmat[4, :]),
    ϵ=tmat[5, :]
)

CSV.write("/tmp/linsys.csv", df)

"""
import pandas as pd
df = pd.read_csv("/tmp/linsys.csv")
print(
    df.set_index(["name", "method"]
).to_latex(
    multirow=True, 
    longtable=True,
    float_format="%.1e"
)
)
"""