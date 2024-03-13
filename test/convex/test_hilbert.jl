@doc raw"""
    We test the eigensolve is stable at an ill-conditioned Hilbert matrix.
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
using SpecialMatrices
Base.@kwdef mutable struct KrylovInfo
    normres::Float64
    numops::Float64
end

n = 100
table = []

Random.seed!(1)


H = Hilbert(n)
# H = rand(Float64, n, n)
# H = H'H + 1e-6 * I
g = rand(Float64, n)

for δ in [1e-5, 1e-7, 1e-9, 1e-10]

    r = Dict()
    # 
    r["GHM (Lanczos)"] = KrylovInfo(normres=0.0, numops=0)
    r["CG"] = KrylovInfo(normres=0.0, numops=0)
    r["GMRES"] = KrylovInfo(normres=0.0, numops=0)
    r["rGMRES"] = KrylovInfo(normres=0.0, numops=0)
    samples = 5
    κ = LinearAlgebra.cond(H + δ .* LinearAlgebra.I)
    for idx in 1:samples
        w₀ = rand(Float64, n)
        Fw(w) = [H * w[1:end-1] + g .* w[end]; w[1:end-1]' * g - δ * w[end]]
        Hw(w) = (H + δ .* LinearAlgebra.I) * w
        Fc = DRSOM.Counting(Fw)
        Hc = DRSOM.Counting(Hw)
        @info "data reading finished"

        max_iteration = 200
        ε = 1e-7
        gₙ = g |> norm
        εₙ = 1e-4 * gₙ

        rl = KrylovKit.eigsolve(
            Fc, [w₀; 1], 1, :SR, Lanczos(tol=ε, maxiter=max_iteration, verbosity=3, eager=true);
        )
        λ₁ = rl[1]
        ξ₁ = rl[2][1]

        r["GHM (Lanczos)"].normres += (Fc(ξ₁) - λ₁ .* ξ₁) |> norm
        r["GHM (Lanczos)"].numops += rl[end].numops

        rl = KrylovKit.linsolve(
            Hc, -g, w₀, CG(; tol=εₙ, maxiter=max_iteration, verbosity=3);
        )
        r["CG"].normres += ((Hw(rl[1]) + g) |> norm)
        r["CG"].numops += rl[end].numops
        rl = KrylovKit.linsolve(
            Hc, -g, w₀, GMRES(; tol=εₙ, maxiter=1, krylovdim=max_iteration, verbosity=3);
        )
        r["GMRES"].normres += ((Hw(rl[1]) + g) |> norm)
        r["GMRES"].numops += rl[end].numops
        rl = KrylovKit.linsolve(
            Hc, -g, w₀, GMRES(; tol=εₙ, maxiter=4, krylovdim=div(max_iteration, 4), verbosity=3);
        )
        r["rGMRES"].normres += ((Hw(rl[1]) + g) |> norm)
        r["rGMRES"].numops += rl[end].numops
    end

    for (k, v) in r
        push!(table, [δ, κ, k, v.numops / samples, v.normres / samples])
    end
end

tmat = hcat(table...)
delta = [@sprintf "%0.1e" k for k in tmat[1, :]]
kappa = [@sprintf "%0.1e" k for k in tmat[2, :]]

df = DataFrame(
    delta=[L"$\delta=$%$k" for k in delta],
    kappa=[L"$\kappa_H=$%$k" for k in kappa],
    method=tmat[3, :],
    k=tmat[4, :],
    ϵ=tmat[5, :]
)

CSV.write("/tmp/linsys-hilbert.csv", df)

"""
import pandas as pd
df = pd.read_csv("/tmp/linsys-hilbert.csv")
print(df.set_index(["delta", "kappa", "method"]).to_latex(multirow=True, longtable=True))
"""

using StatsPlots
pgfplotsx()
fig = groupedbar(
    df.method,
    convert(Vector{Float64}, df.k),
    bar_position=:dodge,
    group=df.kappa,
    palette=:Paired_8,
    # xlabel="Method",
    leg=:topright,
    legendfontsize=14,
    labelfontsize=14,
    ylabel=L"Krylov Iterations: $K$"
)

savefig(fig, "/tmp/hilbert.tex")
savefig(fig, "/tmp/hilbert.pdf")
savefig(fig, "/tmp/hilbert.png")