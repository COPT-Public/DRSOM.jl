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
using ForwardDiff
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

bool_gen = true
bool_solve = true

if bool_gen
    f1(A, d=2) = sqrt.(sum(abs2.(A), dims=d))
    Random.seed!(1)
    m = 6000
    n = 10000
    nnz = 0.3
    
    @info "data reading start" m n nnz
    name = "random"
    A = sprand(Float64, m, n, nnz)
    wₛ = rand(Float64, n)
    wₛ ./= norm(wₛ)
    y = A * wₛ
    ϵ(w) = A*w - y
    f(w) = 0.5*sum(abs2,ϵ(w))
    fₛ = f(wₛ)
    
    g(w) = ForwardDiff.gradient(f, w)
    
    γ = 0.0 # 1e-10
    Random.seed!(1)
    N = y |> length
    # Q = A' * A
    # function gfs(w)
    #     return Q * w - A' * y
    # end
    r = Dict()
    # 
    r["GHM-Lanczos"] = KrylovInfo(normres=0.0, numops=0)
    r["Newton-CG"] = KrylovInfo(normres=0.0, numops=0)
    r["Newton-GMRES"] = KrylovInfo(normres=0.0, numops=0)
    r["Newton-rGMRES"] = KrylovInfo(normres=0.0, numops=0)
    w₀ = rand(Float64, n)
    w₀ ./= norm(w₀)
    r₀ = ϵ(w₀)
    δ₀ = norm(r₀)^2
    g₀ = A'r₀

    Fw(w) = [
        (A' * ϵ(w[1:end-1]) + A'y + g₀  * w[end]); 
        (w[1:end-1]' * g₀  + δ₀ * w[end])
    ]
    Fc = DRSOM.Counting(Fw)
    

end
Fc([w₀; 1])
@info "data reading finished" m n nnz
if bool_solve

    max_iteration = 15
    ε = 1e-7

    ts = time()
    rl = KrylovKit.eigsolve(
        Fc, [w₀; 1], 1, :SR, 
        Lanczos(
            tol=ε, 
            maxiter=100, 
            verbosity=2, 
            eager=true
        );
    )
    λ₁ = rl[1]
    ξ₁ = rl[2][1]

    x = ξ₁[1:n] / ξ₁[n+1]

    r["GHM-Lanczos"].normres += norm(A * x + r₀)
    r["GHM-Lanczos"].numops += rl[end].numops

    te = time()

    @info "result" r["GHM-Lanczos"] (te - ts)

    if false
        rl = KrylovKit.linsolve(
            Hc, -g, w₀, CG(; tol=ε, maxiter=max_iteration, verbosity=3);
        )
        r["Newton-CG"].normres += ((hvp(rl[1]) + g) |> norm)
        r["Newton-CG"].numops += rl[end].numops
        rl = KrylovKit.linsolve(
            Hc, -g, w₀, GMRES(; tol=ε, maxiter=1, krylovdim=max_iteration, verbosity=1);
        )
        r["Newton-GMRES"].normres += ((hvp(rl[1]) + g) |> norm)
        r["Newton-GMRES"].numops += rl[end].numops
        rl = KrylovKit.linsolve(
            Hc, -g, w₀, GMRES(; tol=ε, maxiter=4, krylovdim=div(max_iteration, 4), verbosity=3);
        )
        r["Newton-rGMRES"].normres += ((hvp(rl[1]) + g) |> norm)
        r["Newton-rGMRES"].numops += rl[end].numops
    end
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
        push!(table, [name, n, k, v.numops, v.normres])
    end
    # end

    tmat = hcat(table...)
    df = DataFrame(
        name=tmat[1, :],
        method=tmat[3, :],
        k=string.(tmat[4, :]),
        ϵ=tmat[5, :]
    )

    CSV.write("/tmp/linsys1.csv", df)

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
end