###############
# file: test_lp_smooth.jl
# project: src
# created Date: Tu Mar yyyy
# author: <<author>
# -----
# last Modified: Mon Apr 18 2022
# modified By: Chuwen Zhang
# -----
# (c) 2022 Chuwen Zhang
# -----
# HISTORY:
# Date      	By	Comments
# ----------	---	---------------------------------------------------------
###############

include("helper.jl")
include("helper_plus.jl")
include("helper_l.jl")
include("helper_c.jl")
include("lp.jl")

using ProximalOperators
using DRSOM
using ProximalAlgorithms
using Random
using Distributions
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
using .LP
using .drsom_helper
using .drsom_helper_plus
using .drsom_helper_l
using .drsom_helper_c

params = LP.parse_commandline()
m = n = params.n
D = Normal(0.0, 1.0)
A = rand(D, (n, m)) .* rand(Bernoulli(0.85), (n, m))
v = rand(D, m) .* rand(Bernoulli(0.5), m)
b = A * v + rand(D, (n))
x0 = zeros(m)

Q = A' * A
h = Q' * b
# find Lipschitz constant
L, _ = LinearOperators.normest(Q, 1e-4)
# find convexity constant
# vals, vecs, info = KrylovKit.eigsolve(A'A, 1, :SR)
σ = 0.0
@printf("preprocessing finished\n")

########################################################
# direct evaluation
# Q = Matrix{Float64}([1 0; 0 -1])
# h = Vector{Float64}([0; 0])
# x0 = Vector{Float64}([1; 0])
f_composite(x) = 1 / 2 * x' * Q * x - h' * x
g(x) = Q * x - h
H(x) = Q


options = Optim.Options(
    g_tol=1e-6,
    iterations=10000,
    store_trace=true,
    show_trace=true,
    show_every=10,
    time_limit=500
)
method_objval = Dict{String,AbstractArray{Float64}}()
method_state = Dict{String,Any}()


r = drsom_helper.run_drsomd(
    copy(x0), f_composite, g, H,
    maxiter=1000, tol=1e-6
)

rp1 = drsom_helper_c.run_drsomd(
    copy(x0), f_composite, g, H;
    maxiter=1000, tol=1e-6, direction=:gaussian
)
rp = drsom_helper_plus.run_drsomd(
    copy(x0), f_composite, g, H;
    maxiter=1000, tol=1e-6, direction=:krylov
)
rp2 = drsom_helper_plus.run_drsomd(
    copy(x0), f_composite, g, H;
    maxiter=1000, tol=1e-6, direction=:homokrylov
)
cg = Optim.optimize(f_composite, g, x0, ConjugateGradient(; eta=0.01), options; inplace=false)
# rl = drsom_helper_l.run_drsomb(
#     copy(x0), f_composite;
#     maxiter=1000, tol=1e-6, direction=:gaussian
# )
# rl = drsom_helper_l.run_drsomb(
#     copy(x0), f_composite;
#     maxiter=1000, tol=1e-6, direction=:gaussian,
#     hessian_rank=:∞
# )
results = [
    optim_to_result(cg, "CG-Hager-Zhang"),
    r,
    rp,
    rp1,
    rp2
]


method_objval_ragged = rstack([
        getresultfield.(results, :ϵ)...
    ]; fill=NaN
)
method_names = getname.(results)


@printf("plotting results\n")

pgfplotsx()
title = L"\textrm{ Convex QP}: Q \in \mathcal S_+^{%$(n)\times %$(n)}"
fig = plot(
    1:(method_objval_ragged|>size|>first),
    method_objval_ragged,
    label=permutedims(method_names),
    xscale=:log10,
    yscale=:log10,
    xlabel="Iteration", ylabel=L"\|\nabla f\| = \epsilon",
    title=title,
    size=(1280, 720),
    yticks=[1e-10, 1e-8, 1e-6, 1e-4, 1e-2, 1e2],
    xticks=[1, 10, 100, 200, 500, 1000, 10000, 100000, 1e6],
    dpi=1000,
)

savefig(fig, "/tmp/random-qp-$(n).pdf")

