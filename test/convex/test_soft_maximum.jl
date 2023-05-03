###############
# project: RSOM
# created Date: Tu Mar 2022
# author: <<author>
# -----
# last Modified: Mon Apr 18 2022
# modified By: Chuwen Zhang
# -----
# (c) 2022 Chuwen Zhang
# -----
# A script to test DRSOM on nonconvex logistic regression for 0-1 classification on LIBSVM
# @reference:
# 1. Zhu, X., Han, J., Jiang, B.: An Adaptive High Order Method for Finding Third-Order Critical Points of Nonconvex Optimization, http://arxiv.org/abs/2008.04191, (2020)
###############


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
using .LP


using LIBSVMFileIO
n = 200
m = 500
μ = 0.05
X = [rand(Float64, n) * 2 .- 1 for _ in 1:m]
Xm = hcat(X)'
y = rand(Float64, m) * 2 .- 1
# y = max.(y, 0)
# loss
x0 = ones(n)
function loss(w)
    loss_single(x, y0) = exp((w' * x - y0) / μ)
    _pure = loss_single.(X, y) |> sum
    return μ * log(_pure)
end
function grad(w)
    a = (Xm * w - y) / μ
    ax = exp.(a)
    π0 = ax / (ax |> sum)
    ∇ = Xm' * π0
    return ∇
end
function hess(w)
    a = (Xm * w - y) / μ
    ax = exp.(a)
    π0 = ax / (ax |> sum)
    return 1 / μ * (Xm' * Diagonal(π0) * Xm - Xm' * π0 * π0' * Xm)
end



# g(x) = DRSOM.ForwardDiff.gradient(loss, x)
# H(x) = DRSOM.ForwardDiff.hessian(loss, x)

# compare with GD and LBFGS, Trust region newton,
options = Optim.Options(
    g_tol=1e-6,
    iterations=10000,
    store_trace=true,
    show_trace=true,
    show_every=10,
    time_limit=500
)

# res3 = Optim.optimize(f_composite, g, H, x0, NewtonTrustRegion(), options; inplace=false)
# res2 = Optim.optimize(
#     loss, x0, LBFGS(;
#         linesearch=LineSearches.StrongWolfe()
#     ), options;
#     autodiff=:forward)
r_newton = Optim.optimize(
    loss, grad, hess, x0,
    Newton(; alphaguess=LineSearches.InitialStatic(),
        linesearch=LineSearches.HagerZhang()), options;
    inplace=false
)
r = HSODM()(;
    x0=copy(x0), f=loss, g=grad, H=hess,
    maxiter=10000, tol=1e-6, freq=1,
    direction=:warm, linesearch=:hagerzhang
)
ra = HSODM(name=:HSODM_adaptive)(;
    x0=copy(x0), f=loss, g=grad, H=hess,
    maxiter=10000, tol=1e-6, freq=1,
    direction=:warm, linesearch=:hagerzhang, adaptive=:angle
)



results = [
    # optim_to_result(res1, "GD+Wolfe"),
    # optim_to_result(res2, "LBFGS+Wolfe"),
    optim_to_result(r_newton, "Newton's method"),
    r,
    ra
]
method_names = getname.(results)

for metric in (:ϵ, :fx)
    method_objval_ragged = rstack([
            getresultfield.(results, metric)...
        ]; fill=NaN
    )


    @printf("plotting results\n")

    pgfplotsx()
    title = L"111"
    fig = plot(
        1:(method_objval_ragged|>size|>first),
        method_objval_ragged,
        label=permutedims(method_names),
        xscale=:log2,
        yscale=:log10,
        xlabel="Iteration",
        ylabel=metric == :ϵ ? L"\|\nabla f\| = \epsilon" : L"f(x)",
        title=title,
        size=(1280, 720),
        yticks=[1e-10, 1e-8, 1e-6, 1e-4, 1e-2, 1e2],
        xticks=[1, 10, 100, 200, 500, 1000, 10000, 100000, 1e6],
        dpi=1000,
    )

    savefig(fig, "/tmp/$metric-softmaximum.pdf")

end
