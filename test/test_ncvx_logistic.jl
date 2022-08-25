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

include("helper.jl")
include("helper_plus.jl")
include("helper_l.jl")
include("lp.jl")

using ProximalOperators
using LineSearches
using Optim
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
using ArgParse
using Optim
using .LP
using .drsom_helper
using .drsom_helper_plus
using .drsom_helper_l


using MLDatasets
using Flux
using LIBSVMFileIO


name = "sonar"
# Load data
X, y = libsvmread("example/$name.libsvm"; dense=true)
y = max.(y, 0)
# loss
λ = 1e-5
x0 = zeros(X[1] |> size)
function loss(w)
    loss_single(x, y) = (1 / (1 + exp(-w' * x)) - y)^2
    _pure = loss_single.(X, y) |> sum
    return _pure / 2 + λ / 2 * w' * w
end

#######
iter_scale = 0
#######
# compare with GD and LBFGS, Trust region newton,
options = Optim.Options(
    g_tol=1e-6,
    iterations=10000,
    store_trace=true,
    show_trace=true,
    show_every=10,
    time_limit=500
)


# res1 = Optim.optimize(f_composite, g, x0, GradientDescent(;
#         alphaguess=LineSearches.InitialHagerZhang(),
#         linesearch=LineSearches.StrongWolfe()), options; inplace=false)
# res2 = Optim.optimize(f_composite, g, H, x0, LBFGS(;
#         linesearch=LineSearches.StrongWolfe()), options; inplace=false)
# res3 = Optim.optimize(f_composite, g, H, x0, NewtonTrustRegion(), options; inplace=false)

# res1 = Optim.optimize(
#     loss, x0, GradientDescent(;
#         alphaguess=LineSearches.InitialHagerZhang(),
#         linesearch=LineSearches.StrongWolfe()
#     ), options;
#     autodiff=:forward)
# res2 = Optim.optimize(
#     loss, x0, LBFGS(;
#         linesearch=LineSearches.StrongWolfe()
#     ), options;
#     autodiff=:forward)
res3 = Optim.optimize(
    loss, x0, NewtonTrustRegion(), options;
    autodiff=:forward
)
r = drsom_helper.run_drsomb(
    copy(x0), loss;
    maxiter=10000, tol=1e-7
)

rl = drsom_helper_l.run_drsomb(
    copy(x0), loss;
    maxiter=1000, tol=1e-6,
    hessian=:sr1, hessian_rank=:∞
)
rb = drsom_helper_l.run_drsomb(
    copy(x0), loss;
    maxiter=1000, tol=1e-6,
    hessian=:bfgs, hessian_rank=:∞
)

results = [
    # optim_to_result(res1, "GD+Wolfe"),
    # optim_to_result(res2, "LBFGS+Wolfe"),
    optim_to_result(res3, "Newton-TR*(Analytic)"),
    r,
    # rpk, rpg,
    # rlr,
    rl,
    rb
]

metric = :ϵ
# metric = :fx
method_objval_ragged = rstack([
        getresultfield.(results, metric)...
    ]; fill=NaN
)
method_names = getname.(results)

@printf("plotting results\n")

pgfplotsx()
title = L"\min _{w \in {R}^{d}} \frac{1}{2} \sum_{i=1}^{n}\left(\frac{1}{1+e^{-w^{\top} x_{i}}}-y_{i}\right)^{2}+\frac{\alpha}{2}\|w\|^{2}"
fig = plot(
    1:(method_objval_ragged|>size|>first),
    method_objval_ragged,
    label=permutedims(method_names),
    xscale=:log2,
    yscale=:log10,
    xlabel="Iteration", ylabel=L"\|\nabla f\| = \epsilon",
    title=title,
    size=(1280, 720),
    yticks=[1e-10, 1e-8, 1e-6, 1e-4, 1e-2, 1e2],
    xticks=[1, 10, 100, 200, 500, 1000, 10000, 100000, 1e6],
    dpi=1000,
)

savefig(fig, "/tmp/$metric-ncvx-logistic-$name.pdf")

