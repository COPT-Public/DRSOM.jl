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

using .LP

using AdaptiveRegularization
using ArgParse
using DRSOM
using Dates
using Distributions
using HTTP
using KrylovKit
using LaTeXStrings
using LazyStack
using LineSearches
using LinearAlgebra
using LinearOperators
using NLPModels
using Optim
using Plots
using Printf
using ProgressMeter
using ProximalAlgorithms
using ProximalOperators
using Random
using Statistics
using Test

using LIBSVMFileIO


name = "a1a"
# Load data
X, y = libsvmread("test/instances/$name.libsvm"; dense=true)
y = max.(y, 0)
# loss
λ = 1e-2
x0 = 4 * ones(X[1] |> size)
function loss(w)
    loss_single(x, y) = (1 / (1 + exp(-w' * x)) - y)^2
    _pure = loss_single.(X, y) |> sum
    return _pure / 2 + λ / 2 * w' * w
end


function _gx(w, x)
    ff = exp(-w' * x)
    qq = 1 / (1 + exp(-w' * x))
    return (qq)^2 * (ff) * x
end

function g(w)
    function _g(x, y)
        # ff = exp(-w' * x)
        qq = 1 / (1 + exp(-w' * x))
        return (qq - y) * _gx(w, x)
    end
    _pure = _g.(X, y) |> sum
    return _pure + λ * w
end
function H(w)
    n = length(w)
    function _H(x, y)
        ff = exp(-w' * x)
        qq = 1 / (1 + exp(-w' * x))

        Hh = (3 * (qq)^2 - 2 * y * qq) * _gx(w, x) * ff * x' + (qq - y) * qq^2 * (ff * I(n) - ff * x * x')
        return Hh
    end
    _pure = _H.(X, y) |> sum
    return _pure + λ * I(n)
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
    show_every=1,
    time_limit=500
)


# res1 = Optim.optimize(
#     loss, g, x0, GradientDescent(;
#         alphaguess=LineSearches.InitialHagerZhang(),
#         linesearch=LineSearches.HagerZhang()
#     ), options;
#     inplace=false
# )
res2 = Optim.optimize(
    loss, g, x0, LBFGS(;
        alphaguess=LineSearches.InitialHagerZhang(),
        linesearch=LineSearches.HagerZhang()
    ), options;
    inplace=false
)
res3 = Optim.optimize(
    loss, g, H, x0,
    NewtonTrustRegion(;
    # alphaguess=LineSearches.InitialStatic(),
    # linesearch=LineSearches.HagerZhang()
    ), options;
    inplace=false
)
r = HSODM()(;
    x0=copy(x0), f=loss, g=g, H=H,
    maxiter=10000, tol=1e-6, freq=1,
    maxtime=10000,
    direction=:warm, linesearch=:hagerzhang
    # adaptive=:ar
)
ra = HSODM(; name=:HSODMArC)(;
    x0=copy(x0), f=loss, g=g, H=H,
    maxiter=10000, tol=1e-7, freq=1,
    maxtime=10000,
    direction=:warm, linesearch=:none,
    adaptive=:ar
)

results = [
    # optim_to_result(res1, "GD+Wolfe"),
    optim_to_result(res2, "LBFGS+HZ"),
    optim_to_result(res3, "Newton-TR*(Analytic)"),
    r,
    ra
    # rpc,
    # rpk, rpg,
    # rlr,
    # rl,
    # rb
]


for metric in (:ϵ, :fx)
    method_objval_ragged = rstack([
            getresultfield.(results, metric)...
        ]; fill=NaN
    )
    method_names = getname.(results)

    @printf("plotting results\n")

    pgfplotsx()
    title = L"\min _{w \in {R}^{d}} \frac{1}{2} \sum_{i=1}^{n}\left(\frac{1}{1+e^{-w^T x_{i}}}-y_{i}\right)^{2}+\frac{\lambda}{2}\|w\|^{2}"
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

    savefig(fig, "/tmp/$metric-ncvx-logistic-$name.pdf")

end