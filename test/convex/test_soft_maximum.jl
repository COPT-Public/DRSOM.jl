###############
# project: DRSOM
# created Date: Tu Mar 2022

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
using SparseArrays
using .LP


using LIBSVMFileIO

bool_plot = false
bool_opt = false
bool_setup = true

if bool_setup
    Random.seed!(2)
    n = 500
    m = 1000
    μ = 5e-2
    X = [rand(Float64, n) * 2 .- 1 for _ in 1:m]
    Xm = hcat(X)'
    y = rand(Float64, m) * 2 .- 1
    # y = max.(y, 0)
    # loss
    x0 = ones(n) / 10
    function loss_orig(w)
        loss_single(x, y0) = exp((w' * x - y0) / μ)
        _pure = loss_single.(X, y) |> sum
        return μ * log(_pure)
    end
    function grad_orig(w)
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
        return Symmetric(sparse(1 / μ * (Xm' * Diagonal(π0) * Xm - Xm' * π0 * π0' * Xm)))
    end
    ∇₀ = grad_orig(zeros(x0 |> size))
    grad(w) = grad_orig(w) - ∇₀
    loss(w) = loss_orig(w) - ∇₀'w
    ε = 1e-5
end
if bool_opt
    # compare with GD and LBFGS, Trust region newton,
    options = Optim.Options(
        g_tol=ε,
        iterations=200,
        store_trace=true,
        show_trace=true,
        show_every=10,
        time_limit=500
    )
    r_lbfgs = Optim.optimize(
        loss, grad, x0,
        LBFGS(; alphaguess=LineSearches.InitialStatic(),
            linesearch=LineSearches.BackTracking()), options;
        inplace=false
    )
    r_newton = Optim.optimize(
        loss, grad, hess, x0,
        Newton(; alphaguess=LineSearches.InitialStatic()
        ), options;
        inplace=false
    )
    r = HSODM()(;
        x0=copy(x0), f=loss, g=grad, H=hess,
        maxiter=10000, tol=ε, freq=1,
        direction=:warm, linesearch=:hagerzhang
    )
    r.name = "Adaptive HSODM"
    rh = PFH()(;
        x0=copy(x0), f=loss, g=grad, H=hess,
        maxiter=10000, tol=ε, freq=1,
        step=:hsodm, μ₀=5e-1,
        bool_trace=true,
        maxtime=10000,
        direction=:warm
    )

    ru = UTR(;)(;
        x0=copy(x0), f=loss, g=grad, H=hess,
        maxiter=10000, tol=1e-6, freq=1,
        direction=:warm, bool_subp_exact=1
    )
end

if bool_plot

    results = [
        optim_to_result(r_lbfgs, "LBFGS"),
        optim_to_result(r_newton, "Newton's method"),
        r,
        rh
    ]
    method_names = getname.(results)
    # for metric in (:ϵ, :fx)
    for metric in [:fx]
        # metric = :ϵ
        method_objval_ragged = rstack([
                getresultfield.(results, metric)...
            ]; fill=NaN
        )


        @printf("plotting results\n")

        pgfplotsx()
        title = L"Soft Maximum $n:=$%$(n), $m:=$%$(m), $\mu:=$%$(μ)"
        fig = plot(
            1:(method_objval_ragged|>size|>first),
            method_objval_ragged,
            label=permutedims(method_names),
            # xscale=:log2,
            # yscale=:log10,
            xlabel="Iteration",
            ylabel=metric == :ϵ ? L"\|\nabla f\| = \epsilon" : L"f(x)",
            title=title,
            size=(1100, 500),
            # yticks=[1e-7, 1e-6, 1e-4, 1e-2, 1e-1, 1e0, 1e1],
            xticks=[1, 10, 100, 200, 500, 1000, 10000, 100000, 1e6],
            dpi=500,
            labelfontsize=16,
            xtickfont=font(13),
            ytickfont=font(13),
            leg=:topright,
            legendfontsize=24,
            legendfontfamily="sans-serif",
            titlefontsize=24,
        )

        savefig(fig, "/tmp/$metric-softmaximum.pdf")

    end
end