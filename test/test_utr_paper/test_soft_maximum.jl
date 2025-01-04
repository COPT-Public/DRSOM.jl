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
using LineSearches
using Optim
using ProximalOperators
using ProximalAlgorithms
using Random
using Plots
using Printf
using KrylovKit
using LaTeXStrings
using LinearAlgebra
using Statistics
using LinearOperators
using Optim
using SparseArrays
using .LP


using LIBSVMFileIO

bool_setup = true
bool_opt = true
bool_plot = true

if bool_setup
    Random.seed!(2)
    n = 150
    m = 120
    μ = 0.05
    X = [rand(Float64, n) * 2 .- 1 for _ in 1:m]
    Xm = hcat(X)'
    y = rand(Float64, m) * 2 .- 1
    # y = max.(y, 0)
    # loss
    x0 = zeros(n)
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
    X = [x - ∇₀ for x in X]
    Xm = hcat(X)'
    grad(w) = grad_orig(w)
    loss(w) = loss_orig(w)
    ε = 1e-5
    f₊ = loss(x0)
end
if bool_opt
    results = Dict()
    x₀ = 1.0 * ones(n)

    Mₕ(x) = 2 / μ^2
    results["UTR"] = rd = ATR(name=Symbol("ATR"))(;
        x0=copy(x₀), f=loss, g=grad, H=hess,
        maxiter=2000, tol=1e-4, freq=20,
        bool_trace=true,
        subpstrategy=:direct,
        initializerule=:given,
        adaptiverule=:constant,
        ratio_σ=1.0,
        ratio_Δ=1.0,
        Mₕ=Mₕ
    )

    results["ATR"] = rd = ATR(name=Symbol("ATR"))(;
        x0=copy(x₀), f=loss, g=grad, H=hess,
        maxiter=2000, tol=1e-8, freq=20,
        bool_trace=true,
        subpstrategy=:direct,
        initializerule=:mishchenko,
        ratio_σ=1.0,
        ratio_Δ=1.0,
        Mₕ=Mₕ
    )

    results["Cubic"] = rd = CubicRegularizationVanilla(name=Symbol("Cubic"))(;
        x0=copy(x₀), f=loss, g=grad, H=hess,
        maxiter=2000, tol=1e-4, freq=20,
        bool_trace=true,
        subpstrategy=:direct,
        initializerule=:given,
        Mₕ=Mₕ
    )


    results["Cubic (Acc)"] = rd = CubicRegularizationVanilla(name=Symbol("Cubic"))(;
        x0=copy(x₀), f=loss, g=grad, H=hess,
        maxiter=2000, tol=1e-4, freq=20,
        bool_trace=true,
        subpstrategy=:nesterov,
        initializerule=:given,
        Mₕ=Mₕ
    )

end


if bool_plot

    linestyles = [:dash, :dot, :dashdot, :dashdotdot]
    xaxis = :k
    metric = :fx
    @printf("plotting results\n")

    pgfplotsx()
    title = L"Soft Maximum: $n:=$%$(n), $m:=$%$(m), $\mu:=$%$(μ)"
    fig = plot(
        xlabel=L"\textrm{Iterations}",
        ylabel=L"f(x) - f(x^*)",
        title=title,
        size=(600, 500),
        yticks=[1e-12, 1e-10, 1e-8, 1e-6, 1e-4, 1e-2, 1e-1, 1e0, 1e1],
        # xticks=[1, 10, 100, 200, 500, 1000, 10000, 100000, 1e6],
        yscale=:log10,
        dpi=500,
        labelfontsize=14,
        xtickfont=font(13),
        ytickfont=font(13),
        legendfontsize=14,
        legendfontfamily="sans-serif",
        legendfonthalign=:left,
        titlefontsize=22,
        palette=:Paired_8 #palette(:PRGn)[[1,3,9,10,11]]
    )
    for (k, (nm, rv)) in enumerate(results)
        yv = getresultfield(rv, metric)
        @info "plotting $metric" yv
        plot!(fig,
            xaxis == :t ? getresultfield(rv, :t) : (1:(yv|>length)),
            yv .- f₊ .+ 1e-20,
            label=nm,
            linewidth=2.5,
            # markershape=:circle,
            # markersize=2.0,
            # markercolor=:match,
        )
    end
    savefig(fig, "/tmp/$metric-softmax-$xaxis.png")
    savefig(fig, "/tmp/$metric-softmax-$xaxis.tex")
    savefig(fig, "/tmp/$metric-softmax-$xaxis.pdf")
end