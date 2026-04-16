#############################################
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
using ADNLPModels
using LaTeXStrings
using LinearAlgebra
using Statistics
using LinearOperators
using Optim
using SparseArrays
using .LP
using LIBSVMFileIO

bool_q_preprocessed = true
bool_opt = true
bool_plot = true
f1(A, d=2) = sqrt.(sum(abs2.(A), dims=d))

ε = 1e-6 # * max(g(x0) |> norm, 1)
λ = 1e-9
if bool_q_preprocessed
    name = "a4a"
    # name = "a9a"
    # name = "w4a"
    # name = "w8a"
    # name = "covtype"
    # name = "news20"
    # name = "rcv1"

    X, y = libsvmread("test/instances/$name.libsvm"; dense=false)
    Is = vcat([x.nzind for (j, x) in enumerate(X)]...)
    Js = vcat([j * ones(Int, length(x.nzind)) for (j, x) in enumerate(X)]...)
    Vs = vcat([x.nzval for (j, x) in enumerate(X)]...)
    Xv = sparse(Is, Js, Vs)'
    Rc = 1 ./ f1(Xv)[:]
    Xv = (Rc |> Diagonal) * Xv
    X = Rc .* X

    if name in ["covtype"]
        y = convert(Vector{Float64}, (y .- 1.5) * 2)
    else
    end

    @info "data reading finished"

    # precompute Q Matrix
    Pv = y .^ 2 .* Xv

    n = Xv[1, :] |> length
    Random.seed!(1)
    N = y |> length

    x0 = 10 * randn(Float64, n)

    function loss(w)
        z = log.(1 .+ exp.(-y .* (Xv * w))) |> sum
        return z / N + 0.5 * λ * w'w
    end
    function g(w)
        z = exp.(-y .* (Xv * w))
        fq = -z ./ (1 .+ z)
        return Xv' * (fq .* y) / N + λ * w
    end

    function H(w)
        z = exp.(y .* (Xv * w))
        fq = z ./ (1 .+ z) .^ 2
        return ((fq .* Pv)' * Xv ./ N) + λ * I
    end

    function hvp(w, v, Hv)
        z = exp.(y .* (Xv * w))
        fq = z ./ (1 .+ z) .^ 2
        copyto!(Hv, (fq .* Pv)' * (Xv * v) ./ N .+ λ .* v)
    end

    function hvpdiff(w, v, Hv; eps=1e-5)
        gn = g(w + eps * v)
        gf = g(w)
        copyto!(Hv, (gn - gf) / eps)
    end

    @info "data preparation finished"

    options = Optim.Options(
        g_tol=ε,
        iterations=10000,
        store_trace=true,
        show_trace=true,
        show_every=1,
        time_limit=500
    )
    nlp = ADNLPModel(x -> loss(x), copy(x0))
end


if bool_opt

    # options for Optim.jl package
    options = Optim.Options(
        g_tol=ε,
        iterations=10000,
        store_trace=true,
        show_trace=true,
        show_every=1,
        time_limit=500
    )

    # rn2 = PFH(name=Symbol("iNewton-1e-8"))(;
    #     x0=copy(x0), f=loss, g=g, hvp=hvpdiff,
    #     maxiter=60, tol=ε, freq=1,
    #     step=:newton, μ₀=0.0,
    #     maxtime=500, linesearch=:backtrack,
    #     bool_trace=true,
    #     eigtol=1e-8,
    #     direction=:warm
    # )
    # rn3 = PFH(name=Symbol("iNewton-1e-9"))(;
    #     x0=copy(x0), f=loss, g=g, hvp=hvpdiff,
    #     maxiter=60, tol=ε, freq=1,
    #     step=:newton, μ₀=0.0,
    #     maxtime=500, linesearch=:backtrack,
    #     bool_trace=true,
    #     eigtol=1e-9,
    #     direction=:warm
    # )

    # ra = HSODM(name=Symbol("adaptive-HSODM"))(;
    #     x0=copy(x0), f=loss, g=g, hvp=hvpdiff,
    #     maxiter=10000, tol=ε, freq=1,
    #     maxtime=500,
    #     direction=:warm, linesearch=:hagerzhang,
    #     adaptive=:none
    # )

    rv = UTR(name=Symbol("Universal-TRS"))(;
        x0=copy(x0), f=loss, g=g, H=H,
        maxiter=300, tol=1e-5, freq=1,
        bool_trace=true,
        subpstrategy=:direct,
        initializerule=:mishchenko,
        # adaptiverule=:classic
    )
end


if bool_plot

    ratio = (getresultfield(rv, :θ) + getresultfield(rv, :σ) .* sqrt.(getresultfield(rv, :ϵ))) ./ getresultfield(rv, :Δₙ)
    pgfplotsx()
    # title = L"Logistic Regression $\texttt{name}:=\texttt{%$(name)}$, $n:=$%$(n), $N:=$%$(N)"
    fig = plot(
        xlabel=L"\textrm{Iterations}",
        ylabel=L"\frac{\mu}{\|d\|}",
        title="",
        size=(600, 500),
        yticks=[1e-12, 1e-10, 1e-8, 1e-6, 1e-4, 1e-2, 1e-1, 1e0, 1e1],
        # xticks=[1, 10, 100, 200, 500, 1000, 10000, 100000, 1e6],
        yscale=:log10,
        # xscale=:log2,
        dpi=500,
        labelfontsize=14,
        xtickfont=font(13),
        ytickfont=font(13),
        # leg=:bottomleft,
        legendfontsize=14,
        legendfontfamily="sans-serif",
        titlefontsize=22,
        palette=:Paired_8 #palette(:PRGn)[[1,3,9,10,11]]
    )

    plot!(fig,
        ratio,
        label="",
        linewidth=1.5,
        markershape=:circle,
        markersize=2.0,
        markercolor=:match,
        # linestyle=linestyles[k]
        # seriescolor=colors[k]
    )
    savefig(fig, "/tmp/ratio.tex")
    savefig(fig, "/tmp/ratio.pdf")
    savefig(fig, "/tmp/ratio.png")

end
