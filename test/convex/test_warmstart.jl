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
using LoopVectorization
using LIBSVMFileIO

bool_opt = false
bool_plot = true
bool_q_preprocessed = false
f1(A, d=2) = sqrt.(sum(abs2.(A), dims=d))

ε = 1e-6 # * max(g(x0) |> norm, 1)
λ = 1e-7
if bool_q_preprocessed
    # name = "a4a"
    # name = "a9a"
    # name = "w4a"
    # name = "covtype"
    # name = "news20"
    name = "rcv1"

    # loss


    X, y = libsvmread("test/instances/libsvm/$name.libsvm"; dense=false)
    Xv = hcat(X...)'
    Rc = 1 ./ f1(Xv)[:]
    Xv = (Rc |> Diagonal) * Xv
    X = Rc .* X

    if name in ["covtype"]
        y = convert(Vector{Float64}, (y .- 1.5) * 2)
    else
    end
    @info begin
        a = ccall((:openblas_get_num_threads64_, Base.libblas_name), Cint, ())
        if a > 8
            BLAS.set_num_threads(8)
            a = ccall((:openblas_get_num_threads64_, Base.libblas_name), Cint, ())
        end
        "using BLAS threads $a"
    end
    @info "data reading finished"

    # precompute Q Matrix
    Qc(x, y) = y^2 * x
    bool_q_preprocessed && (P = Qc.(X, y))
    Pv = hcat(P...)'

    n = Xv[1, :] |> length
    Random.seed!(1)
    N = y |> length

    x0 = 1 * randn(Float64, n)

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

    rc = PFH(name=Symbol("PF-HSODM"))(;
        x0=copy(x0), f=loss, g=g, hvp=hvpdiff,
        maxiter=10000, tol=ε, freq=1,
        step=:hsodm, μ₀=5e-2,
        bool_trace=true,
        maxtime=500,
        direction=:cold
    )
    rh = PFH(name=Symbol("PF-HSODM"))(;
        x0=copy(x0), f=loss, g=g, hvp=hvpdiff,
        maxiter=10000, tol=ε, freq=1,
        step=:hsodm, μ₀=5e-2,
        bool_trace=true,
        maxtime=500,
        direction=:warm
    )
end


if bool_plot
    results = [
        # optim_to_result(res1, "GD+Wolfe"),
        # optim_to_result(r_lbfgs, "LBFGS+Wolfe"),
        rc,
        rh,
    ]
    linestyles = [:dash, :dot, :dashdot, :dashdotdot]
    method_names = [
        L"\texttt{no warm-start}",
        L"\texttt{warm-start}",
    ]
    xaxis = :k
    for metric in (:ϵ, :kᵥ)
        @printf("plotting results\n")

        pgfplotsx()
        title = L"Warm-start for Homotopy HSODM on $\texttt{name}:=\texttt{%$(name)}$"
        fig = plot(
            xlabel=xaxis == :t ? L"\textrm{Running Time (s)}" : L"\textrm{Iterations} $k$",
            ylabel=metric == :ϵ ? L"\|\nabla f\|" : L"Krylov Iterations $K$",
            title=title,
            size=(600, 500),
            yticks=[1e-12, 1e-10, 1e-8, 1e-6, 1e-4, 1e-2, 1e-1, 1e0, 1e1],
            # xticks=[1, 10, 100, 200, 500, 1000, 10000, 100000, 1e6],
            yscale=:log10,
            # xscale=:log2,
            dpi=500,
            labelfontsize=14,
            xtickfont=font(13),
            ytickfont=font(13),
            leg=:topleft,
            legendfontsize=14,
            legendfontfamily="sans-serif",
            titlefontsize=22,
            palette=:Paired_8 #palette(:PRGn)[[1,3,9,10,11]]
        )
        for (k, rv) in enumerate(results)
            yv = getresultfield(rv, metric)
            plot!(fig,
                xaxis == :t ? getresultfield(rv, :t) : (1:(yv|>length)),
                yv,
                label=method_names[k],
                linewidth=1.5,
                # markershape=:circle,
                # markersize=2.0,
                # markercolor=:match,
                # linestyle=linestyles[k]
                # seriescolor=colors[k]
            )
        end
        savefig(fig, "/tmp/$metric-warmstart-$name-$xaxis.tex")
        savefig(fig, "/tmp/$metric-warmstart-$name-$xaxis.pdf")
        savefig(fig, "/tmp/$metric-warmstart-$name-$xaxis.png")
    end
end

