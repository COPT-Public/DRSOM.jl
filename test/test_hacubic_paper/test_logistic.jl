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

bool_q_preprocessed = bool_opt = true
bool_plot = true
f1(A, d=2) = sqrt.(sum(abs2.(A), dims=d))

ε = 5e-8 # * max(g(x0) |> norm, 1)
# λ = 1e-5
# Aₗ = 1e-7
λ = 5e-6
Aₗ = 5e-7

if bool_q_preprocessed
    results = []
    # name = "splice"
    # name = "mushroom"
    # name = "a4a"
    # name = "a9a"
    # name = "w4a"
    name = "w8a"
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

    x0 = 100 * randn(Float64, n)
    @info "x0: $(x0 |> norm)"
    # x0 = 50 * ones(Float64, n)

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

    function hvpdiff(w, v, Hv; eps=1e-6)
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
    K = 100
    # options for Optim.jl package
    options = Optim.Options(
        g_tol=ε,
        iterations=10000,
        store_trace=true,
        show_trace=true,
        show_every=1,
        time_limit=500
    )
    rc1 = HaCubic(name=Symbol("HaCubic"))(;
        x0=copy(x0), f=loss, g=g, H=H,
        maxiter=K, tol=ε, freq=1,
        bool_trace=true,
        subpstrategy=:direct,
        A₀=Aₗ,
        α=1.1,
        memory=5,
        memory_type=:i,
    )
    rc2 = HaCubic(name=Symbol("HaCubic"))(;
        x0=copy(x0), f=loss, g=g, H=H,
        maxiter=K, tol=ε, freq=1,
        bool_trace=true,
        subpstrategy=:direct,
        A₀=Aₗ,
        α=1.1,
        memory=3,
        memory_type=:ii,
    )
    # rc3 = HaCubic(name=Symbol("HaCubic"))(;
    #     x0=copy(x0), f=loss, g=g, H=H,
    #     maxiter=K, tol=ε, freq=1,
    #     bool_trace=true,
    #     subpstrategy=:direct,
    #     A₀=1e-7,
    #     α=1.1,
    #     memory=7,
    #     memory_type=:i,
    # )
    # rc4 = HaCubic(name=Symbol("HaCubic"))(;
    #     x0=copy(x0), f=loss, g=g, H=H,
    #     maxiter=K, tol=ε, freq=1,
    #     bool_trace=true,
    #     subpstrategy=:direct,
    #     A₀=1e-7,
    #     α=1.1,
    #     memory=7,
    #     memory_type=:ii,
    # )
    rc5 = HaCubic(name=Symbol("HaCubic"))(;
        x0=copy(x0), f=loss, g=g, H=H,
        maxiter=K, tol=ε, freq=1,
        bool_trace=true,
        subpstrategy=:direct,
        A₀=Aₗ,
        α=1.1,
        memory=10,
        memory_type=:i,
    )
    rc6 = HaCubic(name=Symbol("HaCubic"))(;
        x0=copy(x0), f=loss, g=g, H=H,
        maxiter=K, tol=ε, freq=1,
        bool_trace=true,
        subpstrategy=:direct,
        A₀=Aₗ,
        α=1.1,
        memory=10,
        memory_type=:ii,
    )

    # rac = HaCubic(name=Symbol("HaCubicAccelerated"))(;
    #     x0=copy(x0), f=loss, g=g, H=H,
    #     maxiter=20, tol=ε, freq=1,
    #     bool_trace=true,
    #     subpstrategy=:nesterov,
    #     A₀=1e-7,
    #     α=2.0,
    #     memory=3,
    #     memory_type=:ii,
    # )

    stats, stop = ARCqKOp(
        # stats, stop = ARCqKsparse(
        nlp,
        max_time=1000.0,
        max_iter=10000,
        max_eval=typemax(Int64),
        verbose=true,
        n_listofstates=10000,
        atol=ε,
        # rtol=rtol,
        # @note: how to set |g|?
    )
    rarc = arc_stop_to_result(nlp, stop, :ARC)
end


if bool_plot
    results = [
        # optim_to_result(res1, "GD+Wolfe"),
        # optim_to_result(r_lbfgs, "LBFGS+Wolfe"),
        rarc,
        rc1,
        # rc3,
        rc5,
        rc2,
        # rc4,
        rc6,
        # rac,
    ]
    linestyles = [:dash, :dot, :dashdot, :dashdotdot]
    method_names = [
        L"\texttt{ArC}",
        L"\texttt{HAR-C} (5)",
        # L"\texttt{HAR-C} (7)",
        L"\texttt{HAR-C} (10)",
        L"\texttt{HAR-S} (5)",
        # L"\texttt{HAR-S} (7)",
        L"\texttt{HAR-S} (10)",
        # L"\texttt{HAR-S-Acc} (10)",
    ]
    colors = [
        :black,
        :skyblue1,
        :deepskyblue3,
        :palegreen2,
        :green4,
        :red3
    ]
    # for xaxis in (:t, :k)
    #     for metric in (:ϵ, :fx)
    xaxis = :k
    metric = :ϵ
    @printf("plotting results\n")

    pgfplotsx()
    title = L"Logistic Regression $\texttt{name}:=\texttt{%$(name)}$, $n:=$%$(n), $N:=$%$(N)"
    fig = plot(
        xlabel=xaxis == :t ? L"\textrm{Running Time (s)}" : L"\textrm{Iterations}",
        ylabel=metric == :ϵ ? L"\|\nabla f\|" : L"f(x)",
        title=title,
        size=(600, 500),
        yticks=[1e-12, 1e-10, 1e-8, 1e-6, 1e-4, 1e-2, 1e-1, 1e0, 1e1],
        # xticks=[1, 10, 100, 200, 500, 1000, 10000, 100000, 1e6],
        yscale=:log10,
        # xscale=:log2,
        dpi=500,
        labelfontsize=16,
        xtickfont=font(15),
        ytickfont=font(15),
        # leg=:bottomleft,
        legendfontsize=16,
        legendfontfamily="sans-serif",
        titlefontsize=22,
        palette=palette(:PRGn)[[1, 3, 5, 7, 9, 11, 4]]
        # palette=:Paired_8
    )
    for (k, rv) in enumerate(results)
        yv = getresultfield(rv, metric)
        plot!(fig,
            xaxis == :t ? getresultfield(rv, :t) : (1:(yv|>length)),
            yv,
            label=method_names[k],
            linewidth=3.5,
            markershape=:circle,
            # markersize=2.0,
            # markercolor=:match,
            # linestyle=linestyles[k]
            # seriescolor=colors[k]
            seriescolor=colors[k]
        )
    end
    savefig(fig, "/tmp/e-logistic-$name-$xaxis.tex")
    savefig(fig, "/tmp/e-logistic-$name-$xaxis.pdf")
    savefig(fig, "/tmp/e-logistic-$name-$xaxis.png")
end
