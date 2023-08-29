@doc raw"""
    We test if the last iterate helps in solving 
     the current eigenvalue problem (the benefits of warm-starts).
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
using Statistics
using LinearOperators
using Optim
using SparseArrays
using .LP
using LoopVectorization
using LIBSVMFileIO
using StatsPlots

bool_opt = true
bool_plot = true
bool_q_preprocessed = true
f1(A, d=2) = sqrt.(sum(abs2.(A), dims=d))

ε = 1e-6 # * max(g(x0) |> norm, 1)
λ = 1e-7
if bool_q_preprocessed
    # name = "a4a"
    # name = "a9a"
    # name = "w4a"
    # name = "covtype"
    name = "news20"
    # name = "rcv1"

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
    r = Dict(
        :cold => [],
        :warm => []
    )
    for i in 1:10
        x0 = 2 * randn(Float64, n)
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
        push!(r[:cold], getresultfield(rc,  :kᵥ)) 
        rh = PFH(name=Symbol("PF-HSODM"))(;
            x0=copy(x0), f=loss, g=g, hvp=hvpdiff,
            maxiter=10000, tol=ε, freq=1,
            step=:hsodm, μ₀=5e-2,
            bool_trace=true,
            maxtime=500,
            direction=:warm
        )
        push!(r[:warm], getresultfield(rh,  :kᵥ))
    end
end


if bool_plot
    
    linestyles = [:dash, :dot, :dashdot, :dashdotdot]
    method_id = Dict(
        :warm => 2,
        :cold => 1,
    )
    method_names = Dict(
        :warm => L"\texttt{warm-start}",
        :cold => L"\texttt{no warm-start}",
    )
    colors = palette(:default)[1:2]
    xaxis = :k
    metric = :kᵥ 
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
    for (_, (k, rv)) in enumerate(r)
        yv = rstack(rv..., fill=NaN)
        println(yv)
        errorline!(fig,
            1:60,
            convert(Matrix{Float64}, yv)[1:60,:],
            label=method_names[k],
            linewidth=0.5,
            # errorstyle=:plume,
            groupcolor=colors[method_id[k]],
            percentiles=[:20, :80]
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

