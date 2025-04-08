include("../lp.jl")
include("../tools.jl")
include("../nesterovlb.jl")

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
using AdaptiveRegularization



bool_check = false
bool_solve = true
bool_plot = true
# n, k, p = 150, 50, 2
# n, k, p = 100, 40, 2
n, k, p = 100, 35, 2
K = 3000
data = NesterovHardCase(n, k, p)
c = randn(n)
nlp = get_nlp_nesterovlb(data)

if bool_check
    gf = ForwardDiff.gradient(data.f, c)
    Hf = ForwardDiff.hessian(data.f, c)
    @assert norm(gf - data.g(c)) < 1e-4
    @assert norm(Hf - data.H(c)) < 1e-4
    @assert abs(data.f(data.xₛ) - data.fₛ) < 1e-4
end

if bool_solve
    results = []
    x₀ = zeros(n)
    Mₕ(x) = 2.0^2 * factorial(2)
    rd = ATR(name=Symbol("ATR"))(;
        x0=copy(data.x₀), f=data.f, g=data.g, H=data.H,
        maxiter=K, tol=1e-4, freq=20,
        bool_trace=true,
        subpstrategy=:direct,
        initializerule=:given,
        adaptiverule=:constant,
        ratio_σ=1.0,
        ratio_Δ=1.0,
        Mₕ=Mₕ
    )
    push!(results, ("UTR", rd))
    # results[1] = ("UTR", rd)
    check_nesterovlb(data, rd.state.x)

    rd = ATR(name=Symbol("ATR"))(;
        x0=copy(data.x₀), f=data.f, g=data.g, H=data.H,
        maxiter=K, tol=1e-4, freq=20,
        bool_trace=true,
        subpstrategy=:nesterov,
        initializerule=:given,
        Mₕ=Mₕ
    )
    push!(results, ("ATR", rd))
    check_nesterovlb(data, rd.state.x)

    rd = CubicRegularizationVanilla(name=Symbol("Cubic"))(;
        x0=copy(data.x₀), f=data.f, g=data.g, H=data.H,
        maxiter=K, tol=1e-4, freq=20,
        bool_trace=true,
        subpstrategy=:direct,
        initializerule=:given,
        Mₕ=Mₕ
    )
    push!(results, ("CubicReg", rd))
    check_nesterovlb(data, rd.state.x)

    rd = CubicRegularizationVanilla(name=Symbol("CubicReg-A"))(;
        x0=copy(data.x₀), f=data.f, g=data.g, H=data.H,
        maxiter=K, tol=1e-4, freq=20,
        bool_trace=true,
        subpstrategy=:nesterov,
        initializerule=:given,
        Mₕ=Mₕ
    )
    push!(results, ("CubicReg-A", rd))
    check_nesterovlb(data, rd.state.x)

    # stats, _ = ARCqKOp(
    #     nlp,
    #     max_time=500.0,
    #     max_iter=500,
    #     max_eval=typemax(Int64),
    #     verbose=true
    #     # atol=atol,
    #     # rtol=rtol,
    #     # @note: how to set |g|?
    # )
    # rarc = arc_to_result(nlp, stats, "ARC")
end


if bool_plot

    linestyles = [:dash]
    xaxis = :k
    metric = :ϵ
    @printf("plotting results\n")

    pgfplotsx()
    title = L"Nesterov's Lower Bound Function: $n:=$%$(n), $k:=$%$(k), $p:=$%$(p)"
    @info "title: $title"
    title = ""
    fig = plot(
        xlabel=L"\textrm{Iterations}",
        ylabel=L"\|\nabla f\|",
        title=title,
        size=(600, 500),
        yticks=[1e-12, 1e-10, 1e-8, 1e-6, 1e-4, 1e-2, 1e-1, 1e0, 1e1],
        # xticks=[1, 10, 100, 200, 500, 1000, 10000, 100000, 1e6],
        yscale=:log10,
        dpi=500,
        xtickfont=font(20),
        ytickfont=font(20),
        xlabelfontsize=20,
        ylabelfontsize=17,
        legend=:top,
        legendcolumns=2,
        legendfontsize=24,
        legend_background_color=RGBA(1.0, 1.0, 1.0, 0.7),
        legendfonthalign=:left,
        titlefontsize=22,
        palette=:Paired_8 #palette(:PRGn)[[1,3,9,10,11]]
    )
    for (nm, rv) in results
        yv = getresultfield(rv, metric)
        @info "plotting $metric"
        plot!(fig,
            xaxis == :t ? getresultfield(rv, :t) : (1:(yv|>length)),
            yv,
            label=L"\texttt{%$nm}",
            linewidth=2.0,
            # linestyle=:dash,
            # markershape=:circle,
            # markersize=1.0,
            # markercolor=:match,
        )
    end
    savefig(fig, "/tmp/$metric-nesterov-$xaxis.png")
    savefig(fig, "/tmp/$metric-nesterov-$xaxis.tex")
    savefig(fig, "/tmp/$metric-nesterov-$xaxis.pdf")
end