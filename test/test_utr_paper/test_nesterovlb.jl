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



bool_check = false
bool_solve = true
bool_plot = true
n, k, p = 150, 50, 3
data = NesterovHardCase(n, k, p)
c = randn(n)

if bool_check
    gf = ForwardDiff.gradient(data.f, c)
    Hf = ForwardDiff.hessian(data.f, c)
    @assert norm(gf - data.g(c)) < 1e-4
    @assert norm(Hf - data.H(c)) < 1e-4
    @assert abs(data.f(data.xₛ) - data.fₛ) < 1e-4
end

if bool_solve
    results = Dict()
    x₀ = zeros(n)

    Mₕ(x) = 2.0^2 * factorial(2)
    results["UTR"] = rd = ATR(name=Symbol("ATR"))(;
        x0=copy(data.x₀), f=data.f, g=data.g, H=data.H,
        maxiter=2000, tol=1e-4, freq=20,
        bool_trace=true,
        subpstrategy=:direct,
        initializerule=:given,
        adaptiverule=:constant,
        ratio_σ=1.0,
        ratio_Δ=1.0,
        Mₕ=Mₕ
    )
    check_nesterovlb(data, rd.state.x)

    results["ATR"] = rd = ATR(name=Symbol("ATR"))(;
        x0=copy(data.x₀), f=data.f, g=data.g, H=data.H,
        maxiter=2000, tol=1e-4, freq=20,
        bool_trace=true,
        subpstrategy=:nesterov,
        initializerule=:given,
        Mₕ=Mₕ
    )
    @info "ATR" results["ATR"].state.acc_count
    check_nesterovlb(data, rd.state.x)

    results["Cubic"] = rd = CubicRegularizationVanilla(name=Symbol("Cubic"))(;
        x0=copy(data.x₀), f=data.f, g=data.g, H=data.H,
        maxiter=2000, tol=1e-4, freq=20,
        bool_trace=true,
        subpstrategy=:direct,
        initializerule=:given,
        Mₕ=Mₕ
    )
    @info "Cubic" results["Cubic"].state.acc_count
    check_nesterovlb(data, rd.state.x)

    results["Cubic (Acc)"] = rd = CubicRegularizationVanilla(name=Symbol("Cubic"))(;
        x0=copy(data.x₀), f=data.f, g=data.g, H=data.H,
        maxiter=2000, tol=1e-4, freq=20,
        bool_trace=true,
        subpstrategy=:nesterov,
        initializerule=:given,
        Mₕ=Mₕ
    )
    @info "Cubic (Acc)" results["Cubic (Acc)"].state.acc_count
    check_nesterovlb(data, rd.state.x)


end


if bool_plot

    linestyles = [:dash, :dot, :dashdot, :dashdotdot]
    xaxis = :k
    metric = :ϵ
    @printf("plotting results\n")

    pgfplotsx()
    title = L"Nesterov's Lower Bound Function: $n:=$%$(n), $k:=$%$(k), $p:=$%$(p)"
    fig = plot(
        xlabel=L"\textrm{Iterations}",
        ylabel=L"\|\nabla f\|",
        title=title,
        size=(600, 500),
        yticks=[1e-12, 1e-10, 1e-8, 1e-6, 1e-4, 1e-2, 1e-1, 1e0, 1e1],
        xticks=[1, 10, 100, 200, 500, 1000, 10000, 100000, 1e6],
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
            yv,
            label=nm,
            linewidth=2.5,
            # markershape=:circle,
            # markersize=2.0,
            # markercolor=:match,
        )
    end
    savefig(fig, "/tmp/$metric-nesterov-$xaxis.png")
    savefig(fig, "/tmp/$metric-nesterov-$xaxis.tex")
    savefig(fig, "/tmp/$metric-nesterov-$xaxis.pdf")
end