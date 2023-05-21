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
bool_q_preprocessed = true
f1(A, d=2) = sqrt.(sum(abs2.(A), dims=d))

# name = "a4a"
# name = "a9a"
# name = "w4a"
# name = "covtype"
name = "news20"
# name = "rcv1"
# Load data
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
# loss
λ = 1e-5
n = Xv[1, :] |> length
Random.seed!(1)
N = y |> length


function g1(w)
    function _g(x, y, w)
        ff = exp(-y * w' * x)
        return -ff / (1 + ff) * y * x
    end
    _pure = vmapreduce(
        (x, y) -> _g(x, y, w),
        +,
        X,
        y
    )
    return _pure / N + λ * w
end
function hvp1(w, v, Hv; eps=1e-8)
    function _hvp(x, y, q, w, v)
        wx = w' * x
        ff = exp(-y * wx)
        return ff / (1 + ff)^2 * q * x' * v
    end
    _pure = vmapreduce(
        (x, y, q) -> _hvp(x, y, q, w, v),
        +,
        X,
        y,
        P
    )
    # copyto!(Hv, 1 / eps .* g(w + eps .* v) - 1 / eps .* g(w))
    copyto!(Hv, _pure ./ N .+ λ .* v)
end

function loss1(w)
    _pure = vmapreduce(
        (x, c) -> log(1 + exp(-c * w' * x)),
        +,
        X,
        y
    )
    return _pure / N + 0.5 * λ * w'w
end
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
x0 = 10 * randn(Float64, n)
ε = 1e-8 # * max(g(x0) |> norm, 1)
options = Optim.Options(
    g_tol=ε,
    iterations=10000,
    store_trace=true,
    show_trace=true,
    show_every=1,
    time_limit=500
)


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

    # r_lbfgs = Optim.optimize(
    #     loss, g, x0,
    #     LBFGS(; m=5, alphaguess=LineSearches.InitialStatic(),
    #         linesearch=LineSearches.BackTracking()), options;
    #     inplace=false
    # )

    r = HSODM(name=Symbol("adaptive-HSODM"))(;
        x0=copy(x0), f=loss, g=g, hvp=hvpdiff,
        maxiter=10000, tol=ε, freq=1,
        maxtime=10000,
        direction=:warm, linesearch=:hagerzhang,
        adaptive=:none
    )

    rn = PFH(name=Symbol("inexact-Newton"))(;
        x0=copy(x0), f=loss, g=g, hvp=hvpdiff,
        maxiter=10000, tol=ε, freq=1,
        step=:newton, μ₀=0.0,
        maxtime=10000, linesearch=:backtrack,
        bool_trace=true,
        direction=:warm
    )
    # rh = PFH()(;
    #     x0=copy(x0), f=loss, g=g, hvp=hvpdiff,
    #     maxiter=10000, tol=ε, freq=1,
    #     step=:hsodm, μ₀=5e-1,
    #     maxtime=10000,
    #     direction=:warm
    # )
    rh = PFH()(;
        x0=copy(x0), f=loss, g=g, hvp=hvpdiff,
        maxiter=10000, tol=ε, freq=1,
        step=:hsodm, μ₀=5e-2,
        bool_trace=true,
        maxtime=10000,
        direction=:warm
    )

end


if bool_plot
    results = [
        # optim_to_result(res1, "GD+Wolfe"),
        # optim_to_result(r_lbfgs, "LBFGS+Wolfe"),
        rn,
        r,
        rh
    ]
    linestyles = [:dash, :dot, :dashdot, :dashdotdot]
    method_names = getname.(results)
    for xaxis in (:t, :k)
        for metric in (:ϵ, :fx)
            @printf("plotting results\n")

            pgfplotsx()
            title = L"Logistic Regression $\texttt{name}:=$%$(name), $n:=$%$(n), $N:=$%$(N)"
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
                labelfontsize=14,
                xtickfont=font(13),
                ytickfont=font(13),
                # leg=:bottomleft,
                legendfontsize=14,
                legendfontfamily="sans-serif",
                titlefontsize=22,)
            for (k, rv) in enumerate(results)
                yv = getresultfield(rv, metric)
                plot!(fig,
                    xaxis == :t ? getresultfield(rv, :t) : (1:(yv|>length)),
                    yv,
                    label=method_names[k],
                    linewidth=1.5,
                    markershape=:circle,
                    # markeralpha=0.8,
                    # markerstrokecolor=:match,
                    # linestyle=linestyles[k]
                )
            end
            savefig(fig, "/tmp/$metric-logistic-$name-$xaxis.pdf")
        end
    end
end
