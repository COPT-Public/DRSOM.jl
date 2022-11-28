
__precompile__()



include("../helper.jl")
include("../helper_plus.jl")
include("../helper_l.jl")
include("../helper_c.jl")
include("../helper_f.jl")
include("../helper_h.jl")
include("../lp.jl")

using .LP
using .drsom_helper
using .drsom_helper_c
using .drsom_helper_f
using .drsom_helper_l
using .drsom_helper_plus
using .hsodm_helper
using AdaptiveRegularization
using ArgParse
using CUTEst
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

log_freq = 200
tol_grad = 1e-5
max_iter = 20000
max_time = 200.0
options = Optim.Options(
    g_tol=tol_grad,
    iterations=max_iter,
    store_trace=true,
    show_trace=true,
    show_every=log_freq,
    time_limit=max_time
)
options_drsom = Dict(
    :maxiter => max_iter,
    :maxtime => max_time,
    :tol => tol_grad,
    :freq => log_freq
)

wrapper_gd(x, loss, g, H) =
    optim_to_result(
        Optim.optimize(
            loss, g, x,
            GradientDescent(;
                alphaguess=LineSearches.InitialStatic(),
                linesearch=LineSearches.HagerZhang()
            ),
            options;
            inplace=false
        ), "GD+Wolfe"
    )
wrapper_cg(x, loss, g, H) =
    optim_to_result(
        Optim.optimize(
            loss, g, x,
            ConjugateGradient(;
                alphaguess=LineSearches.InitialStatic(),
                linesearch=LineSearches.HagerZhang()
            ),
            options;
            inplace=false
        ), "CG"
    )
wrapper_lbfgs(x, loss, g, H) =
    optim_to_result(
        Optim.optimize(
            loss, g, x,
            LBFGS(;
                linesearch=LineSearches.HagerZhang()
            ),
            options;
            inplace=false
        ), "LBFGS+Wolfe"
    )
wrapper_newton(x, loss, g, H) =
    optim_to_result(
        Optim.optimize(
            loss, g, H, x,
            NewtonTrustRegion(),
            options;
            inplace=false
        ), "Newton+TR"
    )
wrapper_drsom(x, loss, g, H) =
    drsom_helper.run_drsomd(
        copy(x), loss, g, H;
        options_drsom...
    )
wrapper_drsom_homo(x, loss, g, H) =
    drsom_helper_plus.run_drsomd(
        copy(x), loss, g, H;
        direction=:homokrylov,
        options_drsom...
    )
wrapper_hsodm(x, loss, g, H) =
    hsodm_helper.run_drsomd(
        copy(x), loss, g, H;
        direction=:warm,
        options_drsom...
    )
function wrapper_arc(nlp)
    stats = ARCqKOp(
        nlp,
        max_time=max_time,
        max_iter=max_iter,
        max_eval=typemax(Int64),
        verbose=true
        # atol=atol,
        # rtol=rtol,
        # @note: how to set |g|?
    )
    # AdaptiveRegularization.jl to my style of results
    return arc_to_result(nlp, stats, "ARC")
end

# My solvers and those in Optim.jl
OPTIMIZERS = Dict(
    :GD => wrapper_gd,
    :LBFGS => wrapper_lbfgs,
    :NewtonTR => wrapper_newton,
    :DRSOM => wrapper_drsom,
    :DRSOMHomo => wrapper_drsom_homo,
    :HSODM => wrapper_hsodm,
    :CG => wrapper_cg
)
# solvers in AdaptiveRegularization.jl 
OPTIMIZERS_NLP = Dict(
    :ARC => wrapper_arc
)
