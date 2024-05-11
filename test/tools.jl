
__precompile__()


using AdaptiveRegularization
using ArgParse
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


# utilities
getresultfield(x, y=:fx) = getfield.(getfield(x, :trajectory), y)
getname(x) = getfield(x, :name) |> string
geteps(x) = x.g_norm

Base.@kwdef mutable struct StateOptim
    fx::Float64
    ϵ::Float64
    t::Float64
    kf::Int = 0
    kg::Int = 0
    kH::Int = 0
    kh::Int = 0
end
function optim_to_result(rr, name)
    traj = map(
        (x) -> StateOptim(fx=x.value, ϵ=x.g_norm, t=x.metadata["time"]), rr.trace
    )
    traj[end].kf = rr.f_calls
    traj[end].kg = rr.g_calls
    traj[end].kH = rr.h_calls
    return Result(name=name, iter=rr, state=traj[end], k=rr.iterations, trajectory=traj)
end

function optim_to_result(nlp, rr, name)
    traj = map(
        (x) -> StateOptim(fx=x.value, ϵ=x.g_norm, t=rr.time_run), rr.trace
    )
    traj[end].kf = rr.f_calls
    traj[end].kg = rr.g_calls
    traj[end].kH = rr.h_calls
    traj[end].kh = neval_hprod(nlp)
    return Result(name=name, iter=rr, state=traj[end], k=rr.iterations, trajectory=traj)
end

function arc_to_result(nlp, stats, name)
    state = StateOptim(
        fx=stats.objective,
        ϵ=NLPModels.grad(nlp, stats.solution) |> norm,
        t=stats.elapsed_time
    )
    state.kf = neval_obj(nlp) # return number of f call
    state.kg = neval_grad(nlp) # return number of gradient call
    state.kH = neval_hess(nlp) # return number of Hessian call
    state.kh = neval_hprod(nlp)

    return Result(name=name, iter=stats, state=state, k=stats.iter, trajectory=[])
end

function arc_stop_to_result(nlp, rr, name)
    ts = rr.listofstates[1][1].current_time
    traj = []
    for i in 1:rr.listofstates.i
        x = rr.listofstates[i]
        st = StateOptim(fx=x[1].fx, ϵ=x[1].gx |> norm, t=x[1].current_time - ts)
        push!(traj, st)
    end
    traj[end].kf = neval_obj(nlp)
    traj[end].kg = neval_grad(nlp)
    traj[end].kH = neval_hess(nlp)
    traj[end].kh = neval_hprod(nlp)
    stats = AdaptiveRegularization.stopping_to_stats(rr)
    return Result(name=name, iter=rr, state=stats, k=stats.iter, trajectory=traj)
end

export StateOptim, optim_to_result, arc_to_result, arc_stop_to_result

add_optim = true
add_adaptive_reg_jl = true
if add_optim
    wrapper_gd(x, loss, g, H, options; kwargs...) =
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

    wrapper_cg(x, loss, g, H, options; kwargs...) =
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
    wrapper_lbfgs(x, loss, g, H, options; kwargs...) =
        optim_to_result(
            Optim.optimize(
                loss, g, x,
                LBFGS(;
                    m=10,
                    alphaguess=LineSearches.InitialStatic(),
                    linesearch=LineSearches.HagerZhang()
                ),
                options;
                inplace=false
            ), "LBFGS+Wolfe"
        )
    wrapper_newton(x, loss, g, H, options; kwargs...) =
        optim_to_result(
            Optim.optimize(
                loss, g, H, x,
                NewtonTrustRegion(),
                options;
                inplace=false
            ), "Newton+TR"
        )
end
if add_adaptive_reg_jl
    function wrapper_arc(nlp)
        reset!(nlp)
        stats, _ = ARCqKOp(
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

    function wrapper_tr_st(nlp)
        reset!(nlp)
        stats, _ = ST_TROp(
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
        return arc_to_result(nlp, stats, "TRST")
    end
end



##########################################################
# MY VARIANTS
##########################################################
add_drsom = true
add_hsodm = true
add_utr = true
if add_drsom
    alg_drsom = DRSOM2()
    wrapper_drsom(x, loss, g, H, options; kwargs...) =
        alg_drsom(;
            x0=copy(x), f=loss, g=g, H=H,
            fog=:direct,
            sog=:hess,
            options...
        )
    wrapper_drsomh(x, loss, g, H, options; kwargs...) =
        alg_drsom(;
            x0=copy(x), f=loss, g=g, H=H,
            fog=:direct,
            sog=:hess,
            options...
        )
    wrapper_drsomf(x, loss, g, H, options; kwargs...) =
        alg_drsom(;
            x0=copy(x), f=loss,
            fog=:forward,
            sog=:forward,
            options...
        )
    wrapper_drsomb(x, loss, g, H, options; kwargs...) =
        alg_drsom(;
            x0=copy(x), f=loss,
            fog=:backward,
            sog=:backward,
            options...
        )
    wrapper_drsomd(x, loss, g, H, options; kwargs...) =
        alg_drsom(;
            x0=copy(x), f=loss, g=g,
            fog=:direct,
            sog=:prov,
            kwargs...,
            options...
        )
end
if add_hsodm
    alg_hsodm = HSODM()
    wrapper_hsodm(x, loss, g, H, options; kwargs...) =
        alg_hsodm(;
            x0=copy(x), f=loss, g=g, H=H,
            linesearch=:hagerzhang,
            direction=:warm,
            adaptive=:mishchenko,
            options...
        )
    alg_hsodm_hvp = HSODM()
    wrapper_hsodm_hvp(x, loss, g, H, options; kwargs...) =
        alg_hsodm_hvp(;
            x0=copy(x), f=loss, g=g,
            linesearch=:hagerzhang,
            direction=:auto,
            adaptive=:mishchenko,
            kwargs...,
            options...
        )
    alg_hsodm_arc = HSODM(; name=:HSODMA)
    wrapper_hsodm_arc(x, loss, g, H, options; kwargs...) =
        alg_hsodm_arc(;
            x0=copy(x), f=loss, g=g, H=H,
            linesearch=:none,
            direction=:warm,
            adaptive=:arc,
            options...
        )
end
if add_utr
    alg_utr = UTR(; name=:UTR)
    wrapper_utr(x, loss, g, H, options; kwargs...) =
        alg_utr(;
            x0=copy(x), f=loss, g=g, H=H,
            subpstrategy=:direct,
            options...
        )

    wrapper_iutr(x, loss, g, H, options; kwargs...) =
        alg_utr(;
            x0=copy(x), f=loss, g=g, H=H,
            subpstrategy=:lanczos,
            options...
        )

    wrapper_iutr_hvp(x, loss, g, H, options; kwargs...) =
        alg_utr(;
            x0=copy(x), f=loss, g=g,
            subpstrategy=:lanczos,
            kwargs...,
            options...
        )
end


# My solvers and those in Optim.jl
MY_OPTIMIZERS = Dict(
    # :DRSOM => wrapper_drsom,
    :DRSOM => wrapper_drsomd,
    # :DRSOMHomo => wrapper_drsom_homo,
    :HSODM => wrapper_hsodm,
    :HSODMhvp => wrapper_hsodm_hvp,
    # :HSODMArC => wrapper_hsodm_arc,
    :UTR => wrapper_utr,
    :iUTR => wrapper_iutr,
    :iUTRhvp => wrapper_iutr_hvp,
)

OPTIMIZERS_OPTIM = Dict(
    :GD => wrapper_gd,
    :LBFGS => wrapper_lbfgs,
    :NewtonTR => wrapper_newton,
    :CG => wrapper_cg
)

# solvers in AdaptiveRegularization.jl 
OPTIMIZERS_NLP = Dict(
    :ARC => wrapper_arc, # adaptive cubic regularization
    :TRST => wrapper_tr_st # trust-region with Steihaug–Toint CG
)
