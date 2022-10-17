# DRSOM-F:
# @author: Chuwen Z.
#    DRSOM with using new Trust-region update of Curtis
#     referred to as the DRSOM with Curtis-style (DRSOM-F).
#     and iterative method to guarantee Assumption 2
# @reference:
# [1].  Curtis, F.E., Robinson, D.P., Samadi, M.: A trust region algorithm with a worst-case iteration complexity of $$\mathcal{O}(\epsilon ^{-3/2})$$for nonconvex optimization. 
#   Math. Program. 162, 1–32 (2017). https://doi.org/10.1007/s10107-016-1026-2
# 
using Base.Iterators
using LinearAlgebra
using Printf
using Dates
using KrylovKit
using Distributions

"""
    DRSOMFIteration(; <keyword-arguments>)
"""
Base.@kwdef mutable struct DRSOMFIteration{Tx,Tf,Tr,Tg,TH,Tψ,Tc,Tt}
    f::Tf             # f: smooth f
    ψ::Tψ = nothing   # ψ: nonsmooth part (not implemented yet)
    rh::Tr = nothing  # hessian-vector product function to produce [g, Hg, Hd]
    g::Tg = nothing   # gradient function
    H::TH = nothing   # hessian function
    x0::Tx            # initial point
    cfg::Tc = nothing # gradient config
    tp::Tt = nothing  # gradient tape
    t::Dates.DateTime = Dates.now()
    itermax::Int64 = 20
    mode = :forward
    direction = :krylov
    direction_num::Int64 = 10 # extra direction numbers beyond initial
    direction_style = :truncate
    η::Float64 = 1e-4
    σmin = 1e-2
    σmax = 1e2
    γc = 0.4
    γe = 1.1
    γλ = 2
end


Base.IteratorSize(::Type{<:DRSOMFIteration}) = Base.IsInfinite()

"""
This is a modified parameterization of Curtis, 2017, Mathematical Programming
We let:
    - σ the fraction of |λ|/|d|
"""
Base.@kwdef mutable struct DRSOMFState{R,Tx,Tq,Tc}
    x::Tx             # iterate
    fx::R             # new value f at x: x(k)
    fz::R             # old value f at z: x(k-1)
    ∇f::Tx            # gradient of f at x
    ∇fz::Tx           # gradient of f at z
    ∇fb::Tx           # gradient buffer (for temporary use)
    y::Tx             # forward point
    z::Tx             # previous point
    d::Tx             # momentum/fixed-point diff at iterate (= x - z)
    α::Tx             # stepsizes for directions...
    Q::Tq             # Q for trs
    G::Tq             # G for trs
    c::Tc             # c for trs
    Δ::R              # trs radius
    dq::R             # decrease of estimated quadratic model
    df::R             # decrease of the real function value
    ρ::R              # trs descrease ratio: ρ = df/dq
    ρd::R             # trs ratio of order: ρ = df/|d|^3
    ϵ::R              # eps 2: residual for gradient 
    σ::R = 1.0        # scaling parameter σ
    λ::R = 1e-16      # dual λ
    k::Int = 0        # outer iteration #. 
    kₜ::Int = 0        # inner iteration #. for trs adjustment
    t::R = 0.0        # running time
end



function Base.iterate(iter::DRSOMFIteration)
    iter.t = Dates.now()
    z = copy(iter.x0)
    fz = iter.f(z)
    grad_f_x = similar(z)
    grad_f_b = similar(z)
    if iter.mode ∈ (:forward, false)
        Hg, _ = iter.rh(iter.f, grad_f_x, z - z, z; cfg=iter.cfg)
    elseif iter.mode ∈ (:backward, true)
        Hg, _ = iter.rh(grad_f_x, grad_f_b, z - z, z; tp=iter.tp)
    else
        grad_f_x = iter.g(z)
        H = iter.H(z)
        Hg = H * grad_f_x
    end
    Q11 = grad_f_x' * Hg
    ######################################
    c1 = -grad_f_x' * grad_f_x
    Q = [Q11 0; 0 0]
    c = [c1; 0]
    # now use a TRS to solve (gamma, alpha)
    ls = 1.0
    kₜ = 1
    while true
        if Q11 > 0
            a1 = -c1 / Q11 * ls
            dq = -a1 * Q11 * a1 / 2 - a1 * c1
        else
            a1 = 0.1 * ls
            dq = -a1 * c1
        end
        y = z - a1 .* grad_f_x
        # ######################################
        # todo, eval proximal here?
        #   - we should design special alg. for proximal case.
        # evaluate TRS
        fx = iter.f(y)
        df = fz - fx
        ro = df / dq
        if ro > 0.1 || it == iter.itermax
            t = (Dates.now() - iter.t).value / 1e3
            d = y - z
            state = DRSOMFState(
                x=y,
                y=y,
                z=z,
                fx=fx,
                fz=fz,
                Q=Q,
                c=c,
                ∇f=grad_f_x,
                ∇fz=z,
                ∇fb=grad_f_b,
                α=[a1, 0],
                d=d,
                Δ=norm(d, 2),
                dq=dq,
                df=df,
                G=[a1^2*norm(grad_f_x)^2 0; 0 0],
                ρ=ro,
                ρd=ro,
                ϵ=norm(grad_f_x, 2),
                λ=1e-6,
                σ=1.0,
                kₜ=kₜ,
                t=t,
            )
            return state, state
        else
            ls *= 0.5
            kₜ += 1
        end
    end
end

"""
Solve an iteration using TRS to produce stepsizes,
"""
function Base.iterate(iter::DRSOMFIteration, state::DRSOMFState{R,Tx}) where {R,Tx}

    n = length(state.x)
    state.z = z = state.x
    state.fz = fz = state.fx
    state.∇fz = state.∇f
    # construct trs
    # compute Hg, Hd first
    if iter.mode ∈ (:forward, false)
        Hg, Hd = iter.rh(iter.f, state; cfg=iter.cfg)
    elseif iter.mode ∈ (:backward, true)
        # todo, not ready yet.
        # compute gradient first
        Hg, Hd = iter.rh(state; tp=iter.tp)
    else
        state.∇f = iter.g(state.x)
        H = iter.H(state.x)
        Hg = H * state.∇f
        Hd = H * state.d
    end
    gnorm = norm(state.∇f)
    dnorm = norm(state.d)


    HD = [-Hg / gnorm Hd / dnorm]
    D = [-state.∇f / gnorm state.d / dnorm]

    state.Q = Q = D' * HD
    state.c = c = D' * state.∇f
    state.G = G = D' * D
    kₜ = 1
    dim = 0
    bool_reduce = false
    while true
        alp = TrustRegionSubproblem(Q, c, state; G=G, mode=:tr, Δ=state.Δ)
        x = y = state.z + D * alp
        fx = iter.f(x)
        df = fz - fx
        dq = -alp' * Q * alp / 2 - alp' * c
        s = sqrt(alp' * G * alp)
        ρ = df / dq
        ρd = df / (s)^3
        σ = state.λ / s
        acc_tr = false
        acc_direction = false
        if bool_reduce
            state.σ = max(state.σ, state.λ / s)
            bool_reduce = false
        end
        if (df < 0) || (ρd < iter.η)
            # reduce "radius"
            # in Curtis, it is called the Constract step
            contract(iter, state, s)
            # mark that we have reduce this
            # and should enlarge σ in the next iterate
            bool_reduce = true
        elseif σ > state.σ
            # function reduction is good, 
            # whereas λ is too big, 
            # this means we can actually go further
            # simply enlarge and go again.   
            state.Δ = state.λ / state.σ
        else
            # everything works well
            state.Δ = max(state.Δ, s * iter.γe)
            state.σ = max(state.σ, σ)
            acc_tr = true
            # accept and go to acc stage
        end
        # direction quality check
        acc_direction = (dim >= iter.direction_num) || (mod(state.k, 3) != 0)
        if ~acc_direction && kₜ < iter.itermax
            # todo, how to check if it is good enough?
            # now with of maximum inner max
            v = HD * alp / norm(alp)
            # update subspace
            if iter.direction_style == :truncate
                D = [D * alp v]
                HD = [HD * alp H * v]
            elseif iter.direction_style == :full
                D = [D v]
                HD = [HD H * v]
            else

            end
            state.Q = Q = D' * HD
            state.c = c = D' * state.∇f
            state.G = G = D' * D
            dim += 1
        end
        if (acc_tr && acc_direction) || kₜ == iter.itermax
            state.α = alp
            state.x = x
            state.y = y
            state.fx = fx
            state.ρ = ρ
            state.ρd = ρd
            state.df = df
            state.d = x - z
            state.ϵ = norm(state.∇f)
            state.t = (Dates.now() - iter.t).value / 1e3
            state.k += 1
            state.kₜ = kₜ
            return state, state

        end
        kₜ += 1
    end
end

"""

@note:
    The "CONTRACT" step in the Curtis's trust-region framework
    since the decrease is not satisfactory, 
    we should somewhat decrease Δ or increase λ
    while still keeping λ/|d| in some interval
"""
function contract(iter::DRSOMFIteration, state::DRSOMFState{R,Tx}, s::Float64; Δϵ::Float64=1e-4) where {R,Tx}
    if state.λ < iter.σmin * s
        # increase a little bit 
        #   cannot be bigger :λmax
        λmax = state.λ + sqrt(iter.σmin * norm(state.∇f))
        alp = TrustRegionSubproblem(
            state.Q, state.c, state;
            G=state.G, mode=:reg, λ=λmax
        )
        # size of the new step
        sn = sqrt(alp' * state.G * alp)
        if λmax / sn <= iter.σmax
            # good enough
            state.Δ = sn
            return
        else
            # this implies, :λ is too big this time
            # we should slightly decrease it 
            # (between this λ and the previous state.λ)
            # i.e.
            #   λ ∈ (state.λ, lmax)
            # let us have a bisection to allow inaccurate search
            λmin = state.λ
            acc = false
            sn = 0.0
            while (λmax - λmin) / λmin > Δϵ
                λ = (λmin + λmax) / 2
                alp = TrustRegionSubproblem(
                    state.Q, state.c, state;
                    G=state.G, mode=:reg, λ=λ
                )
                sn = sqrt(alp' * state.G * alp) # size
                ro = λ / sn
                if ro >= iter.σmax
                    λmax = λ
                elseif ro <= iter.σmin
                    λmin = λ
                else
                    # good enough
                    acc = true
                    break
                end
            end
            state.Δ = sn
            return
        end
    else
        # λ is big enough resp. to |d|,
        # increase by a fixed fraction 
        # and this will still guarantee fraction λ/|d|
        λ = state.λ * iter.γλ
        alp = TrustRegionSubproblem(
            state.Q, state.c, state;
            G=state.G, mode=:reg, λ=λ
        )
        sn = sqrt(alp' * state.G * alp)
        state.Δ = max(iter.γc * s, sn)
    end
end


drsom_stopping_criterion(tol, state::DRSOMFState) =
    (state.Δ <= tol / 1e2) || (state.ϵ <= tol) && abs(state.fz - state.fx) <= tol


function drsom_display(it, state::DRSOMFState)
    if it == 1
        log = @sprintf("%5s | %10s | %13s | %7s | %7s | %5s | %5s | %6s | %6s | %2s | %2s | %6s \n",
            "k", "f", "α ($(state.α |> length))", "Δ", "|∇f|", "σ", "λ", "ρ", "ρd", "kₜ", "r", "t",
        )
        format_header(log)
        @printf("%s", log)
    end
    if mod(it, 10) == 0
        @printf("%5s | %10s | %13s | %7s | %7s | %5s | %5s | %6s | %6s | %2s | %2s | %6s \n",
            "k", "f", "α ($(state.α |> length))", "Δ", "|∇f|", "σ", "λ", "ρ", "ρd", "kₜ", "r", "t",
        )

    end
    @printf("%5d | %+.3e | %13s | %.1e | %.1e | %.0e | %.0e | %+.0e | %+.0e | %2d | %2d | %6.1f \n",
        it, state.fx, sprintarray(state.α[1:min(2, end)]), state.Δ, state.ϵ, state.σ, state.λ, state.ρ, state.ρd, state.kₜ, (state.α |> length), state.t
    )
end

default_solution(::DRSOMFIteration, state::DRSOMFState) = state.x
