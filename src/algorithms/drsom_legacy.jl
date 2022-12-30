using Base.Iterators
using LinearAlgebra
using Printf
using Dates

"""
    DRSOMFreeIteration(; <keyword-arguments>)
"""
Base.@kwdef mutable struct DRSOMFreeIteration{Tx,Tf,Tr,Tg,TH,Tψ,Tc,Tt}
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
end


Base.IteratorSize(::Type{<:DRSOMFreeIteration}) = Base.IsInfinite()

Base.@kwdef mutable struct DRSOMFreeState{R,Tx,Tq,Tc}
    x::Tx             # iterate
    fx::R             # new value f at x: x(k)
    fz::R             # old value f at z: x(k-1)
    ∇f::Tx            # gradient of f at x
    ∇fz::Tx           # gradient of f at z
    ∇fb::Tx           # gradient buffer (for temporary use)
    y::Tx             # forward point
    z::Tx             # previous point
    d::Tx             # momentum/fixed-point diff at iterate (= x - z)
    a1::R             # stepsize 1 parameter of gradient
    a2::R             # stepsize 2 parameter of momentum
    Q::Tq             # Q for trs
    c::Tc             # c for trs
    Δ::R              # trs radius
    dq::R             # decrease of estimated quadratic model
    df::R             # decrease of the real function value
    ρ::R              # trs descrease ratio: ρ = df/dq
    ϵ1::R             # eps 1: residual for fix-point 
    ϵ2::R             # eps 2: residual for gradient 
    γ::R = 1e-16      # scaling parameter γ for λ
    λ::R = 1e-16      # dual λ
    it::Int = 1       # inner iteration #. for trs adjustment
    t::R = 0.0        # running time
end

function TrustRegionSubproblemLegacy(Q, c, state::DRSOMFreeState; G=diagmQ(ones(2)))
    # for d it is too small, reduce to a Cauchy point ?
    eigvalues = eigvals(Q)
    sort!(eigvalues)
    lmin, lmax = eigvalues
    lb = max(0, -lmin)
    lmax = max(lb, lmax) + 1e4
    state.λ = state.γ * lmax + max(1 - state.γ, 0) * lb
    try
        alpha = -(Q + state.λ .* G) \ c
        return alpha
    catch
        print(Q)
        print(state.λ)
    end
end


function Base.iterate(iter::DRSOMFreeIteration)
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
    a2 = 0.0
    ls = 1.0
    it = 1
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
            state = DRSOMFreeState(
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
                a1=a1,
                a2=a2,
                d=d,
                Δ=0.0,
                dq=dq,
                df=df,
                ρ=ro,
                ϵ1=norm(d, 2),
                ϵ2=norm(grad_f_x, 2),
                γ=1e-6,
                λ=1e-6,
                it=it,
                t=t,
            )
            return state, state
        else
            ls *= 0.5
            it += 1
        end
    end
end


"""
Solve an iteration using TRS to produce stepsizes,
alpha: extrapolation
gamma: gradient step
"""
function Base.iterate(iter::DRSOMFreeIteration, state::DRSOMFreeState{R,Tx}) where {R,Tx}

    n = length(state.x)
    state.z = z = state.x
    state.fz = fz = state.fx
    state.∇fz = state.∇f
    # construct trs
    if iter.mode ∈ (:forward, false)
        Hg, Hd = iter.rh(iter.f, state; cfg=iter.cfg)
    elseif iter.mode ∈ (:backward, true)
        Hg, Hd = iter.rh(state; tp=iter.tp)
    else
        state.∇f = iter.g(state.x)
        H = iter.H(state.x)
        Hg = H * state.∇f
        Hd = H * state.d
    end

    Q11 = state.∇f' * Hg
    Q12 = -state.∇f' * Hd
    Q22 = state.d' * Hd
    c1 = -state.∇f'state.∇f
    c2 = state.∇f'state.d
    state.Q = Q = [Q11 Q12; Q12 Q22]
    state.c = c = [c1; c2]
    gg = state.∇f' * state.∇f
    gd = state.∇f' * state.d
    dd = state.d' * state.d
    G = [gg -gd; -gd dd]
    # G = diagm(ones(2))
    it = 1
    # if Q22 > 1e-4
    while true
        a1, a2 = TrustRegionSubproblemLegacy(Q, c, state; G=G)
        x = y = state.z - a1 .* state.∇f + a2 .* state.d
        fx = iter.f(x)
        alp = [a1; a2]
        dq = -alp' * Q * [a1; a2] / 2 - alp' * c
        df = fz - fx
        ro = df / dq
        if (df < 0) || (ro <= 0.1)
            # state.λ *= 10
            state.γ *= 5
        elseif ro >= 0.5
            # state.λ = max(min(sqrt(state.λ), state.λ / 100), 1e-10)
            state.γ = max(min(sqrt(state.γ), log(10, state.γ + 1)), 1e-16)
        end
        if (ro > 0.05 && df > 0) || it == iter.itermax
            state.a1 = a1
            state.a2 = a2
            state.x = x
            state.y = y
            state.fx = fx
            state.ρ = ro
            state.dq = dq
            state.df = df
            state.d = x - z
            state.Δ = sqrt(a1^2 + a2^2)
            state.it = it
            state.ϵ1 = norm(x - z)
            state.ϵ2 = norm(state.∇f)
            state.it = it
            state.t = (Dates.now() - iter.t).value / 1e3
            return state, state
        end
        it += 1
    end
end

hsodm_stopping_criterion(tol, state::DRSOMFreeState) =
    (state.ϵ2 <= tol) || (state.ϵ1 <= tol) && abs(state.fz - state.fx) <= tol


function hsodm_display(it, state::DRSOMFreeState)
    if mod(it, 30) == 1
        @printf("%5s | %10s | %8s | %8s | %7s | %7s | %7s | %7s | %7s | %5s | %2s | %6s |\n",
            "k", "f", "α1", "α2", "Δ", "|d|", "|∇f|", "γ", "λ", "ρ", "it", "t",
        )
    end
    @printf("%5d | %+.3e | %+.1e | %+.1e | %.1e | %.1e | %.1e | %.1e | %.1e | %+.2f | %2d | %6.1f |\n",
        it, state.fx, state.a1, state.a2, state.Δ, state.ϵ1, state.ϵ2, state.γ, state.λ, state.ρ, state.it, state.t
    )
end

default_solution(::DRSOMFreeIteration, state::DRSOMFreeState) = state.x


# Aliases
