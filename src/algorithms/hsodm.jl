
using Base.Iterators
using LinearAlgebra
using Printf
using Dates
using KrylovKit
using Distributions
using LineSearches

"""
    HSODMIteration(; <keyword-arguments>)
"""
Base.@kwdef mutable struct HSODMIteration{Tx,Tf,Tr,Tg,TH,Tψ,Tc,Tt}
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
    direction = :cold
    eigtol::Float64 = 1e-8
end


Base.IteratorSize(::Type{<:HSODMIteration}) = Base.IsInfinite()

Base.@kwdef mutable struct HSODMState{R,Tx}
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
    Δ::R              # trs radius
    dq::R             # decrease of estimated quadratic model
    df::R             # decrease of the real function value
    ρ::R              # trs descrease ratio: ρ = df/dq
    ϵ::R              # eps 2: residual for gradient 
    γ::R = 1          # stepsize parameter γ
    kλ::Int = 1       # krylov iterations
    it::Int = 1       # inner iteration #. for trs adjustment
    t::R = 0.0        # running time
    λ₁::Float64 = 0.0 # smallest curvature if available
    ξ::Tx             #
end



function Base.iterate(iter::HSODMIteration)
    iter.t = Dates.now()
    z = copy(iter.x0)
    fz = iter.f(z)
    grad_f_x = similar(z)
    grad_f_b = similar(z)

    grad_f_x = iter.g(z)
    H = iter.H(z)
    n = length(z)
    # construct homogeneous system
    B = [H grad_f_x; grad_f_x' -1e-3]
    vals, vecs, info = KrylovKit.eigsolve(B, n + 1, 1, :SR, Float64; tol=iter.eigtol)
    kλ = info.numops
    λ₁ = vals[1]
    ξ = vecs[1]
    v = reshape(ξ[1:end-1], n)
    t₀ = ξ[end]
    if abs(t₀) > 1e-3
        v = v / t₀
    end
    vn = norm(v)

    vHv = (v'*H*v/2)[]
    vg = (v'*grad_f_x)[]
    # now use a TRS to solve (gamma, state.γha)
    γ = 1.0
    it = 1
    while true
        Δ = vn * γ
        y = z - γ .* grad_f_x
        fx = iter.f(y)
        dq = -γ^2 * vHv / 2 - γ * vg
        df = fz - fx
        ro = df / dq
        if ro > 0.7 || it == iter.itermax
            t = (Dates.now() - iter.t).value / 1e3
            d = y - z
            state = HSODMState(
                x=y,
                y=y,
                z=z,
                fx=fx,
                fz=fz,
                ∇f=grad_f_x,
                ∇fz=z,
                ∇fb=grad_f_b,
                α=[γ],
                d=d,
                Δ=Δ,
                dq=dq,
                df=df,
                ρ=ro,
                ϵ=norm(grad_f_x, 2),
                γ=1e-6,
                kλ=kλ,
                it=it,
                t=t,
                ξ=ones(length(z) + 1),
                λ₁=λ₁
            )
            return state, state
        else
            γ *= 0.6
            it += 1
        end
    end
end


"""
Solve an iteration using TRS to produce stepsizes,
state.γha: extrapolation
gamma: gradient step
"""
function Base.iterate(iter::HSODMIteration, state::HSODMState{R,Tx}) where {R,Tx}

    n = length(state.x)
    state.z = z = state.x
    state.fz = fz = state.fx
    state.∇fz = state.∇f
    # construct trs
    # compute Hg, Hd first
    # if iter.mode ∈ (:forward, false)
    #     Hg, Hd = iter.rh(iter.f, state; cfg=iter.cfg)
    # elseif iter.mode ∈ (:backward, true)
    #     # todo, not ready yet.
    #     # compute gradient first
    #     Hg, Hd = iter.rh(state; tp=iter.tp)
    # else
    state.∇f = iter.g(state.x)
    H = iter.H(state.x)

    # construct homogeneous system
    B = [H state.∇f; state.∇f' 0]
    if iter.direction == :cold
        vals, vecs, info = KrylovKit.eigsolve(B, n + 1, 1, :SR, Float64; tol=iter.eigtol, eager=true)
    else
        vals, vecs, info = KrylovKit.eigsolve(B, state.ξ, 1, :SR; tol=iter.eigtol, eager=true)
    end
    kλ = info.numops
    state.λ₁ = vals[1]
    ξ = vecs[1]
    v = reshape(ξ[1:end-1], n)
    t₀ = ξ[end]
    if abs(t₀) > 1e-3
        v = v / t₀
    end
    vn = norm(v)


    vHv = (v'*H*v/2)[]
    vg = (v'*state.∇f)[]

    it = 1
    state.γ = state.Δ / vn * 2
    while true
        state.Δ = state.γ * vn
        x = y = state.z + v * state.γ
        fx = iter.f(x)
        dq = -state.γ^2 * vHv / 2 - state.γ * vg
        df = fz - fx
        ro = df / dq
        if (df < 0) || (ro <= 0.7)
            state.γ *= 0.8
        end

        if (ro > 0.7 && df > 0) || it == iter.itermax
            state.α = [state.γ]
            state.x = x
            state.y = y
            state.fx = fx
            state.ρ = ro
            state.dq = dq
            state.df = df
            state.d = x - z
            state.kλ = kλ
            state.it = it
            state.ϵ = norm(state.∇f)
            state.it = it
            state.ξ = ξ
            state.t = (Dates.now() - iter.t).value / 1e3
            return state, state
        end
        it += 1
    end
end

drsom_stopping_criterion(tol, state::HSODMState) =
    (state.Δ <= tol / 1e4) || (state.ϵ <= tol) && abs(state.fz - state.fx) <= tol


function drsom_display(it, state::HSODMState)
    sprintarray(arr) = join(map(x -> @sprintf("%+.1e", x), arr), ",")
    if it == 1
        log = @sprintf("%5s | %10s | %8s | %8s | %8s | %5s | %5s | %6s | %2s | %6s |\n",
            "k", "f", "α ($(state.α |> length))", "Δ", "|∇f|", "λ", "kλ", "ρ", "kₜ", "t",
        )
        format_header(log)
        @printf("%s", log)
    end
    if mod(it, 30) == 0
        @printf("%5s | %10s | %8s | %8s | %8s | %5s | %5s | %6s | %2s | %6s |\n",
            "k", "f", "α ($(state.α |> length))", "Δ", "|∇f|", "λ", "kλ", "ρ", "kₜ", "t",
        )

    end
    @printf("%5d | %+.3e | %8s | %.2e | %.1e | %+.0e | %.0e | %+.0e | %2d | %6.1f |\n",
        it, state.fx, sprintarray(state.α[1:min(2, end)]), state.Δ, state.ϵ, state.λ₁, state.kλ, state.ρ, state.it, state.t
    )
end

default_solution(::HSODMIteration, state::HSODMState) = state.x
