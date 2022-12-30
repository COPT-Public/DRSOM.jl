
using Base.Iterators
using LinearAlgebra
using Printf
using Dates
using KrylovKit
using Distributions
using LineSearches

LOG_SLOTS = @sprintf(
    "%5s | %10s | %8s | %8s | %8s | %5s | %5s | %2s | %6s |\n",
    "k", "f", "α", "Δ", "|∇f|", "λ", "kλ", "kₜ", "t"
)
Base.@kwdef mutable struct HSODMIteration{Tx,Tf,Tϕ,Tg,TH}
    f::Tf             # f: smooth f
    ϕ::Tϕ = nothing   # ϕ: nonsmooth part (not implemented yet)
    g::Tg = nothing   # gradient function
    H::TH = nothing   # hessian function
    x0::Tx            # initial point
    t::Dates.DateTime = Dates.now()
    eigtol::Float64 = 1e-10
    itermax::Int64 = 20
    direction = :cold
    linesearch = :hagerzhang
    LOG_SLOTS::String = LOG_SLOTS
    ALIAS::String = "HSODM"
    DESC::String = "Homogeneous Second-order Descent Method"
end


Base.IteratorSize(::Type{<:HSODMIteration}) = Base.IsInfinite()

Base.@kwdef mutable struct HSODMState{R,Tx}
    x::Tx             # iterate
    fx::R             # new value f at x: x(k)
    fz::R             # old value f at z: x(k-1)
    ∇f::Tx            # gradient of f at x
    ∇fz::Tx           # gradient of f at z
    y::Tx             # forward point
    z::Tx             # previous point
    d::Tx             # momentum/fixed-point diff at iterate (= x - z)
    Δ::R              # trs radius
    dq::R             # decrease of estimated quadratic model
    df::R             # decrease of the real function value
    ρ::R              # trs descrease ratio: ρ = df/dq
    ϵ::R              # eps 2: residual for gradient 
    α::R = 1e-5       # step size
    γ::R = 1e-5       # trust-region style parameter γ
    kλ::Int = 1       # krylov iterations
    kₜ::Int = 1        # inner iterations 
    t::R = 0.0        # running time
    λ₁::Float64 = 0.0 # smallest curvature if available
    ξ::Tx             # eigenvector
    kf::Int = 0       # function evaluations
    kg::Int = 0       # gradient evaluations
    kH::Int = 0       # hessian  evaluations
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
    (abs(t₀) > 1e-3) && (v = v / t₀)

    vn = norm(v)
    vHv = (v'*H*v/2)[]
    vg = (v'*grad_f_x)[]
    bool_reverse_v = vg > 0
    # reverse this v if g'v > 0
    v = (-1)^bool_reverse_v * v
    vg = (-1)^bool_reverse_v * vg
    # now use a LS to solve (state.α)
    if iter.linesearch == :trustregion
        α, fx, kₜ = TRStyleLineSearch(iter, z, v, vHv, vg, 1.0)
    elseif iter.linesearch == :hagerzhang
        # use Hager-Zhang line-search algorithm
        α, fx, kₜ = HagerZhangLineSearch(iter, grad_f_x, fz, z, v)
    else
    end
    y = z + α .* v
    fx = iter.f(y)
    dq = -α^2 * vHv / 2 - α * vg
    df = fz - fx
    ro = df / dq
    Δ = vn * α
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
        α=α,
        d=d,
        Δ=Δ,
        dq=dq,
        df=df,
        ρ=ro,
        ϵ=norm(grad_f_x, 2),
        γ=1e-6,
        kλ=kλ,
        kₜ=kₜ,
        t=t,
        ξ=ones(length(z) + 1),
        λ₁=λ₁
    )
    return state, state
end



function Base.iterate(iter::HSODMIteration, state::HSODMState{R,Tx}) where {R,Tx}

    n = length(state.x)
    state.z = z = state.x
    state.fz = fz = state.fx
    state.∇fz = state.∇f

    state.∇f = iter.g(state.x)
    H = iter.H(state.x)
    gnorm = norm(state.∇f)
    # construct homogeneous system
    B = [H/gnorm state.∇f/gnorm; state.∇f'/gnorm 0]
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
    (abs(t₀) > 1e-3) && (v = v / t₀)
    vn = norm(v)

    vHv = (v'*H*v/2)[]
    vg = (v'*state.∇f)[]
    bool_reverse_v = vg > 0
    # reverse this v if g'v > 0
    v = (-1)^bool_reverse_v * v
    vg = (-1)^bool_reverse_v * vg
    if iter.linesearch == :trustregion
        state.α, fx, kₜ = TRStyleLineSearch(iter, state.z, v, vHv, vg, 4 * state.Δ / vn)
    elseif iter.linesearch == :hagerzhang
        # use Hager-Zhang line-search algorithm
        s = v
        x = state.x
        state.α, fx, kₜ = HagerZhangLineSearch(iter, state.∇f, state.fx, x, s)
    else
        throw(Error("unknown option of line-search $(iter.linesearch)"))
    end
    # summarize
    state.Δ = state.α * vn
    x = y = state.z + v * state.α
    dq = -state.α^2 * vHv / 2 - state.α * vg
    df = fz - fx
    ro = df / dq
    state.x = x
    state.y = y
    state.fx = fx
    state.ρ = ro
    state.dq = dq
    state.df = df
    state.d = x - z
    state.kλ = kλ
    state.kₜ = kₜ
    state.ϵ = norm(state.∇f)
    state.ξ = ξ
    state.t = (Dates.now() - iter.t).value / 1e3
    counting(iter, state)
    return state, state

end

hsodm_stopping_criterion(tol, state::HSODMState) =
    (state.Δ <= 1e-20) || (state.ϵ <= tol) && abs(state.fz - state.fx) <= tol

function counting(iter::T, state::S) where {T<:HSODMIteration,S<:HSODMState}
    state.kf = getfield(iter.f, :counter)
    state.kg = getfield(iter.g, :counter)
    state.kH = hasproperty(iter.H, :counter) ? getfield(iter.H, :counter) : 0
end


function hsodm_display(k, state::HSODMState)
    if k == 1 || mod(k, 30) == 0
        @printf("%s", LOG_SLOTS)
    end
    @printf("%5d | %+.3e | %.2e | %.2e | %.1e | %+.0e | %.0e | %2d | %6.1f |\n",
        k, state.fx, state.α, state.Δ, state.ϵ, state.λ₁, state.kλ, state.kₜ, state.t
    )
end

default_solution(::HSODMIteration, state::HSODMState) = state.x




HomogeneousSecondOrderDescentMethod(;
    name=:HSODM,
    stop=hsodm_stopping_criterion,
    display=hsodm_display
) = IterativeAlgorithm(HSODMIteration, HSODMState; name=name, stop=stop, display=display)
