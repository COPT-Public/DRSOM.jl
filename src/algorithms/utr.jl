
using Base.Iterators
using LinearAlgebra
using Printf
using Dates
using KrylovKit
using Distributions
using LineSearches
using SparseArrays

const UTR_LOG_SLOTS = @sprintf(
    "%5s | %5s | %11s | %9s | %7s | %6s | %6s | %8s | %6s \n",
    "k", "kₜ", "f", "|∇f|", "Δ", "σ", "r", "θ", "t"
)
Base.@kwdef mutable struct UTRIteration{Tx,Tf,Tϕ,Tg,TH,Th}
    f::Tf             # f: smooth f
    ϕ::Tϕ = nothing   # ϕ: nonsmooth part (not implemented yet)
    g::Tg = nothing   # gradient function
    hvp::Th = nothing
    fvp::Union{Function,Nothing} = nothing
    ff::Union{Function,Nothing} = nothing
    H::TH = nothing   # hessian function
    x0::Tx            # initial point
    t::Dates.DateTime = Dates.now()
    adaptive_param = AR() # todo
    eigtol::Float64 = 1e-9
    itermax::Int64 = 20
    direction = :warm
    linesearch = :none
    adaptive = :none
    verbose::Int64 = 1
    bool_subp_exact::Int64 = 1
    LOG_SLOTS::String = UTR_LOG_SLOTS
    ALIAS::String = "UTR"
    DESC::String = "Universal Trust-Region Method"
    error::Union{Nothing,Exception} = nothing
end


Base.IteratorSize(::Type{<:UTRIteration}) = Base.IsInfinite()

Base.@kwdef mutable struct UTRState{R,Tx}
    status::Bool = true # status
    x::Tx             # iterate
    fx::R             # new value f at x: x(k)
    fz::R             # old value f at z: x(k-1)
    ∇f::Tx            # gradient of f at x
    ∇fz::Tx           # gradient of f at z
    ∇fb::Tx           # buffer of hvps
    y::Tx             # forward point
    z::Tx             # previous point
    d::Tx             # momentum/fixed-point diff at iterate (= x - z)
    θ::R = 0.0        # dual variable
    α::R = 1.0        # step size
    Δ::R = 0.0        # trust-region radius
    Δₙ::R = 0.0       # norm of this step 
    dq::R = 0.0       # decrease of estimated quadratic model
    df::R = 0.0       # decrease of the real function value
    ρ::R = 0.0        # trs descrease ratio: ρ = df/dq
    ϵ::R = 0.0        # eps 2: residual for gradient 
    r::R = 1e1        # universal trust-region radius         parameter r
    σ::R = 1e-1       # universal trust-region regularization parameter σ
    k::Int = 1        # outer iterations
    kᵥ::Int = 1       # krylov iterations
    kₜ::Int = 1        # inner iterations 
    t::R = 0.0        # running time
    kf::Int = 0       # function evaluations
    kg::Int = 0       # gradient evaluations
    kH::Int = 0       #  hessian evaluations
    kh::Int = 0       #      hvp evaluations
    k₂::Int = 0       # 2 oracle evaluations
end


@doc raw"""
 Initialize the state, change of behavior:
    do not optimize at the first (0th) iterate.
"""
function Base.iterate(iter::UTRIteration)
    iter.t = Dates.now()
    z = copy(iter.x0)
    fz = iter.f(z)
    grad_f_x = iter.g(z)
    Hv = similar(grad_f_x) # this is a buffer for Hvp
    gₙ = norm(grad_f_x, 2)
    state = UTRState(
        x=z,
        y=z,
        z=z,
        d=zeros(z |> size),
        fx=fz,
        fz=fz,
        ∇f=grad_f_x,
        ∇fb=Hv,
        ∇fz=grad_f_x,
        ϵ=gₙ,
        Δ=gₙ * 1e1
    )
    if iter.hvp === nothing
        return state, state
    end

    function hvp(y, v)
        iter.hvp(state.x, v, Hv)
        copy!(y, Hv)
    end
    iter.ff = (y, v) -> hvp(y, v)

    return state, state
end


function iterate_evolve_lanczos(
    iter::UTRIteration,
    state::UTRState{R,Tx};
) where {R,Tx}
    state.z = z = state.x
    state.fz = fz = state.fx
    state.∇fz = state.∇f
    state.∇f = iter.g(state.x)
    state.ϵ = norm(state.∇f)
    grad_regularizer = state.ϵ |> sqrt
    decs_regularizer = grad_regularizer^3

    if iter.hvp === nothing
        H = iter.H(state.x)
    else
        throw(
            ErrorException("currently only support Hessian mode")
        )
    end
    k₂ = 0
    γ = 1.5
    η = 1.0
    ρ = 1.0

    σ = state.σ * grad_regularizer
    Δ = max(state.r * grad_regularizer, 1e-1)

    Df = (η / ρ) * decs_regularizer
    # initialize
    n = state.∇f |> length
    Sₗ = DRSOM.Lanczos(n, 2n + 1, state.∇f)
    while true
        # use evolving subspaces
        v, θ, info = DRSOM.InexactLanczosTrustRegionBisect(
            H,
            -state.∇f,
            Δ,
            Sₗ;
            σ=σ,
            k=Sₗ.k
        )
        λ₁ = info.λ₁

        # construct iterate
        state.α = 1.0
        fx = iter.f(state.z + v * state.α)

        # summarize
        x = y = state.z + v * state.α
        df = fz - fx
        dq = -state.α^2 * v'H * v / 2 - state.α * v'state.∇f
        ρₐ = sign(df) * df / dq
        # @info df, dq
        k₂ += 1
        if (df < 0) || ((df < Df) && (ρₐ < 0.6) && (Δ > 1e-6))  # not satisfactory
            if abs(λ₁) >= 1e-3 # too cvx or ncvx
                σ = 0.0
            else
                σ *= γ
            end
            # dec radius
            Δ /= γ
            Df /= γ
            continue
        end
        if ρₐ > 0.9
            σ /= γ
            Δ *= γ
        end
        # do this when accept
        state.σ = max(1e-8, σ / grad_regularizer)
        state.r = max(Δ / grad_regularizer, 1e-1)
        state.k₂ += k₂
        state.kₜ = k₂
        state.x = x
        state.y = y
        state.fx = fx
        state.ρ = ρₐ
        state.dq = dq
        state.df = df
        state.θ = θ
        state.d = x - z
        state.Δ = Δ
        state.Δₙ = state.d |> norm
        state.t = (Dates.now() - iter.t).value / 1e3
        counting(iter, state)
        state.status = true
        # @info ρₐ
        state.k += 1
        return state, state
    end
end


function iterate_full_lanczos(
    iter::UTRIteration,
    state::UTRState{R,Tx};
) where {R,Tx}

    state.z = z = state.x
    state.fz = fz = state.fx
    state.∇fz = state.∇f
    state.∇f = iter.g(state.x)
    state.ϵ = norm(state.∇f)
    grad_regularizer = state.ϵ |> sqrt
    decs_regularizer = grad_regularizer^3

    if iter.hvp === nothing
        H = iter.H(state.x)
    else
        throw(
            ErrorException("currently only support Hessian mode")
        )
    end
    k₂ = 0
    γ = 1.5
    η = 1.0
    ρ = 1.0
    T, V, _, κ = LanczosTridiag(H, -state.∇f; tol=1e-5, bool_reorth=true)
    λ₁ = eigvals(T, 1:1)[]
    σ = state.σ * grad_regularizer
    Δ = max(state.r * grad_regularizer, 1e-1)
    λ₁ = max(-λ₁, 0)
    λᵤ = state.ϵ / Δ
    Df = (η / ρ) * decs_regularizer
    while true
        v, θ, kᵢ = LanczosTrustRegionBisect(
            T + σ * I,
            V,
            -state.∇f,
            Δ,
            max(0, λ₁ - σ),
            λᵤ;
            bool_interior=true
        )
        state.α = 1.0
        fx = iter.f(state.z + v * state.α)
        @debug """inner""" κ kᵢ
        # summarize
        x = y = state.z + v * state.α
        df = fz - fx
        dq = -state.α^2 * v'H * v / 2 - state.α * v'state.∇f
        ρₐ = df / dq
        k₂ += 1

        if (df < 0) || ((df < Df) && (ρₐ < 0.6) && (Δ > 1e-6))  # not satisfactory
            if abs(λ₁) >= 1e-3 # too cvx or ncvx
                σ = 0.0
            else
                σ *= γ
            end
            # dec radius
            Δ /= γ
            Df /= γ
            continue
        end
        if ρₐ > 0.9
            σ /= γ
            Δ *= γ
        end
        # do this when accept
        state.σ = max(1e-8, σ / grad_regularizer)
        state.r = max(Δ / grad_regularizer, 1e-1)
        state.k₂ += k₂
        state.kₜ = k₂
        state.x = x
        state.y = y
        state.fx = fx
        state.ρ = ρₐ
        state.dq = dq
        state.df = df
        state.θ = θ
        state.d = x - z
        state.Δ = Δ
        state.Δₙ = state.d |> norm
        state.t = (Dates.now() - iter.t).value / 1e3
        counting(iter, state)
        state.status = true
        # @info ρₐ
        state.k += 1
        return state, state
    end

end


function Base.iterate(
    iter::UTRIteration,
    state::UTRState{R,Tx};
) where {R,Tx}
    # use inexact method (a Lanczos method)
    if (iter.bool_subp_exact == 0)
        n = (state.x |> length)
        if n < 2e3
            return iterate_full_lanczos(iter, state)
        else
            # use inexact method of evolving Lanczos
            return iterate_evolve_lanczos(iter, state)
        end
    end

    state.z = z = state.x
    state.fz = fz = state.fx
    state.∇fz = state.∇f
    state.∇f = iter.g(state.x)
    state.ϵ = norm(state.∇f)
    grad_regularizer = state.ϵ |> sqrt
    decs_regularizer = grad_regularizer^3

    if iter.hvp === nothing
        H = iter.H(state.x)
    else
        throw(
            ErrorException("currently only support Hessian mode")
        )
    end
    k₂ = 0
    γ = 1.5
    η = 1.0
    ρ = 1.0

    σ = state.σ * grad_regularizer
    Δ = max(state.r * grad_regularizer, 1e-1)

    Df = (η / ρ) * decs_regularizer
    # initialize
    n = state.∇f |> length
    # dual estimate
    λ₁ = 0.0
    while true

        v, θ, λ₁, kᵢ = TrustRegionCholesky(
            H,
            state.∇f,
            Δ;
            λ₁=λ₁
        )
        state.α = 1.0
        fx = iter.f(state.z + v * state.α)
        # summarize
        x = y = state.z + v * state.α
        df = fz - fx
        dq = -state.α^2 * v'H * v / 2 - state.α * v'state.∇f
        ρₐ = df / dq
        k₂ += 1
        @debug """inner""" v |> norm, Δ, θ, λ₁, kᵢ, df, ρₐ
        Δ = min(Δ, v |> norm)
        if (Δ > 1e-8) && ((df < 0) || ((df < Df) && (ρₐ < 0.6)))  # not satisfactory
            if abs(λ₁) >= 1e-3 # too cvx or ncvx
                σ = 0.0
            else
                σ *= γ
            end
            # dec radius
            Δ /= γ
            Df /= γ
            # in this case, λ (dual) must increase
            continue
        end
        if ρₐ > 0.9
            σ /= γ
            Δ *= γ
        end
        # do this when accept
        state.σ = max(1e-12, σ / grad_regularizer)
        state.r = max(Δ / grad_regularizer, 1e-1)
        state.k₂ += k₂
        state.kₜ = k₂
        state.x = x
        state.y = y
        state.fx = fx
        state.ρ = ρₐ
        state.dq = dq
        state.df = df
        state.θ = θ
        state.d = x - z
        state.Δ = Δ
        state.Δₙ = state.d |> norm
        state.t = (Dates.now() - iter.t).value / 1e3
        counting(iter, state)
        state.status = true
        # @info ρₐ
        state.k += 1
        return state, state
    end

end

utr_stopping_criterion(tol, state::UTRState) =
    (state.ϵ <= tol) || (state.Δ <= 1e-12)

function counting(iter::T, state::S) where {T<:UTRIteration,S<:UTRState}
    try
        state.kf = getfield(iter.f, :counter)
        state.kg = getfield(iter.g, :counter)
        state.kH = hasproperty(iter.H, :counter) ? getfield(iter.H, :counter) : 0
        state.kh = 0 # todo, accept hvp iterative in the future
    catch
    end
end


function utr_display(k, state::UTRState)
    if k == 1 || mod(k, 30) == 0
        @printf("%s", UTR_LOG_SLOTS)
    end
    @printf(
        "%5d | %5d | %+.4e | %.3e | %.1e | %+.0e | %+.0e | %+.1e | %6.1f \n",
        k, state.kₜ, state.fx, state.ϵ, state.Δ, state.σ, state.r, state.θ, state.t
    )
end

default_solution(::UTRIteration, state::UTRState) = state.x




UniversalTrustRegion(;
    name=:UTR,
    stop=utr_stopping_criterion,
    display=utr_display
) = IterativeAlgorithm(UTRIteration, UTRState; name=name, stop=stop, display=display)



####################################################################################################
# Universal Trust Region Method
####################################################################################################
function (alg::IterativeAlgorithm{T,S})(;
    maxiter=10000,
    maxtime=1e2,
    tol=1e-6,
    freq=10,
    verbose=1,
    eigtol=1e-9,
    direction=:cold,
    linesearch=:none,
    adaptive=:none,
    bool_trace=false,
    kwargs...
) where {T<:UTRIteration,S<:UTRState}

    arr = Vector{S}()
    kwds = Dict(kwargs...)

    for cf ∈ [:f :g :H]
        apply_counter(cf, kwds)
    end
    iter = T(; eigtol=eigtol, linesearch=linesearch, adaptive=adaptive, direction=direction, verbose=verbose, kwds...)
    (verbose >= 1) && show(iter)
    for (k, state) in enumerate(iter)

        bool_trace && push!(arr, copy(state))
        if k >= maxiter || state.t >= maxtime || alg.stop(tol, state) || state.status == false
            (verbose >= 1) && alg.display(k, state)
            (verbose >= 1) && summarize(k, iter, state)
            return Result(name=alg.name, iter=iter, state=state, k=k, trajectory=arr)
        end
        (verbose >= 1) && (k == 1 || mod(k, freq) == 0) && alg.display(k, state)
    end
end


function Base.show(io::IO, t::T) where {T<:UTRIteration}
    format_header(t.LOG_SLOTS)
    @printf io "  algorithm alias       := %s\n" t.ALIAS
    @printf io "  algorithm description := %s\n" t.DESC
    @printf io "  inner iteration limit := %s\n" t.itermax
    @printf io "  subproblem            := %s\n" t.bool_subp_exact
    if t.hvp !== nothing
        @printf io "      second-order info := using provided Hessian-vector product\n"
    elseif t.H !== nothing
        @printf io "      second-order info := using provided Hessian matrix\n"
    else
        @printf io " unknown mode to compute Hessian info\n"
        throw(ErrorException("unknown differentiation mode\n"))
    end
    println(io, "-"^length(t.LOG_SLOTS))
    flush(io)
end

function summarize(io::IO, k::Int, t::T, s::S) where {T<:UTRIteration,S<:UTRState}
    println(io, "-"^length(t.LOG_SLOTS))
    println(io, "summary:")
    @printf io " (main)          f       := %.2e\n" s.fx
    @printf io " (first-order)  |g|      := %.2e\n" s.ϵ
    println(io, "oracle calls:")
    @printf io " (main)          k       := %d  \n" s.k
    @printf io " (function)      f       := %d  \n" s.kf
    @printf io " (first-order)   g(+hvp) := %d  \n" s.kg
    @printf io " (second-order)  H       := %d  \n" s.kH
    @printf io " (sub-problem)   P       := %d  \n" s.k₂
    @printf io " (running time)  t       := %.3f  \n" s.t
    println(io, "-"^length(t.LOG_SLOTS))
    flush(io)
end

summarize(k::Int, t::T, s::S) where {T<:UTRIteration,S<:UTRState} =
    summarize(stdout, k, t, s)

