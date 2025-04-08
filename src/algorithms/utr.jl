using Base.Iterators
using LinearAlgebra
using Printf
using Dates
using KrylovKit
using LineSearches
using SparseArrays
using .TRS

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
    fc::Union{Function,Nothing} = nothing
    opH::Union{LinearOperator,Nothing} = nothing
    H::TH = nothing   # hessian function
    x0::Tx            # initial point
    t::Dates.DateTime = Dates.now()
    σ₀::Float64 = 1e-3
    adaptive_param = AR() # todo
    eigtol::Float64 = 1e-9
    itermax::Int64 = 20
    direction = :warm
    linesearch = :none
    adaptive = :none
    verbose::Int64 = 1
    mainstrategy = :utr
    subpstrategy = :direct
    adaptiverule = :utr
    initializerule = :undef
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
    r::R = 1e2        # universal trust-region radius         parameter r
    σ::R = 1e-3       # universal trust-region regularization parameter σ
    k::Int = 1        # outer iterations
    kᵥ::Int = 1       # krylov iterations
    kₜ::Int = 1        # inner iterations 
    t::R = 0.0        # running time
    kf::Int = 0       # function evaluations
    kg::Int = 0       # gradient evaluations
    kgh::Int = 0      # gradient + hvp evaluations
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
    n = z |> length
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
        Δ=gₙ * 1e1,
        σ=iter.σ₀
    )
    if iter.hvp === nothing
        return state, state
    end

    function hvp(v)
        iter.hvp(state.x, v, Hv)
        return Hv
    end
    iter.fc = hvp
    function hvps(y, v)
        iter.hvp(state.x, v, Hv)
        copy!(y, Hv .+ state.σ * v)
    end
    iter.ff = (y, v) -> hvps(y, v)
    iter.opH = LinearOperator(Float64, n, n, true, true, (y, v) -> iter.ff(y, v))
    return state, state
end


function Base.iterate(
    iter::UTRIteration,
    state::UTRState{R,Tx};
) where {R,Tx}
    # use inexact method (a Lanczos method)
    if (iter.subpstrategy == :direct)
        return iterate_cholesky(iter, state)
    elseif (iter.subpstrategy == :lanczos)
        return iterate_evolve_lanczos(iter, state)
    else
        throw(ErrorException("""
        unsupported mode $(iter.subpstrategy),
        currently: {:lanczos, :direct}
        """
        ))
    end
end

function iterate_cholesky(
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
        H = Symmetric(iter.H(state.x))
    else
        throw(
            ErrorException("only support Hessian mode for direct method")
        )
    end
    k₂ = 0
    γ₁ = 2.0
    γ₂ = 2.0
    η = 1.0
    ρ = 1.0

    # initialize an estimation
    if iter.initializerule == :mishchenko
        σ₀, _, Df, bool_acc = initial_rules_mishchenko(
            state, iter.g, H, iter.H,
            state.σ
        )
        σ = σ₀ / 2
        r = 1 / 3 / σ₀
        σ₀ = σ₀ * grad_regularizer
    elseif iter.initializerule == :unscaled
        σ = iter.σ₀
        r = max(state.r, 1e-1)
    elseif iter.initializerule == :classic
        σ = σ₀ = 0.0
        r = state.r
    else
        throw(ErrorException("unrecognized initialize rule $(iter.initializerule)"))
    end
    σ = σ * grad_regularizer
    Δ = r * grad_regularizer
    Df = (η / ρ) * decs_regularizer
    # initialize
    n = state.∇f |> length
    θ = 0.0
    while true
        if iter.mainstrategy == :newton
            # if you use Regularized Newton, 
            #  make sure it is convex optimization
            F = cholesky(H + σ₀ * I)
            v = F \ (-state.∇f)
            kᵢ = 1
            θ = λ₁ = 0.0
        else
            # if not accepted
            #  λ (dual) must increase
            v, θ, λ₁, kᵢ = TrustRegionCholesky(
                H + σ * I,
                state.∇f,
                # 1e3;
                Δ;
                # λₗ=θ - σ
            )
            Δ = min(Δ, v |> norm)
        end
        state.α = 1.0
        fx = iter.f(state.z + v * state.α)
        # summarize
        x = y = state.z + v * state.α
        df = fz - fx
        dq = -state.α^2 * v'H * v / 2 - state.α * v'state.∇f
        ρₐ = df / dq
        k₂ += 1
        @debug """periodic check (main iterate)
            |d|: $(v |> norm):, Δ: $Δ, 
            θ:  $θ, λₗ: $λ₁, 
            kᵢ: $kᵢ, df: $df, 
            ρₐ: $ρₐ
        """
        if iter.adaptiverule == :utr
            Δ, σ, Df, bool_acc = adaptive_rules_utr(
                state, df,
                Df, Δ, σ, ρₐ, γ₁, γ₂,
                θ, λ₁ - σ
            )
        elseif iter.adaptiverule ∈ [:constant, :mishchenko]
            # do nothing
            bool_acc = true
        elseif iter.adaptiverule == :classic
            Δ, σ, Df, bool_acc = classic_rules_utr(
                state, df,
                Df, Δ, σ, ρₐ, γ₁, γ₂ / 1.5,
                θ, λ₁ - σ
            )
        else
            throw(
                ErrorException("unrecognized adaptive mode $(iter.adaptiverule)")
            )
        end
        !bool_acc && continue
        # do this when accept
        if iter.adaptiverule != :classic
            state.σ = max(1e-18, σ / grad_regularizer)
            state.r = max(Δ / grad_regularizer, 1e-1)
        else
            state.r = max(Δ, 1e-8)
        end
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
        checknan(state)
        return state, state
    end
end

function iterate_evolve_lanczos(
    iter::UTRIteration,
    state::UTRState{R,Tx};
) where {R,Tx}
    @debug "start @" Dates.now()
    state.z = z = state.x
    state.fz = fz = state.fx
    state.∇fz = state.∇f
    state.∇f = iter.g(state.x)
    state.ϵ = norm(state.∇f)
    grad_regularizer = state.ϵ |> sqrt
    decs_regularizer = grad_regularizer^3

    n = (state.x |> length)

    k₂ = 0
    γ₁ = 2.0
    γ₂ = 2.0
    η = 1.0
    ρ = 1.0

    # initialize an estimation
    if iter.initializerule == :mishchenko
        throw(ErrorException("""
         not implemented
        """
        ))
        σ₀, _, Df, bool_acc = initial_rules_mishchenko(
            state, iter.g, H, iter.H,
            state.σ
        )
        σ = σ₀ / 2
        r = 1 / 3 / σ₀
        σ₀ = σ₀ * grad_regularizer
        σ = σ * grad_regularizer
        state.σ = σ
    elseif iter.initializerule == :unscaled
        state.σ = σ = iter.σ₀
        r = max(state.r, 1e-1)
    else
        σ = σ₀ = state.σ
        σ = σ * grad_regularizer
        state.σ = σ
        r = max(state.r, 1e-1)
    end
    @debug "initialization @" Dates.now()
    Δ = r * grad_regularizer
    Df = (η / ρ) * decs_regularizer
    # initialize
    n = state.∇f |> length
    θ = λ₁ = 0.0
    while true

        # use evolving subspaces
        state.α = 1.0
        k₁ = (n <= 500) ? round(Int, n * 0.9) : round(Int, n * 0.02)
        Ξ = 1e-1 * min(state.∇f |> norm |> sqrt, 1e0)
        @debug "minimum subspace size $k₁ @" Dates.now()

        if iter.mainstrategy == :newton
            # throw(ErrorException(
            # "we do not support regularized Newton in Lanczos mode yet; use Cholesky instead"
            # ))
            # todo
            if iter.hvp === nothing
                H = iter.H(state.x)
                kᵥ, k₂, v, vn, vg, vHv = NewtonStep(
                    H, state.σ, state.∇f, state; verbose=iter.verbose > 1
                )
            else
                kᵥ, k₂, v, vn, vg, vHv = NewtonStep(
                    iter, state.σ, state.∇f, state; verbose=iter.verbose > 1
                )
            end
            dq = -state.α^2 * vHv / 2 - state.α * vg
            # @printf "NewtonStep: %d %d %e %e %e %e\n" kᵥ k₂ vn vg vHv dq
        else
            Sₗ = DRSOM.Lanczos(n, 2n + 1, state.∇f)
            if iter.hvp === nothing
                H = iter.H(state.x)
                v, θ, info = DRSOM.InexactLanczosTrustRegionBisect(
                    H,
                    -state.∇f,
                    Δ,
                    Sₗ;
                    σ=σ * (σ >= 1e-8),
                    k=Sₗ.k,
                    k₁=k₁,
                    Ξ=Ξ
                )
                dq = -state.α^2 * v'H * v / 2 - state.α * v'state.∇f
            else
                v, θ, info = DRSOM.InexactLanczosTrustRegionBisect(
                    iter.fc,
                    -state.∇f,
                    Δ,
                    Sₗ;
                    σ=σ * (σ >= 1e-8),
                    k=Sₗ.k,
                    k₁=k₁,
                    Ξ=Ξ
                )
                dq = -state.α^2 * v'iter.fc(v) / 2 - state.α * v'state.∇f
            end
            Δ = min(Δ, v |> norm)
            λ₁ = info.λ₁
            kᵥ = info.kₗ
        end
        fx = iter.f(state.z + v * state.α)
        # summarize
        x = y = state.z + v * state.α
        df = fz - fx
        ρₐ = df / dq
        k₂ += 1
        @debug """periodic check (main iterate)
            |d|: $(v |> norm):,
            Ξ: $Ξ, 
            Δ: $Δ, 
            σ: $σ,
            θ: $θ, 
            λ₁: $λ₁, 
            kᵢ: $kᵥ, k₁: $k₁, 
            df: $df, 
            ρₐ: $ρₐ
        """
        if iter.adaptiverule == :utr
            Δ, σ, Df, bool_acc = adaptive_rules_utr(
                state, df,
                Df, Δ, σ, ρₐ, γ₁, γ₂,
                θ, λ₁ - σ
            )
        elseif iter.adaptiverule ∈ [:constant, :mishchenko]
            # do nothing
            bool_acc = true
        elseif iter.adaptiverule == :classic
            Δ, σ, Df, bool_acc = classic_rules_utr(
                state, df,
                Df, Δ, σ, ρₐ, γ₁, γ₂,
                θ, λ₁ - σ
            )
        else
            throw(
                ErrorException("unrecognized adaptive mode $(iter.adaptiverule)")
            )
        end
        !bool_acc && continue
        # do this when accept
        if iter.adaptiverule != :classic
            state.σ = max(1e-18, σ / grad_regularizer)
            state.r = max(Δ / grad_regularizer, 1e-1)
        end

        state.k₂ += k₂
        state.kᵥ += kᵥ
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

@doc """
    implement the strategy of Mishchenko [Algorithm 2.3, AdaN+](SIOPT, 2023)
    this is always accepting method
"""
function initial_rules_mishchenko(state::UTRState, funcg, Hx, funcH, args...)
    σ, _... = args
    Δ = 10.0
    bool_acc = true
    if state.k == 1
        # first iteration
        dx = randn(Float64, state.x |> size)
        y = state.x + dx
        gx = state.∇f
        gy = funcg(y)
        M = approximate_lip(dx, gx, gy, Hx)
        σ1 = √M
    else
        dx = state.d
        gx = state.∇fz
        Hx = funcH(state.z)
        gy = state.∇f
        M = approximate_lip(dx, gx, gy, Hx)
        σ1 = max(
            σ / √2,
            √M
        )
    end
    @debug "details:" norm(dx) norm(gy - gx) σ σ1
    return σ1, M, 0.0, bool_acc
end

function classic_rules_utr(state, args...)
    df, Df, Δ, σ, ρₐ, γ₁, γ₂, θ, λ₁, _... = args
    # @info "details:" λ₁ σ
    bool_acc = true
    if (Δ > 1e-8) && ((df < 0) || ((df < Df) && (ρₐ < 0.2)))  # not satisfactory
        # dec radius
        Δ /= γ₂
        bool_acc = false
    end
    if ρₐ > 0.6
        Δ *= γ₂
    end
    return Δ, σ, Df, bool_acc
end

function adaptive_rules_utr(state, args...)
    df, Df, Δ, σ, ρₐ, γ₁, γ₂, θ, λ₁, _... = args
    # @info "details:" λ₁ σ
    bool_acc = true
    if (Δ > 1e-8) && ((df < 0) || ((df < Df) && (ρₐ < 0.2)))  # not satisfactory
        if abs(λ₁) >= -1e-8 # too cvx or ncvx
            σ = 0.0
        else
            σ *= γ₁
        end
        # dec radius
        Δ /= γ₂
        Df /= γ₁
        bool_acc = false
    end
    if ρₐ > 0.9
        σ /= γ₁
        Δ *= γ₂
    end
    return Δ, σ, Df, bool_acc
end

function approximate_lip(dx, gx, gy, Hx)
    return (norm(gy - gx - Hx * dx)) / norm(dx)^2
end

####################################################################################################
# Basic Tools
####################################################################################################
utr_stopping_criterion(tol, state::UTRState) =
    (state.ϵ <= tol) || (state.Δ <= 1e-12)

function counting(iter::T, state::S) where {T<:UTRIteration,S<:UTRState}
    try
        state.kf = getfield(iter.f, :counter)
        state.kH = hasproperty(iter.H, :counter) ? getfield(iter.H, :counter) : 0
        state.kh = hasproperty(iter.hvp, :counter) ? getfield(iter.hvp, :counter) : 0
        state.kg = getfield(iter.g, :counter)
        state.kgh = state.kg + state.kh * 2
    catch
    end
end

function checknan(state::S) where {S<:UTRState}
    if any(isnan, state.x)
        @warn(ErrorException("NaN detected in Lanczos, use debugging to fix"))
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

    for cf ∈ [:f :g :H :hvp]
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
    @printf io "  main       strategy   := %s\n" t.mainstrategy
    @printf io "  subproblem strategy   := %s\n" t.subpstrategy
    @printf io "  adaptive (σ,Δ) rule   := %s\n" t.adaptiverule
    if t.hvp !== nothing
        @printf io "      second-order info := using provided Hessian-vector product\n"
    elseif t.H !== nothing
        @printf io "      second-order info := using provided Hessian matrix\n"
    else
        @printf io " unknown mode to compute Hessian info\n"
        throw(ErrorException("unknown differentiation mode\n"))
    end

    (t.mainstrategy == :newton) && @printf io "  !!! reduce to Regularized Newton Method\n"
    (t.initializerule == :mishchenko) && @printf io "  !!! - use Mishchenko's strategy\n"
    (t.initializerule == :unscaled) && @printf io "  !!! - use regularization without gradient norm\n"
    (t.adaptiverule == :constant) && @printf io "  !!! - use fixed regularization\n"
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
    @printf io " (first-order)   g       := %d  \n" s.kg
    @printf io " (first-order)   g(+hvp) := %d  \n" s.kgh
    @printf io " (second-order)  H       := %d  \n" s.kH
    @printf io " (second-order)  hvp     := %d  \n" s.kh
    @printf io " (sub-problem)   P       := %d  \n" s.k₂
    @printf io " (sub-calls)     kᵥ      := %d  \n" s.kᵥ
    @printf io " (running time)  t       := %.3f  \n" s.t
    println(io, "-"^length(t.LOG_SLOTS))
    flush(io)
end

summarize(k::Int, t::T, s::S) where {T<:UTRIteration,S<:UTRState} =
    summarize(stdout, k, t, s)



####################################################################################################
# KEEP FOR REFERENCE
# This is a check that lanczos tridiagonalization works,
# @note, comment out later.
####################################################################################################
# function iterate_full_lanczos(
#     iter::UTRIteration,
#     state::UTRState{R,Tx};
# ) where {R,Tx}
#     state.z = z = state.x
#     state.fz = fz = state.fx
#     state.∇fz = state.∇f
#     state.∇f = iter.g(state.x)
#     state.ϵ = norm(state.∇f)
#     grad_regularizer = state.ϵ |> sqrt
#     decs_regularizer = grad_regularizer^3
#     k₂ = 0
#     γ₁ = 8.0
#     γ₂ = 2.0
#     η = 1.0
#     ρ = 1.0
#     if iter.hvp === nothing
#         H = iter.H(state.x)
#     else
#         throw(
#             ErrorException("only support Hessian mode for direct method")
#         )
#     end
#     T, V, _, κ = LanczosTridiag(H, -state.∇f; tol=1e-5, bool_reorth=true)
#     λ₁ = eigvals(T, 1:1)[]
#     σ = state.σ * grad_regularizer
#     Δ = max(state.r * grad_regularizer, 1e-1)
#     λ₁ = max(-λ₁, 0)
#     λᵤ = state.ϵ / Δ
#     Df = (η / ρ) * decs_regularizer
#     while true
#         v, θ, kᵢ = LanczosTrustRegionBisect(
#             T + σ * I,
#             V,
#             -state.∇f,
#             Δ,
#             max(0, λ₁ - σ),
#             λᵤ;
#             bool_interior=false
#         )
#         # construct iterate
#         state.α = 1.0
#         fx = iter.f(state.z + v * state.α)
#         # summarize
#         x = y = state.z + v * state.α
#         df = fz - fx
#         dq = -state.α^2 * v'H * v / 2 - state.α * v'state.∇f
#         ρₐ = df / dq
#         k₂ += 1
#         @info """inner 
#         v |> norm: $(v |> norm), Δ: $Δ 
#         θ:$θ,  λ₁:$λ₁, 
#         kᵢ:$kᵢ, df:$df, ρₐ:$ρₐ 
#         """
#         Δ = min(Δ, v |> norm)
#         if (Δ > 1e-8) && ((df < 0) || ((df < Df) && (ρₐ < 0.2)))  # not satisfactory
#             if abs(λ₁) >= 1e-8 # too cvx or ncvx
#                 σ = 0.0
#             else
#                 σ *= γ₁
#             end
#             # dec radius
#             Δ /= γ₂
#             Df /= γ₁
#             continue
#         end
#         if ρₐ > 0.9
#             σ /= γ₁
#             Δ *= γ₂
#         end
#         # do this when accept
#         state.σ = max(1e-12, σ / grad_regularizer)
#         state.r = max(Δ / grad_regularizer, 1e-1)
#         state.k₂ += k₂
#         state.kₜ = k₂
#         state.x = x
#         state.y = y
#         state.fx = fx
#         state.ρ = ρₐ
#         state.dq = dq
#         state.df = df
#         state.θ = θ
#         state.d = x - z
#         state.Δ = Δ
#         state.Δₙ = state.d |> norm
#         state.t = (Dates.now() - iter.t).value / 1e3
#         counting(iter, state)
#         state.status = true
#         # @info ρₐ
#         state.k += 1
#         return state, state
#     end
# end
