using Base.Iterators
using LinearAlgebra
using Printf
using Dates
using KrylovKit

using LineSearches
using SparseArrays

const ATR_LOG_SLOTS = @sprintf(
    "%5s | %5s | %11s | %9s | %7s | %6s | %6s | %8s | %6s | %6s \n",
    "k", "kₜ", "f", "|∇f|", "Δ", "σ", "r", "θ", "t", "pts"
)
Base.@kwdef mutable struct ATRIteration{Tx,Tf,Tϕ,Tg,TH,Th}
    f::Tf             # f: smooth f
    ϕ::Tϕ = nothing   # ϕ: nonsmooth part (not implemented yet)
    g::Tg = nothing   # gradient function
    hvp::Th = nothing
    fvp::Union{Function,Nothing} = nothing
    ff::Union{Function,Nothing} = nothing
    fc::Union{Function,Nothing} = nothing
    opH::Union{LinearOperator,Nothing} = nothing
    Mₕ::Union{Function,Nothing} = nothing
    H::TH = nothing   # hessian function
    x0::Tx            # initial point
    t::Dates.DateTime = Dates.now()
    # ----------------------------------------------------------------
    # atr parameters
    # scaling parameters for σ and Δ
    ℓ::Float64 = 0.95
    σ₀::Float64 = 1e-3
    ratio_σ::Float64 = 2e1
    ratio_Δ::Float64 = 4e0
    Mμ₋::Float64 = 0.9
    Mμ₊::Float64 = 1.2
    γ₁::Float64 = 0.95
    γ₂::Float64 = 1.0
    # ----------------------------------------------------------------
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
    trs::Union{Function,Nothing} = nothing
    LOG_SLOTS::String = ATR_LOG_SLOTS
    ALIAS::String = "ATR"
    DESC::String = "Accelerated Universal Trust-Region Method"
    error::Union{Nothing,Exception} = nothing
end


Base.IteratorSize(::Type{<:ATRIteration}) = Base.IsInfinite()

Base.@kwdef mutable struct ATRState{R,Tx}
    status::Bool = true # status
    fx::R             # new value f at x: x(k)
    fz::R             # old value f at z: x(k-1)
    ∇f::Tx            # gradient of f at x
    ∇fz::Tx           # gradient of f at z
    ∇fb::Tx           # buffer of hvps
    # ----------------------------------------------------------------
    x::Tx             # iterate
    y::Tx             # forward point
    v::Tx             # remaining point in the estimation sequence
    v₀::Tx             # remaining point in the estimation sequence
    s::Tx             # ancillary point in the estimation sequence
    z::Tx             # previous point
    d::Tx             # momentum/fixed-point diff at iterate (= x - z)
    # ----------------------------------------------------------------
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
    # ----------------------------------------------------------------
    acc_style::Symbol = :_
    acc_count::Dict{Symbol,Int} = Dict(:I => 0, :II => 0, :III => 0)
end

@doc raw"""
 Initialize the state, change of behavior:
    do not optimize at the first (0th) iterate.
"""
function Base.iterate(iter::ATRIteration)
    iter.t = Dates.now()
    z = copy(iter.x0)
    fz = iter.f(z)
    grad_f_x = iter.g(z)
    Hv = similar(grad_f_x) # this is a buffer for Hvp
    gₙ = norm(grad_f_x, 2)
    n = z |> length
    state = ATRState(
        x=z,
        v=z,
        y=z,
        z=z,
        s=zeros(z |> size),
        d=zeros(z |> size),
        v₀=zeros(z |> size),
        fx=fz,
        fz=fz,
        ∇f=grad_f_x,
        ∇fb=Hv,
        ∇fz=grad_f_x,
        ϵ=gₙ,
        Δ=gₙ * 1e1,
        σ=iter.σ₀
    )
    if isnothing(iter.trs)
        iter.trs = TrustRegionCholesky
    end
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
    iter::ATRIteration,
    state::ATRState{R,Tx};
) where {R,Tx}
    # use inexact method (a Lanczos method)
    @debug """
    subpstrategy: $(iter.subpstrategy)
    """
    if (iter.subpstrategy == :direct)
        return iterate_cholesky(iter, state)
    elseif (iter.subpstrategy == :nesterov)
        return iterate_cholesky_nesterov(iter, state)
    elseif (iter.subpstrategy == :monteiro)
        return iterate_cholesky_monteiro_svaiter(iter, state)
    else
        throw(ErrorException("""
        unsupported mode $(iter.subpstrategy),
        currently: {:direct, :nesterov, :monteiro}
        we do not implement Lanczos method yet
        """
        ))
    end
end

function iterate_cholesky(
    iter::ATRIteration,
    state::ATRState{R,Tx};
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
        σ₀, Mₕ, Df, bool_acc = initial_rules_mishchenko(
            state, iter.g, H, iter.H,
            state.σ
        )
        σ₀ = σ₀ / iter.ratio_σ
        r₀ = iter.ratio_Δ / σ₀
        @debug """Mishchenko's initialization
            estimated Mₕ: $Mₕ,
            σ: $σ₀
            r: $r₀
        """
    elseif iter.initializerule == :given
        Mₕ = iter.Mₕ(state.x)
        σ₀ = sqrt(Mₕ) / 3
        r₀ = 1 / sqrt(Mₕ) / 3
        @debug """Given second-order smoothness initialization
            given Mₕ: $Mₕ,
            σ: $σ₀
            r: $r₀
        """
    elseif iter.initializerule == :unscaled
        σ₀ = iter.σ₀
        r₀ = max(state.r, 1e-1)
    else
        σ₀ = state.σ
        r₀ = max(state.r, 1e-1)
    end
    σ = σ₀ * grad_regularizer
    Δ = r₀ * grad_regularizer
    Df = (η / ρ) * decs_regularizer
    @debug """After initialization
        σ: $σ, 
        Δ: $Δ, 
        Df: $Df
        """
    # initialize
    n = state.∇f |> length
    θ = 0.0
    while true

        # if not accepted
        #  λ (dual) must increase
        v, θ, λ₁, kᵢ = iter.trs(
            H + σ * I,
            state.∇f,
            Δ;
            # λₗ=θ - σ
        )
        Δ = min(Δ, v |> norm)

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
        else
            throw(
                ErrorException("unrecognized adaptive mode $(iter.adaptiverule)")
            )
        end
        !bool_acc && continue
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
        state.acc_style = :I
        checknan(state)
        return state, state
    end
end


function iterate_cholesky_nesterov(
    iter::ATRIteration,
    state::ATRState{R,Tx};
) where {R,Tx}
    @debug """Nesterov acceleration style
    """
    state.z = z = state.x
    state.fz = fz = state.fx
    state.∇fz = state.∇f

    # --------------------------------------------------------
    # compute extrapolation
    β = 3
    denom = state.k + β

    # now state.x is the half update after extrapolation
    state.x = state.k / denom * state.z + β / denom * state.v
    # --------------------------------------------------------

    state.∇f = iter.g(state.x)
    state.ϵ = norm(state.∇f)
    grad_regularizer = max(5e-3, state.ϵ |> sqrt)

    if iter.hvp === nothing
        H = Symmetric(iter.H(state.x))
    else
        throw(
            ErrorException("only support Hessian mode for direct method")
        )
    end

    k₂ = 0
    # initialize an estimation
    if iter.initializerule == :mishchenko
        σ₀, Mₕ, Df, bool_acc = initial_rules_mishchenko(
            state, iter.g, H, iter.H,
            state.σ
        )
        # σ₀ = σ₀ / iter.ratio_σ
        # r₀ = iter.ratio_Δ / σ₀
        σ₀ = 1 / 3 * sqrt(Mₕ)
        r₀ = 1 / 3 / sqrt(Mₕ)
        @debug """Mishchenko's initialization
            estimated Mₕ: $Mₕ,
            σ: $σ₀
            r: $r₀
        """
    elseif iter.initializerule == :given
        Mₕ = iter.Mₕ(state.x)
        σ₀ = 0.333 * sqrt(Mₕ)
        r₀ = 1.533 / sqrt(Mₕ)
        @debug """Given second-order smoothness initialization
            given Mₕ: $Mₕ,
            σ: $σ₀
            r: $r₀
        """
    else
        throw(
            ErrorException("unrecognized initializerule $(iter.initializerule)")
        )
    end
    Ω = sqrt(iter.ℓ / (0.8 * Mₕ))
    @debug """Parameter initialization
        estimated Mₕ: $Mₕ,
        σ: $σ₀
        r: $r₀
    """
    σ = σ₀ * grad_regularizer * iter.ratio_σ
    Δ = r₀ * grad_regularizer * iter.ratio_Δ
    @debug """After initialization
        σ: $σ, 
        Δ: $Δ, 
        """

    # --------------------------------------------------------
    # phase-I
    # --------------------------------------------------------
    v, θ, λ₁, kᵢ = iter.trs(H + σ * I, state.∇f, Δ; Δϵ=1e-12)

    @debug """periodic check step I
    |d|: $(v |> norm):, 
    Δ: $Δ, 
    θ:  $θ, 
    λₗ: $λ₁, 
    kᵢ: $kᵢ
    """
    bool_acc = false
    acc_style = :_
    if θ > 0.0
        bool_acc = true
        acc_style = :I
    end
    if acc_style == :I
    else
        @debug """periodic check step II start
        """
        j₁ = 0
        gl = 1e3
        while (j₁ < 30)
            g₊ = iter.g(state.x + v)
            grad_regularizer = g₊ |> norm |> sqrt
            σ = σ₀ * grad_regularizer * iter.ratio_σ
            Δ = r₀ * grad_regularizer
            v, θ, λ₁, kᵢ = iter.trs(
                H + σ * I,
                state.∇f,
                Δ;
                Δϵ=1e-12
            )
            @debug """stage II, inner iterates:
            j₁:  $j₁
            |d|:  $(v |> norm)
            σ:  $σ,
            Δ:  $Δ
            θ:  $θ
            |g₊|:  $(norm(g₊))
            |g|:  $(norm(state.∇f))
            ratio:  $(grad_regularizer ./ gl)
            """
            j₁ += 1
            state.acc_count[:II] += 1
            if θ > 0.0
                break
            end
            if grad_regularizer ./ gl > iter.γ₁
                break
            end
            gl = min(gl, grad_regularizer)
        end
        bool_acc = (θ <= iter.γ₂ * σ)
        acc_style = bool_acc ? :II : :III
        @debug """periodic check step II finish
        """
        Mₜ = iter.γ₂ * Mₕ # the estimated ratio between σ and |d|
        if !bool_acc
            @debug """start smart bisection at μ
            """
            μ₋ = iter.γ₂ * σ
            μ₊ = iter.γ₂ * (σ + θ)
            v, μ, j = bisect_μ(iter, state, Mₜ, H, state.∇f, μ₋, μ₊)
            k₂ += j
        end
    end

    x = state.x + v
    fx = iter.f(x)
    df = fz - fx
    dq = -v'H * v / 2 - v'state.∇f
    ρₐ = df / dq
    # update s, v
    state.s += (state.k + 1) * (state.k + 2) / 2 * iter.g(x)
    ns = norm(state.s)
    state.v = state.v₀ - Ω * state.s / sqrt(ns)
    @debug """periodic check (main iterate)
        |d|: $(v |> norm):, Δ: $Δ, Ω: $Ω
        θ:  $θ, λₗ: $λ₁, 
        kᵢ: $kᵢ
    """

    # do this when accept
    state.σ = max(1e-12, σ / grad_regularizer)
    state.r = max(Δ / grad_regularizer, 1e-1)
    state.k₂ += k₂
    state.kₜ = k₂
    state.x = x
    state.y = x
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
    state.acc_style = acc_style
    state.acc_count[acc_style] += 1
    # @info ρₐ
    state.k += 1
    checknan(state)
    return state, state

end

function iterate_cholesky_monteiro_svaiter(
    iter::ATRIteration,
    state::ATRState{R,Tx};
) where {R,Tx}
    @debug """Monteiro Svaiter acceleration style
    """
    state.z = z = state.x
    state.fz = fz = state.fx
    state.∇fz = state.∇f

    # --------------------------------------------------------
    # compute extrapolation
    β = 3
    denom = state.k + β

    # now state.x is the half update after extrapolation
    state.x = state.k / denom * state.z + β / denom * state.v
    # --------------------------------------------------------

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
    σ₀, Mₕ, Df, bool_acc = initial_rules_mishchenko(
        state, iter.g, H, iter.H,
        state.σ
    )
    σ₀ /= 2e1
    r₀ = 4e0 / σ₀
    Ω = sqrt(iter.ℓ / 3 * Mₕ)
    @debug """Mishchenko's initialization
        estimated Mₕ: $Mₕ,
        σ: $σ₀
        r: $r₀
    """
    σ = σ₀ * grad_regularizer
    Δ = r₀ * grad_regularizer
    @debug """After initialization
        σ: $σ, 
        Δ: $Δ, 
        Df: $Df
        """
    # initialize
    n = state.∇f |> length
    θ = 0.0

    v, θ, λ₁, kᵢ = iter.trs(
        H + σ * I,
        state.∇f,
        Δ;
    )
    @debug """periodic check step I
    |d|: $(v |> norm):, 
    Δ: $Δ, 
    θ:  $θ, 
    λₗ: $λ₁, 
    kᵢ: $kᵢ
    """
    bool_acc = false
    acc_style = :nothing
    if θ > 1e-16
        bool_acc = true
        acc_style = :direct
    end
    @debug """periodic check step II
    """
    j₁ = 0
    while (θ <= 1e-16) && (j₁ < 10)
        g₊ = iter.g(state.x + v)
        grad_regularizer = g₊ |> norm |> sqrt
        σ = σ₀ * grad_regularizer
        Δ = r₀ * grad_regularizer
        v, θ, λ₁, kᵢ = iter.trs(
            H + σ * I,
            state.∇f,
            Δ;
        )
        @debug """inner iterates:
        |d|:  $(v |> norm)
          σ:  $σ,
          Δ:  $Δ
          θ:  $θ
       |g₊|:  $(norm(g₊))
        |g|:  $(norm(state.∇f))
        """
        j₁ += 1
    end
    return 1
    bool_acc = θ <= σ
    acc_style = bool_acc ? :lookahead : :nothing
    if !bool_acc
        μ₋ = σ
        μ₊ = σ + θ
        v, μ, j = bisect_μ(state, Mₕ, H, state.∇f, μ₋, μ₊)
        k₂ += j
        acc_style = :bisection
    end

    x = state.x + v
    fx = iter.f(x)
    df = fz - fx
    dq = -v'H * v / 2 - v'state.∇f
    ρₐ = df / dq
    # update s, v
    state.s += (state.k + 1) * (state.k + 2) / 2 * iter.g(x)
    ns = norm(state.s)
    state.v = state.v₀ - Ω * state.s / sqrt(ns)
    @debug """periodic check (main iterate)
        |d|: $(v |> norm):, Δ: $Δ, 
        θ:  $θ, λₗ: $λ₁, 
        kᵢ: $kᵢ
    """

    # do this when accept
    state.σ = max(1e-12, σ / grad_regularizer)
    state.r = max(Δ / grad_regularizer, 1e-1)
    state.k₂ += k₂
    state.kₜ = k₂
    state.x = x
    state.y = x
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

####################################################################################################
# Basic Tools
####################################################################################################
atr_stopping_criterion(tol, state::ATRState) =
    (state.ϵ <= tol) || (state.Δ <= 1e-12)

function counting(iter::T, state::S) where {T<:ATRIteration,S<:ATRState}
    try
        state.kf = getfield(iter.f, :counter)
        state.kH = hasproperty(iter.H, :counter) ? getfield(iter.H, :counter) : 0
        state.kh = hasproperty(iter.hvp, :counter) ? getfield(iter.hvp, :counter) : 0
        state.kg = getfield(iter.g, :counter)
        state.kgh = state.kg + state.kh * 2
    catch
    end
end

function checknan(state::S) where {S<:ATRState}
    if any(isnan, state.x)
        @warn(ErrorException("NaN detected in Lanczos, use debugging to fix"))
    end
end

function atr_display(k, state::ATRState)
    if k == 1 || mod(k, 30) == 0
        @printf("%s", UTR_LOG_SLOTS)
    end
    @printf(
        "%5d | %5d | %+.4e | %.3e | %.1e | %+.0e | %+.0e | %+.1e | %6.1f | %s \n",
        k, state.kₜ, state.fx, state.ϵ,
        state.Δ, state.σ, state.r, state.θ, state.t, state.acc_style
    )
end

default_solution(::ATRIteration, state::ATRState) = state.x


AcceleratedUniversalTrustRegion(;
    name=:ATR,
    stop=atr_stopping_criterion,
    display=atr_display
) = IterativeAlgorithm(ATRIteration, ATRState; name=name, stop=stop, display=display)



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
) where {T<:ATRIteration,S<:ATRState}

    arr = Vector{S}()
    kwds = Dict(kwargs...)

    for cf ∈ [:f :g :H :hvp]
        apply_counter(cf, kwds)
    end

    iter = T(; eigtol=eigtol, linesearch=linesearch, adaptive=adaptive, direction=direction, verbose=verbose, kwds...)
    for (_k, _) in kwds
        @debug _k getfield(iter, _k)
    end
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


function Base.show(io::IO, t::T) where {T<:ATRIteration}
    format_header(t.LOG_SLOTS)
    @printf io "  algorithm alias       := %s\n" t.ALIAS
    @printf io "  algorithm description := %s\n" t.DESC
    @printf io "  inner iteration limit := %s\n" t.itermax
    @printf io "  main       strategy   := %s\n" t.mainstrategy
    @printf io "  subproblem strategy   := %s\n" t.subpstrategy
    @printf io "  trust-region solver   := %s\n" t.trs
    @printf io "  adaptive (σ,Δ) rule   := %s\n" t.adaptiverule
    if t.hvp !== nothing
        @printf io "      second-order info := using provided Hessian-vector product\n"
    elseif t.H !== nothing
        @printf io "      second-order info := using provided Hessian matrix\n"
    else
        @printf io " unknown mode to compute Hessian info\n"
        throw(ErrorException("unknown differentiation mode\n"))
    end

    (t.initializerule == :mishchenko) && @printf io "  !!! - use Mishchenko's strategy\n"
    (t.adaptiverule == :constant) && @printf io "  !!! - use fixed regularization\n"
    println(io, "-"^length(t.LOG_SLOTS))
    flush(io)
end

function summarize(io::IO, k::Int, t::T, s::S) where {T<:ATRIteration,S<:ATRState}
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

summarize(k::Int, t::T, s::S) where {T<:ATRIteration,S<:ATRState} =
    summarize(stdout, k, t, s)




@doc """
    implement the strategy of Mishchenko [Algorithm 2.3, AdaN+](SIOPT, 2023)
    this is always accepting method; †slightly different from UTR version.
"""
function initial_rules_mishchenko(state::ATRState, funcg, Hx, funcH, args...)
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
        dx = randn(Float64, state.x |> size)
        gx = funcg(state.z)
        Hx = funcH(state.z)
        gy = funcg(state.z + dx)
        M = approximate_lip(dx, gx, gy, Hx)
        σ1 = max(
            σ / √2,
            √M
        )
    end
    @debug "details:" norm(dx) norm(gy - gx) σ σ1 M
    return σ1, M, 0.0, bool_acc
end

function bisect_μ(iter::ATRIteration, state::ATRState, Mₕ::Real, H, g, μ₋::Real, μ₊::Real)
    p = nothing
    j = 0
    v, μ, j = nothing, nothing, 0
    while j < 20
        μ = (μ₋ + μ₊) / 2
        F = cholesky(H + μ * I, check=false, perm=p)
        if !issuccess(F)
            throw(ErrorException("indefinite matrix"))
        end
        p === nothing ? F.p : p
        v = F \ (-g)
        nv = norm(v)
        η = μ / nv

        j += 1
        @debug """bisect_μ 
        j: $j
        μ: $μ
        η: $η ? Mₕ: $Mₕ
        """
        if η < iter.Mμ₋ * Mₕ
            # too large step
            μ₋ = μ
            continue
        elseif η > iter.Mμ₊ * Mₕ
            # too small
            μ₊ = μ
            continue
        end
        break
    end
    return v, μ, j
end