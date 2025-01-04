using Base.Iterators
using LinearAlgebra
using Printf
using Dates
using KrylovKit

using LineSearches
using SparseArrays

const CUBIC_LOG_SLOTS = @sprintf(
    "%5s | %5s | %11s | %9s | %7s | %8s | %8s | %6s | %6s \n",
    "k", "kₜ", "f", "|∇f|", "Δ", "θ", "ε", "t", "pts"
)
Base.@kwdef mutable struct CubicIteration{Tx,Tf,Tϕ,Tg,TH,Th}
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
    mainstrategy = :cubic
    subpstrategy = :direct
    initializerule = :undef
    trs::Union{Function,Nothing} = nothing
    LOG_SLOTS::String = CUBIC_LOG_SLOTS
    ALIAS::String = "Cubic"
    DESC::String = "Vanilla Cubic Regularization"
    error::Union{Nothing,Exception} = nothing
end


Base.IteratorSize(::Type{<:CubicIteration}) = Base.IsInfinite()

Base.@kwdef mutable struct CubicState{R,Tx}
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
    ϵ::R = 0.0        # gradient norm / dual feasibility
    ϵₚ::R = 0.0       # subproblem residual
    Δ::R = 0.0       # norm of this step 
    t::R = 0.0        # running time
    k::Int = 1        # outer iterations
    kᵥ::Int = 1       # krylov iterations
    kf::Int = 0       # function evaluations
    kg::Int = 0       # gradient evaluations
    kgh::Int = 0      # gradient + hvp evaluations
    kH::Int = 0       #  hessian evaluations
    kh::Int = 0       #      hvp evaluations
    # ----------------------------------------------------------------
    acc_style::Symbol = :_
    acc_count::Dict{Symbol,Int} = Dict(:I => 0, :II => 0, :III => 0)
end

@doc raw"""
 Initialize the state, change of behavior:
    do not optimize at the first (0th) iterate.
"""
function Base.iterate(iter::CubicIteration)
    iter.t = Dates.now()
    z = copy(iter.x0)
    fz = iter.f(z)
    grad_f_x = iter.g(z)
    Hv = similar(grad_f_x) # this is a buffer for Hvp
    gₙ = norm(grad_f_x, 2)
    n = z |> length
    state = CubicState(
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
    )
    if isnothing(iter.trs)
        iter.trs = CubicSubpCholesky
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
    iter::CubicIteration,
    state::CubicState{R,Tx};
) where {R,Tx}
    # use inexact method (a Lanczos method)
    @debug """
    subpstrategy: $(iter.subpstrategy)
    """
    if (iter.subpstrategy == :direct)
        return iterate_cholesky(iter, state)
    elseif (iter.subpstrategy == :nesterov)
        return iterate_cholesky_nesterov(iter, state)
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
    iter::CubicIteration,
    state::CubicState{R,Tx};
) where {R,Tx}

    state.z = z = state.x
    state.fz = fz = state.fx
    state.∇fz = state.∇f
    state.∇f = iter.g(state.x)
    state.ϵ = norm(state.∇f)

    if iter.hvp === nothing
        H = Symmetric(iter.H(state.x))
    else
        throw(
            ErrorException("only support Hessian mode for direct method")
        )
    end

    if iter.initializerule == :given
        Mₕ = iter.Mₕ(state.x)
        @debug """Given second-order smoothness initialization
            given Mₕ: $Mₕ,
        """
    else
        throw(
            ErrorException("unrecognized initializerule $(iter.initializerule)")
        )
    end
    Lₖ = 2 * Mₕ
    v, θ, λ₁, kᵢ = iter.trs(
        H,
        state.∇f,
        Lₖ;
    )
    Δ = v |> norm

    state.α = 1.0
    fx = iter.f(state.z + v * state.α)
    # summarize
    x = y = state.z + v * state.α

    @debug """periodic check (main iterate)
        |d|: $(v |> norm):, Δ: $Δ, 
        θ:  $θ, λₗ: $λ₁, 
        kᵢ: $kᵢ, df: $df, 
        ρₐ: $ρₐ
    """

    # do this when accept
    state.x = x
    state.y = y
    state.fx = fx
    state.θ = θ
    state.d = x - z
    state.Δ = Δ
    state.t = (Dates.now() - iter.t).value / 1e3
    counting(iter, state)
    state.status = true
    # @info ρₐ
    state.k += 1
    state.acc_style = :I
    checknan(state)
    return state, state
end


function iterate_cholesky_nesterov(
    iter::CubicIteration,
    state::CubicState{R,Tx};
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

    state.z = z = state.x
    state.fz = fz = state.fx
    state.∇fz = state.∇f
    state.∇f = iter.g(state.x)
    state.ϵ = norm(state.∇f)

    if iter.hvp === nothing
        H = Symmetric(iter.H(state.x))
    else
        throw(
            ErrorException("only support Hessian mode for direct method")
        )
    end

    if iter.initializerule == :given
        Mₕ = iter.Mₕ(state.x)
        @debug """Given second-order smoothness initialization
            given Mₕ: $Mₕ,
        """
    else
        throw(
            ErrorException("unrecognized initializerule $(iter.initializerule)")
        )
    end

    Lₖ = 2 * Mₕ
    v, θ, λ₁, kᵢ = iter.trs(
        H,
        state.∇f,
        Lₖ
    )

    # check subp residual
    Δ = v |> norm
    r = Δ - 2 * θ / Lₖ

    state.α = 1.0
    fx = iter.f(state.z + v * state.α)
    # summarize
    x = y = state.z + v * state.α
    # update s, v
    Ω = sqrt(1 / (3Mₕ))
    state.s += (state.k + 1) * (state.k + 2) / 2 * iter.g(x)
    ns = norm(state.s)
    state.v = state.v₀ - Ω * state.s / sqrt(ns)
    @debug """periodic check (main iterate)
        |d|: $(v |> norm):, Δ: $Δ, 
        θ:  $θ, λₗ: $λ₁, 
        kᵢ: $kᵢ, df: $df, 
        ρₐ: $ρₐ
    """

    # do this when accept
    state.x = x
    state.y = y
    state.fx = fx
    state.ϵₚ = r
    state.θ = θ
    state.d = x - z
    state.Δ = Δ
    state.t = (Dates.now() - iter.t).value / 1e3
    counting(iter, state)
    state.status = true
    state.k += 1
    state.acc_style = :I
    checknan(state)
    return state, state
end

function iterate_cholesky_monteiro_svaiter(
    iter::CubicIteration,
    state::CubicState{R,Tx};
) where {R,Tx}
    throw(
        ErrorException("not implemented yet $(iter.subpstrategy)")
    )
end

####################################################################################################
# Basic Tools
####################################################################################################
cubic_stopping_criterion(tol, state::CubicState) =
    (state.ϵ <= tol) || (state.Δ <= 1e-12)

function counting(iter::T, state::S) where {T<:CubicIteration,S<:CubicState}
    try
        state.kf = getfield(iter.f, :counter)
        state.kH = hasproperty(iter.H, :counter) ? getfield(iter.H, :counter) : 0
        state.kh = hasproperty(iter.hvp, :counter) ? getfield(iter.hvp, :counter) : 0
        state.kg = getfield(iter.g, :counter)
        state.kgh = state.kg + state.kh * 2
    catch
    end
end

function checknan(state::S) where {S<:CubicState}
    if any(isnan, state.x)
        @warn(ErrorException("NaN detected in Lanczos, use debugging to fix"))
    end
end

function cubic_display(k, state::CubicState)
    if k == 1 || mod(k, 30) == 0
        @printf("%s", CUBIC_LOG_SLOTS)
    end
    @printf(
        "%5d | %5d | %+.4e | %.3e | %.1e | %+.1e | %+.1e | %6.1f | %s \n",
        k, state.k, state.fx, state.ϵ,
        state.Δ, state.θ, state.ϵₚ, state.t, state.acc_style
    )
end

default_solution(::CubicIteration, state::CubicState) = state.x


CubicRegularizationVanilla(;
    name=:Cubic,
    stop=cubic_stopping_criterion,
    display=cubic_display
) = IterativeAlgorithm(CubicIteration, CubicState; name=name, stop=stop, display=display)



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
) where {T<:CubicIteration,S<:CubicState}

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


function Base.show(io::IO, t::T) where {T<:CubicIteration}
    format_header(t.LOG_SLOTS)
    @printf io "  algorithm alias       := %s\n" t.ALIAS
    @printf io "  algorithm description := %s\n" t.DESC
    @printf io "  inner iteration limit := %s\n" t.itermax
    @printf io "  main       strategy   := %s\n" t.mainstrategy
    @printf io "  subproblem strategy   := %s\n" t.subpstrategy
    @printf io "  subproblem   solver   := %s\n" t.trs
    if t.hvp !== nothing
        @printf io "      second-order info := using provided Hessian-vector product\n"
    elseif t.H !== nothing
        @printf io "      second-order info := using provided Hessian matrix\n"
    else
        @printf io " unknown mode to compute Hessian info\n"
        throw(ErrorException("unknown differentiation mode\n"))
    end

    (t.initializerule == :mishchenko) && @printf io "  !!! - use Mishchenko's strategy\n"
    println(io, "-"^length(t.LOG_SLOTS))
    flush(io)
end

function summarize(io::IO, k::Int, t::T, s::S) where {T<:CubicIteration,S<:CubicState}
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
    @printf io " (sub-calls)     kᵥ      := %d  \n" s.kᵥ
    @printf io " (running time)  t       := %.3f  \n" s.t
    println(io, "-"^length(t.LOG_SLOTS))
    flush(io)
end

summarize(k::Int, t::T, s::S) where {T<:CubicIteration,S<:CubicState} =
    summarize(stdout, k, t, s)
