using Base.Iterators
using LinearAlgebra
using Printf
using Dates
using KrylovKit

using LineSearches
using SparseArrays

const ATRMS_LOG_SLOTS = @sprintf(
    "%5s | %5s | %11s | %9s | %7s | %6s | %6s | %8s | %6s | %6s \n",
    "k", "kₜ", "f", "|∇f|", "Δ", "σ", "r", "θ", "t", "pts"
)
Base.@kwdef mutable struct ATRMSIteration{Tx,Tf,Tϕ,Tg,TH,Th}
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
    ratio_σ::Float64 = 1.0
    ratio_Δ::Float64 = 10.0
    G₀::Float64 = 1e2
    η::Float64 = 0.3
    γ::Float64 = 0.98
    ψ::Float64 = 1.3
    localthres::Float64 = 1e-8
    # ----------------------------------------------------------------
    itermax::Int64 = 20
    direction = :warm
    linesearch = :none
    adaptive = :none
    verbose::Int64 = 1
    subpstrategy = :monteiro
    mainstrategy = :utr
    adaptiverule = :utr
    initializerule = :undef
    trs::Union{Function,Nothing} = nothing
    LOG_SLOTS::String = ATRMS_LOG_SLOTS
    ALIAS::String = "ATRMS"
    DESC::String = "Accelerated Universal Trust-Region Method"
    error::Union{Nothing,Exception} = nothing
end


Base.IteratorSize(::Type{<:ATRMSIteration}) = Base.IsInfinite()

Base.@kwdef mutable struct ATRMSState{R,Tx}
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
    #
    a::R = 0.0
    A::R = 0.0
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
function Base.iterate(iter::ATRMSIteration)
    iter.t = Dates.now()
    z = copy(iter.x0)
    fz = iter.f(z)
    grad_f_x = iter.g(z)
    Hv = similar(grad_f_x) # this is a buffer for Hvp
    gₙ = norm(grad_f_x, 2)
    n = z |> length
    state = ATRMSState(
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
    iter::ATRMSIteration,
    state::ATRMSState{R,Tx};
) where {R,Tx}
    # use inexact method (a Lanczos method)
    @debug """
    subpstrategy: $(iter.subpstrategy)
    """
    if (iter.subpstrategy == :monteiro)
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

# curves 
a(σ, A) = (1 + sqrt(1 + 4A * σ)) / (2σ)
__y_eval(x, v, σ, a, A) = begin
    ratio = A / (A + a)
    return ratio * x + (1 - ratio) * v
end

function iterate_cholesky_monteiro_svaiter(
    iter::ATRMSIteration,
    state::ATRMSState{R,Tx};
) where {R,Tx}
    @debug """Monteiro Svaiter acceleration style
    """
    state.z = z = state.x
    state.fz = fz = state.fx
    state.∇fz = state.∇f
    n = state.x |> length


    bool_acc = false
    acc_style = :_ # default value


    Mₕ = iter.Mₕ(state.x)
    σ₋ = sqrt(iter.localthres * Mₕ * 1e1)
    σ₊ = sqrt(Mₕ * iter.G₀ / iter.η) / 2
    # σ = max(state.σ / 10, σ₋ * 2)
    σ = σ₋ * 2
    k₂ = 0

    H = zeros(n, n)
    while true
        # if not accepted, try another y
        state.a = a(σ, state.A)
        state.y .= __y_eval(state.x, state.v, σ, state.a, state.A)
        state.∇f .= iter.g(state.y)
        state.ϵ = norm(state.∇f)
        if k₂ % 1 == 0
            H = iter.H(state.y)
        end
        grad_regularizer = max(1e-4, state.ϵ |> sqrt)
        #  λ (dual) must increase
        # @note: 
        # -       σ: not scaled by grad_regularizer
        # - state.σ: σ * grad_regularizer * iter.ratio_σ ...
        state.σ = σ * grad_regularizer * iter.ratio_σ
        state.Δ = Δ = state.σ * grad_regularizer * iter.ratio_Δ / Mₕ
        state.status = true
        v, θ, λ₁, kᵢ = iter.trs(
            H + state.σ * I,
            state.∇f,
            state.Δ;
        )
        # size of the step
        dₙ = v |> norm
        Δ = min(Δ, dₙ)

        ub_θ = (iter.ψ - 1) * σ
        ub_d = iter.η / Mₕ * σ
        if (k₂ == 0) && θ <= ub_θ
            # first iteration pass
            bool_acc = true
            acc_style = :I
        end

        __inner_type = :i
        if θ > ub_θ
            σ₋ = σ
            __inner_type = :i
        elseif (θ == 0) && (dₙ <= ub_d)
            σ₊ = σ
            __inner_type = :ii
        else
            bool_acc = true
            acc_style = :II
            __inner_type = :iii
        end
        if k₂ > 15
            bool_acc = true
            acc_style = :III
            # state.status = false
        end
        σ = (σ₋ + σ₊) / 2
        k₂ += 1
        # @info """k₂: $k₂ __inner_type: $__inner_type
        #     σ: $σ @[$σ₋, $σ₊], 
        #     θ: $θ @[0, $ub_θ], 
        #     dₙ: $dₙ @[0, $ub_d]
        # """
        !bool_acc && continue
        # do this when accept
        # summarize
        state.α = 1.0
        x = y = state.y + v * state.α
        state.v .= state.v - iter.γ * state.a .* iter.g(x)
        state.A += state.a
        fx = iter.f(x)
        df = fz - fx
        dq = -state.α^2 * v'H * v / 2 - state.α * v'state.∇f
        ρₐ = df / dq
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
        state.Δₙ = dₙ
        state.t = (Dates.now() - iter.t).value / 1e3
        counting(iter, state)

        # @info ρₐ
        state.k += 1
        state.acc_style = acc_style
        checknan(state)
        return state, state
    end

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
atr_stopping_criterion(tol, state::ATRMSState) =
    (state.ϵ <= tol) || (state.Δ <= 1e-12)

function counting(iter::T, state::S) where {T<:ATRMSIteration,S<:ATRMSState}
    try
        state.kf = getfield(iter.f, :counter)
        state.kH = hasproperty(iter.H, :counter) ? getfield(iter.H, :counter) : 0
        state.kh = hasproperty(iter.hvp, :counter) ? getfield(iter.hvp, :counter) : 0
        state.kg = getfield(iter.g, :counter)
        state.kgh = state.kg + state.kh * 2
    catch
    end
end

function checknan(state::S) where {S<:ATRMSState}
    if any(isnan, state.x)
        @warn(ErrorException("NaN detected in Lanczos, use debugging to fix"))
    end
end

function atr_display(k, state::ATRMSState)
    if k == 1 || mod(k, 30) == 0
        @printf("%s", UTR_LOG_SLOTS)
    end
    @printf(
        "%5d | %5d | %+.4e | %.3e | %.1e | %+.0e | %+.0e | %+.1e | %6.1f | %s \n",
        k, state.kₜ, state.fx, state.ϵ,
        state.Δ, state.σ, state.r, state.θ, state.t, state.acc_style
    )
end

default_solution(::ATRMSIteration, state::ATRMSState) = state.x


AcceleratedUniversalTrustRegionMonteiroSvaiter(;
    name=:ATRMS,
    stop=atr_stopping_criterion,
    display=atr_display
) = IterativeAlgorithm(ATRMSIteration, ATRMSState; name=name, stop=stop, display=display)



####################################################################################################
# Universal Trust Region Method
####################################################################################################
function (alg::IterativeAlgorithm{T,S})(;
    maxiter=10000,
    maxtime=1e2,
    tol=1e-6,
    freq=10,
    verbose=1,
    direction=:cold,
    adaptive=:none,
    bool_trace=false,
    localthres=tol,
    kwargs...
) where {T<:ATRMSIteration,S<:ATRMSState}

    arr = Vector{S}()
    kwds = Dict(kwargs...)

    for cf ∈ [:f :g :H :hvp]
        apply_counter(cf, kwds)
    end

    iter = T(;
        adaptive=adaptive,
        direction=direction,
        verbose=verbose,
        localthres=localthres,
        kwds...
    )
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


function Base.show(io::IO, t::T) where {T<:ATRMSIteration}
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

function summarize(io::IO, k::Int, t::T, s::S) where {T<:ATRMSIteration,S<:ATRMSState}
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

summarize(k::Int, t::T, s::S) where {T<:ATRMSIteration,S<:ATRMSState} =
    summarize(stdout, k, t, s)




@doc """
    implement the strategy of Mishchenko [Algorithm 2.3, AdaN+](SIOPT, 2023)
    this is always accepting method; †slightly different from UTR version.
"""
function initial_rules_mishchenko(state::ATRMSState, funcg, Hx, funcH, args...)
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

function bisect_μ(iter::ATRMSIteration, state::ATRMSState, Mₕ::Real, H, g, μ₋::Real, μ₊::Real)
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