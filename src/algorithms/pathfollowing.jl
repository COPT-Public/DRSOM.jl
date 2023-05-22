
using Base.Iterators
using LinearAlgebra
using Printf
using Dates
using KrylovKit
using Distributions
using LineSearches
using Krylov
using LinearOperators

const PFH_LOG_SLOTS = @sprintf(
    "%4s | %5s | %9s | %10s | %7s | %7s | %9s | %9s | %6s |\n",
    "kᵤ", "k", "μ", "f", "α", "Δ", "|∇f|", "∇fμ", "t"
)
Base.@kwdef mutable struct PathFollowingHSODMIteration{Tx,Tf,Tϕ,Tg,TH,Th}
    f::Tf             # f: smooth f
    ϕ::Tϕ = nothing   # ϕ: nonsmooth part (not implemented yet)
    g::Tg = nothing   # gradient function
    hvp::Th = nothing
    fvp::Union{Function,Nothing} = nothing
    ff::Union{Function,Nothing} = nothing
    H::TH = nothing   # hessian function
    x0::Tx            # initial point
    μ₀::Float64       # initial path-following penalty
    t::Dates.DateTime = Dates.now()
    adaptive_param = AR() # todo
    eigtol::Float64 = 1e-9
    itermax::Int64 = 20
    direction = :warm
    linesearch = :none
    adaptive = :none
    step = :hsodm
    verbose::Int64 = 1
    LOG_SLOTS::String = PFH_LOG_SLOTS
    ALIAS::String = "PFH"
    DESC::String = "Path-Following Homotopy Method"
    error::Union{Nothing,Exception} = nothing
end

# alias
PFHIteration = PathFollowingHSODMIteration

Base.IteratorSize(::Type{<:PFHIteration}) = Base.IsInfinite()

Base.@kwdef mutable struct PFHState{R,Tx,Tg}
    status::Bool = true # status
    x::Tx             # iterate
    ∇fμ::Tg           # gradient of homotopy model at x 
    μ::Float64        # μ in the path-following algorithm
    kᵤ::Int = 0       # outer path-following loop
    ϵμ::R             # eps 3: residual of ∇fμ
    ##################################################################
    fx::R             # new value f at x: x(k)
    fz::R             # old value f at z: x(k-1)
    ∇f::Tg            # gradient of f at x
    ∇fb::Tg           # gradient buffer
    ∇fz::Tg           # gradient of f at z
    y::Tx             # forward point
    z::Tx             # previous point
    d::Tx             # momentum/fixed-point diff at iterate (= x - z)
    Δ::R = 1e6        # trs radius
    dq::R = 0.0       # decrease of estimated quadratic model
    df::R = 0.0       # decrease of the real function value
    ρ::R = 0.0        # trs descrease ratio: ρ = df/dq
    ϵ::R = 0.0        # eps 2: residual for gradient 
    α::R = 0.0       # step size
    γ::R = 1e-5       # trust-region style parameter γ
    σ::R = 1e3        # adaptive arc style parameter σ
    kᵥ::Int = 1       # krylov iterations
    kₜ::Int = 1        # inner iterations 
    t::R = 0.0        # running time
    λ₁::Float64 = 0.0 # smallest curvature if available
    δ::Float64 = -1.0  # smallest curvature if available
    ξ::Tx             # eigenvector
    kf::Int = 0       # function evaluations
    kg::Int = 0       # gradient evaluations
    kH::Int = 0       #  hessian evaluations
    kh::Int = 0       #      hvp evaluations
    k₂::Int = 0       # 2 oracle evaluations
end


@doc raw"""
 Initialize the PFH state, change of behavior:
    do not optimize at the first iterate.
"""
function Base.iterate(iter::PFHIteration)
    iter.t = Dates.now()
    z = copy(iter.x0)
    fz = iter.f(z)
    grad_f_x = iter.g(z)
    grad_fμ = grad_f_x + iter.μ₀ * z
    Hv = similar(grad_f_x)
    state = PFHState(
        x=z,
        y=z,
        z=z,
        d=zeros(z |> size),
        fx=fz,
        fz=fz,
        ∇f=grad_f_x,
        ∇fμ=grad_fμ,
        ∇fb=Hv,
        ∇fz=grad_f_x,
        ϵ=norm(grad_f_x, 2),
        ϵμ=norm(grad_fμ, 2),
        μ=iter.μ₀,
        γ=1e-6,
        ξ=ones(length(z) + 1),
        λ₁=0.0
    )
    if iter.hvp === nothing
        return state, state
    end
    if iter.step == :hsodm
        fvp(x, g, v, Hv, d) = (
            iter.hvp(x, v[1:end-1], Hv);
            [
                Hv + v[end] * g
                g'v[1:end-1] + d * v[end]
            ]
        )
        iter.fvp = fvp
        ff(v) = iter.fvp(state.x, state.∇fμ, v, state.∇fb, -state.μ)
        iter.ff = ff
    else
        function hvp(y, v)
            iter.hvp(state.x, v, Hv)
            copy!(y, Hv .+ state.μ * v)
        end
        iter.ff = (y, v) -> hvp(y, v)
    end
    return state, state
end



function Base.iterate(iter::PFHIteration, state::PFHState{R,Tx}) where {R,Tx}

    state.z = z = state.x
    state.fz = fz = state.fx
    state.∇fz = state.∇f

    state.∇f = iter.g(state.x)
    state.∇fμ = state.∇f + state.μ * state.x

    fh(x) = iter.f(x) + state.μ / 2 * norm(x)^2
    gh(x) = iter.g(x) + state.μ * x
    if iter.step == :hsodm
        # construct homogeneous system
        if iter.hvp === nothing
            H = iter.H(state.x)
            B = Symmetric([H state.∇fμ; state.∇fμ' -state.μ])
            kᵥ, k₂, v, vn, vg, vHv = AdaptiveHomogeneousSubproblem(
                B, iter, state, iter.adaptive_param; verbose=iter.verbose > 1
            )
        else

            # H = iter.H(state.x)
            # B = Symmetric([H state.∇fμ; state.∇fμ' -state.μ])
            # q = randn((state.x |> length) + 1)
            # @assert abs.(B * q - ff(q)) |> maximum < 1e-6
            kᵥ, k₂, v, vn, vg, vHv = AdaptiveHomogeneousSubproblem(
                iter.ff, iter, state, iter.adaptive_param; verbose=iter.verbose > 1
            )
        end
        if iter.linesearch == :backtrack
            state.α, fx = BacktrackLineSearch(fh, gh, gh(state.x), fh(state.x), state.x, v)
        else
            state.α, fx = HagerZhangLineSearch(fh, gh, gh(state.x), fh(state.x), state.x, v)
        end
    else
        if iter.hvp === nothing
            H = iter.H(state.x)
            kᵥ, k₂, v, vn, vg, vHv = NewtonStep(
                H, state.μ, state.∇fμ, state; verbose=iter.verbose > 1
            )
        else

            # println(hvp(state.x))
            kᵥ, k₂, v, vn, vg, vHv = NewtonStep(
                iter, state.μ, state.∇fμ, state; verbose=iter.verbose > 1
            )
        end
        # stepsize choice
        # use Hager-Zhang line-search algorithm
        if iter.linesearch == :backtrack
            state.α, fx = BacktrackLineSearch(fh, gh, gh(state.x), fh(state.x), state.x, v)

        else
            state.α, fx = HagerZhangLineSearch(fh, gh, gh(state.x), fh(state.x), state.x, v)
        end
    end
    (state.α == 0.0) && (state.α = 0.1)
    # state.α = max(1, 1 - state.μ)
    fx = iter.f(state.z + v * state.α)

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
    state.kᵥ = kᵥ
    state.k₂ += k₂
    state.kₜ += 1
    state.ϵ = norm(state.∇f)
    state.ϵμ = norm(state.∇fμ)

    if state.ϵμ < min(5e-1, 10 * state.μ)

        # state.μ = state.μ < 2e-6 ? 0 : 0.06 * state.μ
        state.μ = 0.02 * state.μ
        state.kᵤ += 1
        state.kₜ = 0
    end
    state.t = (Dates.now() - iter.t).value / 1e3
    counting(iter, state)
    state.status = true
    return state, state

end

pfh_stopping_criterion(tol, state::PFHState) =
    (state.ϵ <= tol) || ((state.ϵμ <= tol) && (state.μ <= 1e-12))

function counting(iter::T, state::S) where {T<:PFHIteration,S<:PFHState}
    try
        state.kf = getfield(iter.f, :counter)
        state.kg = getfield(iter.g, :counter)
        state.kH = hasproperty(iter.H, :counter) ? getfield(iter.H, :counter) : 0
        state.kh = 0 # todo, accept hvp iterative in the future
    catch
    end
end


function pfh_display(k, state::PFHState)
    if k == 1 || mod(k, 30) == 0
        @printf("%s", PFH_LOG_SLOTS)
    end
    @printf(
        "%4d | %5d | %+.2e | %+.3e | %.1e | %.1e | %+.2e | %+.2e | %6.1f |\n",
        state.kᵤ, k, state.μ, state.fx, state.α, state.Δ, state.ϵ, state.ϵμ, state.t
    )
end

default_solution(::PFHIteration, state::PFHState) = state.x




PathFollowingHSODM(;
    name=:PFH,
    stop=pfh_stopping_criterion,
    display=pfh_display
) = IterativeAlgorithm(PFHIteration, PFHState; name=name, stop=stop, display=display)



####################################################################################################
# Path-Following HSODM
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
    μ₀=0.1,
    bool_trace=true,
    kwargs...
) where {T<:PFHIteration,S<:PFHState}

    arr = Vector{S}()
    kwds = Dict(kwargs...)

    for cf ∈ [:f :g :H]
        apply_counter(cf, kwds)
    end
    iter = T(; μ₀=μ₀, eigtol=eigtol, linesearch=linesearch, adaptive=adaptive, direction=direction, verbose=verbose, kwds...)
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


function Base.show(io::IO, t::T) where {T<:PFHIteration}
    format_header(t.LOG_SLOTS)
    @printf io "  algorithm alias       := %s\n" t.ALIAS
    @printf io "  algorithm description := %s\n" t.DESC
    @printf io "  inner iteration limit := %s\n" t.itermax
    @printf io "  line-search algorithm := %s\n" t.linesearch
    @printf io "    adaptive μ strategy := %s\n" t.adaptive
    @printf io " homotopy step strategy := %s\n" t.step
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

function summarize(io::IO, k::Int, t::T, s::S) where {T<:PFHIteration,S<:PFHState}
    println(io, "-"^length(t.LOG_SLOTS))
    println(io, "summary:")
    @printf io " (main)          f       := %.2e\n" s.fx
    @printf io " (first-order)  |g|      := %.2e\n" s.ϵ
    println(io, "oracle calls:")
    @printf io " (main)          kᵤ      := %d  \n" s.kᵤ
    @printf io " (main-total)    k       := %d  \n" k
    @printf io " (function)      f       := %d  \n" s.kf
    @printf io " (first-order)   g(+hvp) := %d  \n" s.kg
    @printf io " (second-order)  H       := %d  \n" s.kH
    @printf io " (sub-problem)   P       := %d  \n" s.k₂
    @printf io " (running time)  t       := %.3f  \n" s.t
    println(io, "-"^length(t.LOG_SLOTS))
    flush(io)
end

summarize(k::Int, t::T, s::S) where {T<:PFHIteration,S<:PFHState} =
    summarize(stdout, k, t, s)

