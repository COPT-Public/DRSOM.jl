
using Base.Iterators
using LinearAlgebra
using Printf
using Dates
using KrylovKit
using Distributions
using LineSearches

const PFH_LOG_SLOTS = @sprintf(
    "%5s | %10s | %10s | %7s | %8s | %8s | %6s | %5s | %6s | %6s |\n",
    "k", "μ", "f", "α", "Δ", "|∇f|", "λ", "kᵤ", "ρ", "t"
)
Base.@kwdef mutable struct PathFollowingHSODMIteration{Tx,Tf,Tϕ,Tg,TH}
    f::Tf             # f: smooth f
    ϕ::Tϕ = nothing   # ϕ: nonsmooth part (not implemented yet)
    g::Tg = nothing   # gradient function
    H::TH = nothing   # hessian function
    x0::Tx            # initial point
    μ₀::Float64       # initial path-following penalty
    t::Dates.DateTime = Dates.now()
    adaptive_param = AR() # todo
    eigtol::Float64 = 1e-10
    itermax::Int64 = 20
    direction = :warm
    linesearch = :none
    adaptive = :none
    verbose::Int64 = 1
    LOG_SLOTS::String = PFH_LOG_SLOTS
    ALIAS::String = "PFH"
    DESC::String = "Path-Following Homogeneous Second-order Descent Method"
    error::Union{Nothing,Exception} = nothing
end

# alias
PFHIteration = PathFollowingHSODMIteration

Base.IteratorSize(::Type{<:PFHIteration}) = Base.IsInfinite()

Base.@kwdef mutable struct PFHState{R,Tx}
    status::Bool = true # status
    x::Tx             # iterate
    ∇fμ::Tx           # gradient of homotopy model at x 
    μ::Float64        # μ in the path-following algorithm
    kᵤ::Int = 0       # outer path-following loop
    ϵμ::R             # eps 3: residual of ∇fμ
    ##################################################################
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



function Base.iterate(iter::PFHIteration)
    iter.t = Dates.now()
    z = copy(iter.x0)
    fz = iter.f(z)
    grad_f_x = similar(z)
    grad_f_b = similar(z)

    grad_f_x = iter.g(z)
    grad_fμ = grad_f_x + iter.μ₀ * z
    H = iter.H(z)
    n = length(z)
    # construct homogeneous system
    B = Symmetric([H grad_fμ; grad_fμ' -iter.μ₀])
    vals, vecs, info = KrylovKit.eigsolve(B, n + 1, 1, :SR, Float64; tol=iter.eigtol)
    kᵥ = info.numops
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
    # # now use a LS to solve (state.α)
    # if iter.linesearch == :trustregion
    #     α, fx, kₜ = TRStyleLineSearch(iter, z, v, vHv, vg, 1.0)
    # elseif iter.linesearch == :hagerzhang
    #     # use Hager-Zhang line-search algorithm
    #     α, fx, kₜ = HagerZhangLineSearch(iter, grad_f_x, fz, z, v)
    # elseif iter.linesearch == :none
    # α = 1.0
    # fx = iter.f(z + v * α)
    # kₜ = 1
    # else
    # end
    α = 1.0
    fx = iter.f(z + v * α)
    kₜ = 1
    y = z + α .* v
    fx = iter.f(y)
    dq = -α^2 * vHv / 2 - α * vg
    df = fz - fx
    ro = df / dq
    Δ = vn * α
    t = (Dates.now() - iter.t).value / 1e3
    d = y - z
    state = PFHState(
        x=y,
        y=y,
        z=z,
        fx=fx,
        fz=fz,
        ∇f=grad_f_x,
        ∇fμ=grad_fμ,
        ∇fz=z,
        α=α,
        d=d,
        Δ=Δ,
        dq=dq,
        df=df,
        ρ=ro,
        ϵ=norm(grad_f_x, 2),
        ϵμ=norm(grad_f_x, 2),
        μ=iter.μ₀,
        γ=1e-6,
        kᵥ=kᵥ,
        kₜ=kₜ,
        t=t,
        δ=0.0,
        ξ=ones(length(z) + 1),
        λ₁=λ₁
    )
    return state, state
end



function Base.iterate(iter::PFHIteration, state::PFHState{R,Tx}) where {R,Tx}

    state.z = z = state.x
    state.fz = fz = state.fx
    state.∇fz = state.∇f

    state.∇f = iter.g(state.x)
    state.∇fμ = state.∇f + state.μ * state.x
    H = iter.H(state.x)

    # construct homogeneous system
    B = Symmetric([H state.∇fμ; state.∇fμ' state.μ])
    kᵥ, k₂, v, vn, vg, vHv = AdaptiveHomogeneousSubproblem(
        B, iter, state, iter.adaptive_param; verbose=iter.verbose > 1
    )

    # kᵥ, k₂, v, vn, vg, vHv = NewtonStep(
    #     H, state.μ, state.∇fμ, state; verbose=iter.verbose > 1
    # )

    # # stepsize choice
    # fh(x) = iter.f(x) + state.μ / 2 * norm(x)^2
    # gh(x) = iter.g(x) + state.μ * x
    # # use Hager-Zhang line-search algorithm
    # state.α, fx, _ = HagerZhangLineSearch(fh, gh, gh(state.x), fh(state.x), state.x, v)
    state.α = 1.0
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


    if state.ϵμ < state.μ
        state.μ = state.μ < 1e-7 ? 0 : 0.6 * state.μ
        state.kᵤ += 1
        state.kₜ = 0
    end
    state.t = (Dates.now() - iter.t).value / 1e3
    counting(iter, state)
    state.status = true
    return state, state

end

pfh_stopping_criterion(tol, state::PFHState) =
    (state.Δ <= 1e-20) || (state.ϵ <= tol) && abs(state.fz - state.fx) <= tol # || (state.μ <= 1e-12)

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
    @printf("%5d | %+.3e | %+.3e | %.1e | %.2e | %.2e | %+.0e | %5d | %+.0e | %6.1f |\n",
        k, state.μ, state.fx, state.α, state.Δ, state.ϵ, state.λ₁, state.kᵤ, state.ρ, state.t
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
    direction=:cold,
    linesearch=:none,
    adaptive=:none,
    μ₀=0.1,
    kwargs...
) where {T<:PFHIteration,S<:PFHState}

    arr = Vector{S}()
    kwds = Dict(kwargs...)

    for cf ∈ [:f :g :H]
        apply_counter(cf, kwds)
    end
    iter = T(; μ₀=μ₀, linesearch=linesearch, adaptive=adaptive, direction=direction, verbose=verbose, kwds...)
    (verbose >= 1) && show(iter)
    for (k, state) in enumerate(iter)

        push!(arr, copy(state))
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
    @printf io " algorithm alias       := %s\n" t.ALIAS
    @printf io " algorithm description := %s\n" t.DESC
    @printf io " inner iteration limit := %s\n" t.itermax
    @printf io " line-search algorithm := %s\n" t.linesearch
    @printf io "   adaptive μ strategy := %s\n" t.adaptive

    println(io, "-"^length(t.LOG_SLOTS))
    flush(io)
end

function summarize(io::IO, k::Int, t::T, s::S) where {T<:PFHIteration,S<:PFHState}
    println(io, "-"^length(t.LOG_SLOTS))
    println(io, "summary:")
    @printf io " (main)          f       := %.2e\n" s.fx
    @printf io " (first-order)  |g|      := %.2e\n" s.ϵ
    println(io, "oracle calls:")
    @printf io " (main)          k       := %d  \n" k
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

