
using Base.Iterators
using LinearAlgebra
using Printf
using Dates
using KrylovKit
using Distributions
using LineSearches
using SparseArrays

const HSODM_LOG_SLOTS = @sprintf(
    "%5s | %10s | %8s | %8s | %8s | %6s | %6s | %5s | %6s | %6s |\n",
    "k", "f", "α", "Δ", "|∇f|", "λ", "δ", "k₂", "ρ", "t"
)
Base.@kwdef mutable struct HSODMIteration{Tx,Tf,Tϕ,Tg,TH,Th}
    f::Tf             # f: smooth f
    ϕ::Tϕ = nothing   # ϕ: nonsmooth part (not implemented yet)
    g::Tg = nothing   # gradient function
    H::TH = nothing   # hessian function
    hvp::Th = nothing # hessian-vector product
    fvp::Union{Function,Nothing} = nothing # hessian-vector product of Fk (defined from hvp)
    x0::Tx            # initial point
    t::Dates.DateTime = Dates.now()
    adaptive_param = AR() # todo
    eigtol::Float64 = 1e-12
    itermax::Int64 = 20
    direction = :warm
    linesearch = :hagerzhang
    adaptive = :none
    verbose::Int64 = 1
    LOG_SLOTS::String = HSODM_LOG_SLOTS
    ALIAS::String = "HSODM"
    DESC::String = "Homogeneous Second-order Descent Method"
    error::Union{Nothing,Exception} = nothing
end


Base.IteratorSize(::Type{<:HSODMIteration}) = Base.IsInfinite()

Base.@kwdef mutable struct HSODMState{R,Tx}
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
    Δ::R              # trust-region radius
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
    δ::Float64 = 1e-3  # smallest curvature if available
    ξ::Tx             # eigenvector
    kf::Int = 0       # function evaluations
    kg::Int = 0       # gradient evaluations
    kH::Int = 0       #  hessian evaluations
    kh::Int = 0       #      hvp evaluations
    k₂::Int = 0       # 2 oracle evaluations
end



function Base.iterate(iter::HSODMIteration)
    iter.t = Dates.now()
    z = copy(iter.x0)
    fz = iter.f(z)
    grad_f_x = similar(z)
    grad_f_x = iter.g(z)
    n = length(z)
    Hv = similar(grad_f_x)
    if iter.hvp === nothing
        H = iter.H(z)
        # construct homogeneous system
        B = Symmetric([H grad_f_x; SparseArrays.spzeros(n)' -1e-3])
        vals, vecs, info = KrylovKit.eigsolve(
            B, n + 1, 1, :SR, Float64;
            issymmetric=true, tol=iter.eigtol
        )
    else
        fvp(x, g, v, Hv, d) = (
            iter.hvp(x, v[1:end-1], Hv);
            [
                Hv + v[end] * g
                g'v[1:end-1] + d * v[end]
            ]
        )
        iter.fvp = fvp
        ff(v) = iter.fvp(z, grad_f_x, v, Hv, 0)
        vals, vecs, info = KrylovKit.eigsolve(
            ff, n + 1, 1, :SR, Float64;
            issymmetric=true, tol=iter.eigtol
        )
    end
    kᵥ = info.numops
    λ₁ = vals[1]
    ξ = vecs[1]
    v = reshape(ξ[1:end-1], n)
    t₀ = ξ[end]
    (abs(t₀) > 1e-3) && (v = v / t₀)

    vn = norm(v)
    if iter.hvp === nothing
        vHv = (v'*H*v/2)[]
    else
        vHv = v' * Hv
    end
    vg = (v'*grad_f_x)[]
    bool_reverse_v = vg > 0
    # reverse this v if g'v > 0
    v = (-1)^bool_reverse_v * v
    vg = (-1)^bool_reverse_v * vg
    α = 0.0
    # now use a LS to solve (state.α)
    if iter.linesearch == :hagerzhang
        # use Hager-Zhang line-search algorithm
        α, fx, kₜ = HagerZhangLineSearch(iter, grad_f_x, fz, z, v)
    end
    if (α == 0) || (iter.linesearch == :trustregion)
        α, fx, kₜ = TRStyleLineSearch(iter, z, v, vHv, vg, 1.0)
    end
    if (α == 0) || (iter.linesearch == :backtrack)
        # use Hager-Zhang line-search algorithm
        α, fx, kₜ = BacktrackLineSearch(iter, grad_f_x, fz, z, v)
    end
    if (α == 0) || (iter.linesearch == :none)
        α = 1.0
        fx = iter.f(z + v * α)
        kₜ = 1
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
        ∇fb=Hv,
        α=α,
        d=d,
        Δ=Δ,
        dq=dq,
        df=df,
        ρ=ro,
        ϵ=norm(grad_f_x, 2),
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



function Base.iterate(iter::HSODMIteration, state::HSODMState{R,Tx}) where {R,Tx}

    state.z = z = state.x
    state.fz = fz = state.fx
    state.∇fz = state.∇f

    state.∇f = iter.g(state.x)
    n = state.∇f |> length
    ng = norm(state.∇f)

    if iter.hvp === nothing
        H = iter.H(state.x)
        # construct homogeneous system
        B = Symmetric([H state.∇f; SparseArrays.spzeros(n)' state.δ])

        kᵥ, k₂, v, vn, vg, vHv = AdaptiveHomogeneousSubproblem(
            B, iter, state, iter.adaptive_param; verbose=iter.verbose > 1
        )
    else

        n = length(z)
        ff(v) = iter.fvp(state.x, state.∇f, v, state.∇fb, state.δ)
        kᵥ, k₂, v, vn, vg, vHv = AdaptiveHomogeneousSubproblem(
            ff, iter, state, iter.adaptive_param; verbose=iter.verbose > 1
        )
    end
    # add line search over computed direction 
    if iter.linesearch == :hagerzhang
        # use Hager-Zhang line-search algorithm
        s = v
        x = state.x
        state.α, fx, kₜ = HagerZhangLineSearch(iter, state.∇f, state.fx, x, s)
    end
    if (state.α == 0) || (iter.linesearch == :trustregion)
        state.α, fx, kₜ = TRStyleLineSearch(iter, state.z, v, vHv, vg, 4 * state.Δ / vn)
    end
    if (state.α == 0) || (iter.linesearch == :backtrack)
        # use Hager-Zhang line-search algorithm
        s = v
        x = state.x
        state.α, fx, kₜ = BacktrackLineSearch(iter, state.∇f, state.fx, x, s)
    end
    if (state.α == 0) || (iter.linesearch == :none)
        state.α = 1.0
        fx = iter.f(state.z + v * state.α)
        kₜ = 1
    end
    if (state.α == 0)
        state.status = false
        iter.error = ErrorException(
            "unknown option of line-search $(iter.linesearch)"
        )
        return state, state
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
    state.kᵥ = kᵥ
    state.k₂ += k₂
    state.kₜ = kₜ
    state.ϵ = norm(state.∇f)
    state.t = (Dates.now() - iter.t).value / 1e3

    if ng > 5e-3
        state.δ = 1e-1
    end
    if state.λ₁ < 0
        state.δ = state.δ + 8e-3
    else
        state.δ = state.δ - 1e-2
    end
    counting(iter, state)
    state.status = true
    return state, state

end

hsodm_stopping_criterion(tol, state::HSODMState) =
    (state.Δ <= 1e-20) || (state.ϵ <= tol) && abs(state.fz - state.fx) <= tol

function counting(iter::T, state::S) where {T<:HSODMIteration,S<:HSODMState}
    try
        state.kf = getfield(iter.f, :counter)
        state.kg = getfield(iter.g, :counter)
        state.kH = hasproperty(iter.H, :counter) ? getfield(iter.H, :counter) : 0
        state.kh = hasproperty(iter.hvp, :counter) ? getfield(iter.hvp, :counter) : 0
    catch
    end
end


function hsodm_display(k, state::HSODMState)
    if k == 1 || mod(k, 30) == 0
        @printf("%s", HSODM_LOG_SLOTS)
    end
    @printf("%5d | %+.3e | %.2e | %.2e | %.2e | %+.0e | %+.0e | %.0e | %+.0e | %6.1f |\n",
        k, state.fx, state.α, state.Δ, state.ϵ, state.λ₁, state.δ, state.k₂, state.ρ, state.t
    )
end

default_solution(::HSODMIteration, state::HSODMState) = state.x




HomogeneousSecondOrderDescentMethod(;
    name=:HSODM,
    stop=hsodm_stopping_criterion,
    display=hsodm_display
) = IterativeAlgorithm(HSODMIteration, HSODMState; name=name, stop=stop, display=display)



####################################################################################################
# HSODM
####################################################################################################
function (alg::IterativeAlgorithm{T,S})(;
    maxiter=10000,
    maxtime=1e2,
    tol=1e-6,
    freq=10,
    verbose=1,
    direction=:cold,
    linesearch=:hagerzhang,
    adaptive=:none,
    kwargs...
) where {T<:HSODMIteration,S<:HSODMState}

    arr = Vector{S}()
    kwds = Dict(kwargs...)

    for cf ∈ [:f :g :H :hvp]
        apply_counter(cf, kwds)
    end
    iter = T(; linesearch=linesearch, adaptive=adaptive, direction=direction, verbose=verbose, kwds...)
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


function Base.show(io::IO, t::T) where {T<:HSODMIteration}
    format_header(t.LOG_SLOTS)
    @printf io " algorithm alias       := %s\n" t.ALIAS
    @printf io " algorithm description := %s\n" t.DESC
    @printf io " inner iteration limit := %s\n" t.itermax
    @printf io " line-search algorithm := %s\n" t.linesearch
    @printf io "    adaptive  strategy := %s\n" t.adaptive
    if t.hvp !== nothing
        @printf io "     second-order info := using provided Hessian-vector product\n"
    elseif t.H !== nothing
        @printf io "     second-order info := using provided Hessian matrix\n"
    else
        @printf io " unknown mode to compute Hessian info\n"
        throw(ErrorException("unknown differentiation mode\n"))
    end
    println(io, "-"^length(t.LOG_SLOTS))
    flush(io)
end

function summarize(io::IO, k::Int, t::T, s::S) where {T<:HSODMIteration,S<:HSODMState}
    println(io, "-"^length(t.LOG_SLOTS))
    println(io, "summary:")
    @printf io " (main)          f       := %.2e\n" s.fx
    @printf io " (first-order)  |g|      := %.2e\n" s.ϵ
    println(io, "oracle calls:")
    @printf io " (main)          k       := %d  \n" k
    @printf io " (function)      f       := %d  \n" s.kf
    @printf io " (first-order)   g(+hvp) := %d  \n" s.kg + s.kh
    @printf io " (first-order)  hvp      := %d  \n" s.kh
    @printf io " (second-order)  H       := %d  \n" s.kH
    @printf io " (sub-problem)   P       := %d  \n" s.k₂
    @printf io " (sub-problem)   kₕ       := %d  \n" s.kᵥ
    @printf io " (running time)  t       := %.3f  \n" s.t
    println(io, "-"^length(t.LOG_SLOTS))
    flush(io)
end

summarize(k::Int, t::T, s::S) where {T<:HSODMIteration,S<:HSODMState} =
    summarize(stdout, k, t, s)