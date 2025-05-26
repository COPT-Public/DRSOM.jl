####################################################################################################
# Forward-Backward Envelope Iteration,
# preliminary implementation using DRSOM-type interface
#
# @reference: 
# --------------------------------------------------------------------------------------------------
# [1].  Stella, L., Themelis, A., Sopasakis, P., Patrinos, P.: 
#   A simple and efficient algorithm for nonlinear model predictive control. 
#   In: 2017 IEEE 56th Annual Conference on Decision and Control (CDC). pp. 1939–1944. IEEE (2017)
# [2]  Stella, L., Themelis, A., Patrinos, P.: 
#   Forward–backward quasi-Newton methods for nonsmooth optimization problems. 
#   Computational Optimization and Applications. 67, 443–487 (2017)
# [3].  Themelis, A., Stella, L., Patrinos, P.: 
#   Forward-Backward Envelope for the Sum of Two Nonconvex Functions: Further Properties and Nonmonotone Linesearch Algorithms. 
#   SIAM J. Optim. 28, 2274–2303 (2018). https://doi.org/10.1137/16M1080240
####################################################################################################

using Base.Iterators
using LinearAlgebra
using Printf
using Dates



const FBEDRSOM_LOG_SLOTS = @sprintf(
    "%5s | %10s | %8s | %8s | %8s | %7s | %2s | %2s | %6s \n",
    "k", "f + h", "ℓ", "|∇f|", "|r|", "Δ", "kₗ", "kₜ", "t",
)
@doc raw"""
Iteration object for FBEDRSOM, to initialize an iterator, 
    you must specify the attributes, including 

- ``x_0`` is initial point of the iterate                  
- `f` is the smooth function to minimize 
- `ϕ` is the nonsmooth function          
- `g` is the gradient function           
- `ga` is gradient function via forward or backward diff 
- `hvp` is hvp function via forward or backward diff
- `H` is hessian function 

rest of the attributes have default options:
```julia
    t::Dates.DateTime = Dates.now()
    itermax::Int64 = 20
    fog = :forward
    sog = :forward
```
"""
Base.@kwdef mutable struct FBEDRSOMIteration{Tx,Tf,Th,Tr,Tg,Thvp,Te}
    x0::Tx              # initial point
    f::Tf               # f: smooth function
    h::Th = nothing     # h: nonsmooth part (not implemented yet)
    g::Tg = nothing     # gradient function
    ga::Tr = nothing    # gradient function via forward or backward diff
    hvp::Thvp = nothing # hvp function via forward or backward diff
    H::Te = nothing     # hessian function
    ######################################################################
    t::Dates.DateTime = Dates.now()
    itermax::Int64 = 20
    fog = :forward
    sog = :forward
    LOG_SLOTS::String = FBEDRSOM_LOG_SLOTS
    ALIAS::String = "FBEDRSOM2"
    DESC::String = "FBEDRSOM with gradient and momentum"
    error::Union{Nothing,Exception} = nothing
end

Base.IteratorSize(::Type{<:FBEDRSOMIteration}) = Base.IsInfinite()

@doc raw"""
Struct for FBEDRSOM to keep the iterate state,
    including the following attributes e.g.,

- `x` the current iterate, namely ``x``;
- `fx` the function value, namely ``f(x)``;
- `∇f` the gradient, namely ``\nabla f(x)``;
- `t` the running time;
- `ϵ` gradient norm ``\|\nabla f(x)\|``
"""
Base.@kwdef mutable struct FBEDRSOMState{R,Tx,Tq,Tc}
    # --------------------------------------------------------------------
    x::Tx             # iterate
    fx::R             # new value f at x: x(k)
    hx::R             # new value h at x: x(k)
    fz::R             # old value f at z: x(k-1)
    ∇f::Tx            # gradient of f at x
    ∇fz::Tx           # gradient of f at z
    ∇hvp::Tx          # gradient buffer (for buffer use of hvps)
    z::Tx             # previous point
    d::Tx             # momentum/fixed-point diff at iterate (= x - z)
    Q::Tq             # Q for low-dimensional QP
    c::Tc             # c for low-dimensional QP
    G::Tq             # c for low-dimensional QP
    α₁::R = 1e6       # stepsize 1 parameter of gradient
    α₂::R = 1e6       # stepsize 2 parameter of momentum
    Δ::R = 1e6        # trust-region radius
    ϵ::R = 1e6        # eps: residual for gradient 
    ρ::R = 1.0        # trs descrease ratio: ρ = dϕ/dq
    γ::R = 1e-16      # scaling parameter γ for λ
    y::Tx             # forward point
    ψ::R = 1.0        # 1-D linear search iteration #. if there is only one direction  
    λ::R = 1e-16      # dual λ
    # --------------------------------------------------------------------
    # FBE attributes
    # --------------------------------------------------------------------
    dq::R = 1e6       # decrease of estimated quadratic model
    dϕ::R = 1e6       # decrease of the real function value
    T::Tx             # T point in [2] MINFBE: backward point of y
    r::Tx             # R point in [2] MINFBE: residual
    ℓ::R = 1e-4       # scaling parameter for FBE, correspond to γ in [2]
    ϕ::R = 1e6        # ϕ: the envelope value
    β::R = 0.05       # see algorithm 1 in [2]
    ∇ϕ::Tx            # gradient of ϕ at x
    ϵᵣ::R = 1e6       # residual for r = γ^{-1}(x - T(x))
    dtp::Symbol = :d  # direction type
    stk::Bool = false # direction type
    # --------------------------------------------------------------------
    kₜ::Int = 1       # inner iterations for adjustment α
    kₗ::Int = 1       # inner iterations for adjustment ℓ
    kf::Int = 0       # function evaluations
    kg::Int = 0       # gradient evaluations
    kh::Int = 0       # hvp      evaluations
    kH::Int = 0       # hessian  evaluations
    t::R = 0.0        # running time
    status = true
end

@doc """
    Construct the quadratic model for FBEDRSOM
    the quad model is computed to ϕ (not f along)
- the full-dimension mode computes:
    x₊ - x = - B^{-1} ∇ϕ(x)
    where B = ∇²ϕ(x) and ϕ is the FB envelope function
- so we compute a DRSOM analogue at subspace    
    x₊ - x ∈ span{∇ϕ(x), d} 
    where d is the momentum direction
"""
function construct_quadratic_model(
    iter::FBEDRSOMIteration, state::FBEDRSOMState
)
    # note no matter what oracle of f is provided, 
    #   we always compute the quadratic model of ϕ
    #   based on interpolation.
    if getfield(iter, :hvp) |> isnothing
        throw(ErrorException("hvp function must be provided"))
    end
    gₙ, dₙ, Q, c, G = directional_interpolation_fbe(iter, state; ϵₚ=max(state.ℓ, 1e-4), diag=true)

    return gₙ, dₙ, Q, c, G
end

function Base.iterate(iter::FBEDRSOMIteration)
    # first iteration of FBEDRSOM
    # use a line search algorithm to initialize
    iter.t = Dates.now()
    z = copy(iter.x0)
    y = copy(iter.x0)
    fz = iter.f(z)
    gx = iter.g(z)
    ∇ϕ = similar(z)
    hvp = similar(z)
    T = similar(z)
    r = similar(z)
    d = ones(Float64, length(z))
    t = (Dates.now() - iter.t).value / 1e3
    state = FBEDRSOMState(
        x=y,
        y=y,
        z=z,
        fx=fz,
        fz=fz,
        hx=iter.h(z),
        Q=Matrix{Float64}(undef, (2, 2)),
        G=Matrix{Float64}(undef, (2, 2)),
        c=zeros(2),
        ∇f=gx,
        ∇fz=z,
        ∇hvp=hvp,
        ∇ϕ=∇ϕ,
        T=T,
        r=r,
        d=d,
        ϵ=norm(gx),
        γ=1e-6,
        ℓ=5e+1,
        λ=1e-6,
        kₜ=1,
        t=t,
    )
    return state, state
end



function Base.iterate(iter::FBEDRSOMIteration, state::FBEDRSOMState)

    state.z = z = copy(state.x)
    state.fz = fz = state.fx
    state.∇fz = iter.g(z)


    # start inner iterations
    kₜ = 1
    kₗ = 1

    while true
        # find appropriate step
        # !!! reset x to z
        x = copy(z)
        ϕ = (x, β) -> ϕᵧ(iter, state.ℓ, x; β=β)
        gₙ, dₙ, Q, c, G = construct_quadratic_model(iter, state)
        ϕx = ϕ(z, 0.0)
        state.γ = 1e1
        while true
            # α₁, α₂ = 0.0, 0.0
            # y = x
            # state.dtp = :p
            α₁, α₂ = SimpleTrustRegionSubproblem(Q, c, state; G=state.G, mode=:free)
            y = x - α₁ .* state.∇ϕ / gₙ + α₂ .* state.d / dₙ
            state.dtp = :d
            if (isnan.(y) |> any)
                α₁, α₂ = 0.0, 0.0
                y = x
                state.dtp = :p
            end
            ϕy = ϕ(y, 0.0)
            α = [α₁; α₂]
            dq = -α' * Q * α / 2 - α' * c
            dϕ = ϕx - ϕy
            ro = dϕ / dq
            if (dϕ < 0) || (isnan(ϕy))
                state.γ *= 5
            end
            # @info "$α₁ $α₂ $ϕx $ϕy $dϕ $dq $ro"
            if ((dϕ >= 0)
                || (kₜ == iter.itermax)
                || (state.dtp == :p))
                state.α₁ = α₁
                state.α₂ = α₂
                # @note: save the trial step
                state.y = y
                state.ρ = isnan(ro) ? -1e6 : ro
                state.dq = dq
                state.dϕ = dϕ
                state.kₜ = kₜ
                state.ϵ = norm(state.∇f)
                state.t = (Dates.now() - iter.t).value / 1e3
                break
            end
            kₜ += 1

        end
        # adjusting γ
        # evaluate original step x at γ
        ∇, y, T, r = compute_T_R(iter, state.ℓ, x)
        Δf = state.fz - state.ℓ * dot(state.∇fz, r) + 0.5 * state.ℓ * (1 - state.β) * norm(r)^2
        # evaluate forward step y at γ
        ∇, y, T, r = compute_T_R(iter, state.ℓ, state.y)
        # compute T(y)
        fT = iter.f(T)
        epsf = fT - Δf
        if epsf >= 0
            state.ℓ = max(state.ℓ * 0.4, 1e-8)
            kₗ += 1
        else
            state.ℓ = min(state.ℓ * 1.2, 1e4)
        end
        if (epsf < 0) || (kₗ >= 20)
            state.∇f = iter.g(T)
            state.ϵ = norm(state.∇f)
            state.x = T
            state.fx = fT
            state.hx = iter.h(T)
            state.d = T - z
            state.ϵᵣ = r |> norm
            state.Δ = state.d |> norm
            state.kₗ = kₗ
            state.stk = (kₗ >= 20)
            counting(iter, state)
            return state, state
        else
        end
    end

end

function counting(iter::T, state::S) where {T<:FBEDRSOMIteration,S<:FBEDRSOMState}
    state.kf = getfield(iter.f, :counter)
    state.kg = hasproperty(iter.g, :counter) ? getfield(iter.g, :counter) : getfield(iter.ga, :counter)
    state.kh = hasproperty(iter.hvp, :counter) ? getfield(iter.hvp, :counter) : 0
    state.kH = hasproperty(iter.H, :counter) ? getfield(iter.H, :counter) : 0
end

drsom_stopping_criterion(tol::Real, state::FBEDRSOMState) =
    (state.stk ||
     ((state.ϵᵣ <= tol) && abs(state.fz - state.fx) <= tol))

function drsom_display(k::Int, state::FBEDRSOMState)
    if k == 1 || mod(k, 30) == 0
        @printf("%s", FBEDRSOM_LOG_SLOTS)
    end
    _s = @sprintf("%5d | %+.3e | %+.1e | %+.1e | %+.1e | %.1e | %2d | %2d | %6.1f",
        k, state.fx + state.hx, state.ℓ, state.ϵ, state.ϵᵣ, state.Δ, state.kₗ, state.kₜ, state.t
    )
    @printf "%s [%s]\n" _s state.dtp
end

default_solution(::FBEDRSOMIteration, state::FBEDRSOMState) = state.x


ForwardBackwardDimensionReducedSecondOrderMethod(;
    name=:FBEDRSOM,
    stop=drsom_stopping_criterion,
    display=drsom_display
) = IterativeAlgorithm(FBEDRSOMIteration, FBEDRSOMState; name=name, stop=stop, display=display)

# Aliases


####################################################################################################
# FBEDRSOM
####################################################################################################
function (alg::IterativeAlgorithm{T,S})(;
    maxiter=10000,
    maxtime=1e2,
    tol=1e-6,
    freq=10,
    verbose=true,
    fog=:backward,
    sog=:direct,
    kwargs...
) where {T<:FBEDRSOMIteration,S<:FBEDRSOMState}

    arr = Vector{S}()
    kwds = Dict(kwargs...)
    x0 = get(kwds, :x0, nothing)
    x0 === nothing && throw(ErrorException("an initial point must be provided"))
    f = get(kwds, :f, nothing)
    f === nothing && throw(ErrorException("target function f must be provided"))
    if get(kwds, :g, nothing) !== nothing
        fog = :direct
    elseif fog == :forward
        cfg = ForwardDiff.GradientConfig(f, x0, ForwardDiff.Chunk(x0))
        gf(g_buffer, x) = ForwardDiff.gradient!(g_buffer, f, x, cfg)
        kwds[:ga] = gf
        if sog == :forward
            hvpf(x, v, hvp, ∇hvp, ∇f) = hessfa(f, x, v, hvp, ∇hvp, ∇f; cfg=cfg)
            kwds[:hvp] = hvpf
        end
    elseif fog == :backward
        f_tape = ReverseDiff.GradientTape(f, x0)
        f_tape_compiled = ReverseDiff.compile(f_tape)
        gb(g_buffer, x) = ReverseDiff.gradient!(g_buffer, f_tape_compiled, x)
        kwds[:ga] = gb
        if sog == :backward
            hvpb(x, v, hvp, ∇hvp, ∇f) = hessba(x, v, hvp, ∇hvp, ∇f; tp=f_tape_compiled)
            kwds[:hvp] = hvpb
        end
    else
        throw(ErrorException("""function g must be provided, you must specify g directly
         or a correct first-order oracle mode via keyword :fog"""))
    end
    # 
    if sog == :prov
        hvp = get(kwds, :hvp, nothing)
        hvp === nothing && throw(ErrorException("hvp function must be provided if you choose sog==:prov"))
        kwds[:hvp] = hvp
    end

    for cf ∈ [:f :g :H :ga :hvp]
        apply_counter(cf, kwds)
    end
    iter = T(; fog=fog, sog=sog, kwds...)
    verbose && show(iter)
    for (k, state) in enumerate(iter)

        push!(arr, copy(state))
        if k >= maxiter || state.t >= maxtime || alg.stop(tol, state)
            verbose && alg.display(k, state)
            verbose && summarize(k, iter, state)
            return Result(name=alg.name, iter=iter, state=state, k=k, trajectory=arr)
        end
        verbose && (k == 1 || mod(k, freq) == 0) && alg.display(k, state)
    end
end


function Base.show(io::IO, t::T) where {T<:FBEDRSOMIteration}
    format_header(t.LOG_SLOTS)
    @printf io " algorithm alias       := %s\n" t.ALIAS
    @printf io " algorithm description := %s\n" t.DESC
    @printf io " inner iteration limit := %s\n" t.itermax
    if t.g !== nothing
        @printf io " oracle (first-order)  := %s => %s\n" t.fog "provided"
    else
        @printf io " oracle (first-order)  := %s\n" t.fog
    end
    @printf io " oracle (second-order) := %s => " t.sog
    if t.sog ∈ (:forward, :backward)
        @printf io "use forward diff or backward tape\n"
    elseif t.sog == :direct
        @printf io "use interpolation\n"
    elseif t.sog == :hess
        @printf io "use provided Hessian\n"
    elseif t.sog == :hvp
        @printf io "use provided Hessian-vector product\n"
    else
        throw(ErrorException("unknown differentiation mode\n"))
    end
    println(io, "-"^length(t.LOG_SLOTS))
    flush(io)
end

function summarize(io::IO, k::Int, t::T, s::S) where {T<:FBEDRSOMIteration,S<:FBEDRSOMState}
    println(io, "-"^length(t.LOG_SLOTS))
    println(io, "summary:")
    @printf io " (main)          f + h      := %.2e\n" s.fx + s.hx
    @printf io " (first-order)  |∇f|        := %.2e\n" s.ϵ
    @printf io " (first-order)  |∇f + ∂h|   := %.2e\n" s.ϵᵣ

    println(io, "oracle calls:")
    @printf io " (main)          k          := %d  \n" k
    @printf io " (function)      f          := %d  \n" s.kf
    @printf io " (first-order)   g          := %d  \n" s.kg
    @printf io " (first-order)  hvp         := %d  \n" s.kh
    @printf io " (second-order)  h          := %d  \n" s.kH
    @printf io " (line-search)   ψ          := %d  \n" s.ψ
    @printf io " (running time)  t          := %.3f  \n" s.t
    println(io, "-"^length(t.LOG_SLOTS))
    flush(io)
end

summarize(k::Int, t::T, s::S) where {T<:FBEDRSOMIteration,S<:FBEDRSOMState} =
    summarize(stdout, k, t, s)
