
# DRSOMEx with more than gradient and momentum
#   a new interface

using Base.Iterators
using LinearAlgebra
using Printf
using Dates



const DRSOMEx_LOG_SLOTS = @sprintf(
    "%5s | %10s | %8s | %8s | %7s | %7s | %7s | %6s | %6s | %6s \n",
    "k", "f", "α₁", "α₂", "Δ", "|∇f|", "λ", "ρ", "kₜ", "t",
)
@doc raw"""
Iteration object for DRSOM, to initialize an iterator, 
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
Base.@kwdef mutable struct DRSOMExIteration{Tx,Tf,Tϕ,Tr,Tg,Th,Te}
    x0::Tx            # initial point
    f::Tf             # f: smooth function
    ϕ::Tϕ = nothing   # ϕ: nonsmooth part (not implemented yet)
    g::Tg = nothing   # gradient function
    ga::Tr = nothing  # gradient function via forward or backward diff
    hvp::Th = nothing # hvp function via forward or backward diff
    H::Te = nothing   # hessian function
    t::Dates.DateTime = Dates.now()
    dimmax::Int64 = 5
    itermax::Int64 = 20
    fog = :forward
    sog = :forward
    LOG_SLOTS::String = DRSOMEx_LOG_SLOTS
    ALIAS::String = "DRSOM"
    DESC::String = "DRSOM with (2 ≤ d ≤ $dimmax) directions"
    error::Union{Nothing,Exception} = nothing
end

Base.IteratorSize(::Type{<:DRSOMExIteration}) = Base.IsInfinite()

@doc raw"""
Struct for DRSOMEx to keep the iterate state,
    including the following attributes e.g.,

- `x` the current iterate, namely ``x``;
- `fx` the function value, namely ``f(x)``;
- `∇f` the gradient, namely ``\nabla f(x)``;
- `t` the running time;
- `ϵ` gradient norm ``\|\nabla f(x)\|``
"""
Base.@kwdef mutable struct DRSOMExState{R,Tx,Tq,Tc}
    x::Tx             # iterate
    fx::R             # new value f at x: x(k)
    fz::R             # old value f at z: x(k-1)
    ∇f::Tx            # gradient of f at x
    ∇fz::Tx           # gradient of f at z
    ∇hvp::Tx          # gradient buffer (for buffer use of hvps)
    y::Tx             # forward point
    z::Tx             # previous point
    d::Tx             # momentum/fixed-point diff at iterate (= x - z)
    Q::Tq             # Q for low-dimensional QP
    c::Tc             # c for low-dimensional QP
    G::Tq             # G for low-dimensional QP
    Δ::R              # trust-region radius
    α::Tx             # step sizes
    m::Int            # number of directions
    dq::R             # decrease of estimated quadratic model
    df::R             # decrease of the real function value
    ρ::R              # trs descrease ratio: ρ = df/dq
    ϵ::R              # eps: residual for gradient 
    γ::R = 1e-16      # scaling parameter γ for λ
    ψ::R = 1.0        # 1-D linear search iteration #. if there is only one direction  
    λ::R = 1e-16      # dual λ
    kₜ::Int = 1        # inner iterations for adjustment
    kf::Int = 0       # function evaluations
    kg::Int = 0       # gradient evaluations
    kh::Int = 0       # hvp      evaluations
    kH::Int = 0       # hessian  evaluations
    t::R = 0.0        # running time
    status = true
end

function construct_quadratic_model(
    iter::DRSOMExIteration,
    gf::V, hvp::V, z::V
) where {V<:VecOrMat}
    if iter.fog == :direct
        copyto!(gf, iter.g(z))
    else
        iter.ga(gf, z)
    end
    if iter.sog ∈ (:forward, :backward)
        ∇hvp = similar(hvp)
        iter.hvp(z, gf, hvp, ∇hvp, gf)
    elseif iter.sog == :hess
        copy!(hvp, iter.H(z) * gf)
    elseif iter.sog == :prov
        # means the hvp is provided,
        #   e.g. in the NLPModels
        iter.hvp(z, gf, hvp)
    else
        # wild treatment in the direct mode 
        #   for the first iteration
        hvp = LinearAlgebra.I * gf
    end
end

function Base.iterate(iter::DRSOMExIteration)
    # first iteration of DRSOMEx
    # use a line search algorithm to initialize
    iter.t = Dates.now()
    z = copy(iter.x0)
    fz = iter.f(z)
    gx = similar(z)
    hvp = similar(z)


    construct_quadratic_model(
        iter, gx, hvp, z
    )
    Q11 = gx' * hvp
    c1 = -gx' * gx

    # now use a LS to find the first iterate
    s = -copy(gx)
    α₁, fx, kₜ = HagerZhangLineSearch(iter, gx, fz, z, s)

    dq = -α₁ * Q11 * α₁ / 2 - α₁ * c1
    y = z - α₁ .* gx
    df = fz - fx
    ro = df / dq

    t = (Dates.now() - iter.t).value / 1e3
    d = y - z
    state = DRSOMExState(
        x=y,
        y=y,
        z=z,
        fx=fx,
        fz=fz,
        Q=Matrix{Float64}(undef, (1, 1)),
        G=Matrix{Float64}(undef, (1, 1)),
        c=[c1],
        ∇f=gx,
        ∇fz=z,
        ∇hvp=hvp,
        α=[α₁; 0.0],
        d=d,
        Δ=α₁,
        dq=dq,
        df=df,
        ρ=isnan(ro) ? -1e6 : ro,
        m=1,
        ϵ=norm(gx),
        γ=1e-6,
        λ=1e-6,
        kₜ=kₜ,
        t=t,
    )
    return state, state
end



function Base.iterate(iter::DRSOMExIteration, state::DRSOMExState)

    state.z = z = state.x
    state.fz = fz = state.fx
    state.∇fz = state.∇f
    state.m = 2
    # evaluate gradient
    copy!(state.∇f, iter.g(state.x))
    gₙ = state.ϵ = norm(state.∇f)
    dₙ = norm(state.d)

    if gₙ <= 1e-10
        return state, state
    end
    # start inner iterations
    kₜ = 1
    # ubiquitous buffer
    Hv = similar(z)
    V = [-state.∇f / gₙ state.d / dₙ]
    iter.hvp(state.x, V[:, 1], Hv)
    Hvs = copy(Hv)
    iter.hvp(state.x, V[:, 2], Hv)
    Hvs = [Hvs copy(Hv)]
    state.Q = V' * Hvs
    state.G = V' * V
    state.c = c = V' * state.∇f
    while true
        α = SimpleTrustRegionSubproblem(Symmetric(state.Q, :U), state.c, state; G=Symmetric(state.G), mode=:reg)
        x = y = state.z + V * α
        fx = iter.f(x)
        dq = -α' * state.Q * α / 2 - α' * state.c
        df = fz - fx
        ro = df / dq
        if (df < 0) || (ro <= 0.15)
            if (state.m < iter.dimmax)
                newd = copy(Hvs[:, end])
                newdₙ = norm(newd)
                newd .*= 1 / newdₙ
                iter.hvp(state.x, newd, Hv)
                V = [V newd]
                Hvs = [Hvs Hv]
                state.Q = V' * Hvs
                state.G = V' * V
                state.c = V' * state.∇f
                state.m += 1
                state.γ *= 2
            else
                state.γ *= 5
            end
        elseif ro >= 0.8
            state.γ = max(min(sqrt(state.γ), log(10, state.γ + 1)), 1e-16)
        end
        if (ro > 0.15 && df > 0) || kₜ == iter.itermax
            state.α = α
            state.x = x
            state.y = y
            state.fx = fx
            state.ρ = isnan(ro) ? -1e6 : ro
            state.dq = dq
            state.df = df
            state.d = x - z
            state.Δ = sqrt(α' * state.G * α)
            state.kₜ = kₜ
            state.ϵ = norm(state.∇f)
            state.t = (Dates.now() - iter.t).value / 1e3
            counting(iter, state)
            return state, state
        end
        kₜ += 1
    end

end

function counting(iter::T, state::S) where {T<:DRSOMExIteration,S<:DRSOMExState}
    state.kf = getfield(iter.f, :counter)
    state.kg = hasproperty(iter.g, :counter) ? getfield(iter.g, :counter) : getfield(iter.ga, :counter)
    state.kh = hasproperty(iter.hvp, :counter) ? getfield(iter.hvp, :counter) : 0
    state.kH = hasproperty(iter.H, :counter) ? getfield(iter.H, :counter) : 0
end

drsom_stopping_criterion(tol::Real, state::DRSOMExState) =
    (abs.(state.α) |> maximum <= 1e-20) || (state.ϵ <= tol) && abs(state.fz - state.fx) <= tol


function drsom_display(k::Int, state::DRSOMExState)
    if k == 1 || mod(k, 30) == 0
        @printf("%s", DRSOMEx_LOG_SLOTS)
    end
    @printf("%5d | %+.3e | %+.1e | %+.1e | %.1e | %.1e | %.1e | %+.0e | %2d(%2d) | %6.1f \n",
        k, state.fx, state.α[1:2]..., state.Δ, state.ϵ, state.λ, state.ρ, state.kₜ, state.m, state.t
    )
end

default_solution(::DRSOMExIteration, state::DRSOMExState) = state.x


DimensionReducedSecondOrderMethodEx(;
    name=:DRSOMEx,
    stop=drsom_stopping_criterion,
    display=drsom_display
) = IterativeAlgorithm(DRSOMExIteration, DRSOMExState; name=name, stop=stop, display=display)

# Aliases


####################################################################################################
# DRSOMEx
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
) where {T<:DRSOMExIteration,S<:DRSOMExState}

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


function Base.show(io::IO, t::T) where {T<:DRSOMExIteration}
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

function summarize(io::IO, k::Int, t::T, s::S) where {T<:DRSOMExIteration,S<:DRSOMExState}
    println(io, "-"^length(t.LOG_SLOTS))
    println(io, "summary:")
    @printf io " (main)          f       := %.2e\n" s.fx
    @printf io " (first-order)  |g|      := %.2e\n" s.ϵ
    println(io, "oracle calls:")
    @printf io " (main)          k       := %d  \n" k
    @printf io " (function)      f       := %d  \n" s.kf
    @printf io " (first-order)   g       := %d  \n" s.kg
    @printf io " (first-order)  hvp      := %d  \n" s.kh
    @printf io " (second-order)  h       := %d  \n" s.kH
    @printf io " (line-search)   ψ       := %d  \n" s.ψ
    @printf io " (running time)  t       := %.3f  \n" s.t
    println(io, "-"^length(t.LOG_SLOTS))
    flush(io)
end

summarize(k::Int, t::T, s::S) where {T<:DRSOMExIteration,S<:DRSOMExState} =
    summarize(stdout, k, t, s)
