module DRSOM


using Printf

const RealOrComplex{R} = Union{R,Complex{R}}
const Maybe{T} = Union{T,Nothing}

# various utilities
include("utilities/ad.jl")
include("utilities/fb_tools.jl")
include("utilities/iteration_tools.jl")
include("utilities/display_tools.jl")
include("utilities/trs.jl")
include("utilities/counter.jl")
include("utilities/interpolation.jl")

# supplement copy
Base.copy(x::T) where {T} = T([deepcopy(getfield(x, k)) for k ∈ fieldnames(T)]...)

# algorithm interface
Base.@kwdef mutable struct IterativeAlgorithm{I,S,H,D}
    stop::H
    display::D
    name::Symbol = :DRSOM
end

Base.@kwdef mutable struct Result{I,S,Int}
    k::Int
    name::Symbol
    state::S
    trajectory::Vector{S}
    iter::I
end

IterativeAlgorithm(T, S; name, stop, display) =
    IterativeAlgorithm{T,S,typeof(stop),typeof(display)}(
        name=name, stop=stop, display=display
    )

function apply_counter(cf, kwds)
    va = get(kwds, cf, nothing)
    if va !== nothing
        kwds[cf] = Counting(va)
    end
end

function (alg::IterativeAlgorithm{IteratorType,StateType})(;
    maxit=10000,
    maxtime=1e2,
    tol=1e-6,
    freq=10,
    verbose=true,
    fog=:backward,
    sog=:direct,
    kwargs...
) where {IteratorType,StateType}

    arr = Vector{StateType}()
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

    for cf ∈ [:f :g :H :ga :hvp]
        apply_counter(cf, kwds)
    end
    iter = IteratorType(; fog=fog, sog=sog, kwds...)
    verbose && show(iter)
    for (k, state) in enumerate(iter)
        push!(arr, copy(state))
        if k >= maxit || state.t >= maxtime || alg.stop(tol, state)
            verbose && alg.display(k, state)
            verbose && summarize(k, iter, state)
            return Result(name=alg.name, iter=iter, state=state, k=k, trajectory=arr)
        end
        verbose && (k == 1 || mod(k, freq) == 0) && alg.display(k, state)
    end

end

# algorithm implementations
include("algorithms/drsom_legacy.jl")
include("algorithms/drsom.jl")
include("algorithms/drsom_plus.jl")
include("algorithms/drsom_l.jl")
include("algorithms/drsom_c.jl")
include("algorithms/drsom_f.jl")
include("algorithms/hsodm.jl")

# Algorithm Aliases
DRSOM2 = DimensionReducedSecondOrderMethod


function Base.show(io::IO, t::T) where {T<:DRSOMIteration}
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
    else
        throw(ErrorException("unknown differentiation mode\n"))
    end
    println(io, "-"^length(t.LOG_SLOTS))
    flush(io)
end

function summarize(io::IO, k::Int, t::T, s::S) where {T<:DRSOMIteration,S<:DRSOMState}
    println(io, "-"^length(t.LOG_SLOTS))
    println(io, "summary:")
    @printf io " (main)          f       := %.2e\n" s.fx
    @printf io " (first-order)  |g|      := %.2e\n" s.ϵ
    println(io, "oracle calls:")
    @printf io " (main)          k       := %d  \n" k
    @printf io " (function)      f       := %d  \n" s.kf
    @printf io " (first-order)   g(+hvp) := %d  \n" s.kg
    @printf io " (first-order)  hvp      := %d  \n" s.kh
    @printf io " (second-order)  H       := %d  \n" s.kH
    @printf io " (line-search)   ψ       := %d  \n" s.ψ
    @printf io " (running time)  t       := %.3f  \n" s.t
    println(io, "-"^length(t.LOG_SLOTS))
    flush(io)
end

summarize(k::Int, t::T, s::S) where {T<:DRSOMIteration,S<:DRSOMState} =
    summarize(stdout, k, t, s)
export DRSOM2
end # module
