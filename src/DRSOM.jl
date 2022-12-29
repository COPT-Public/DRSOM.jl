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

function (alg::IterativeAlgorithm{IteratorType,StateType})(
    maxit=10000,
    maxtime=1e2,
    tol=1e-6,
    freq=10,
    verbose=true;
    kwargs...
) where {IteratorType,StateType}
    arr = Vector{StateType}()
    kwds = Dict(kwargs...)
    for cf ∈ [:f :g :H :rh]
        apply_counter(cf, kwds)
    end
    iter = IteratorType(; kwds...)
    for (k, state) in enumerate(iter)
        push!(arr, copy(state))
        if k >= maxit || state.t >= maxtime || alg.stop(tol, state)
            verbose && alg.display(k, state)
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

export DRSOM2
end # module
