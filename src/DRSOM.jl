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
include("utilities/interpolation.jl")

# algorithm interface
mutable struct IterativeAlgorithm{IteratorType,H,S,D,K}
    maxit::Int
    stop::H
    solution::S
    verbose::Bool
    freq::Int
    display::D
    kwargs::K
end

"""
# this part borrows from ProximalAlgorithms
    IterativeAlgorithm(T; maxit, stop, solution, verbose, freq, display, kwargs...)

Wrapper for an iterator type `T`, adding termination and verbosity options on top of it.

This is a conveniency constructor to allow for "partial" instantiation of an iterator of type `T`.
The resulting "algorithm" object `alg` can be called on a set of keyword arguments, which will be merged
to `kwargs` and passed on to `T` to construct an iterator which will be looped over.
Specifically, if an algorithm is constructed as

    alg = IterativeAlgorithm(T; maxit, stop, solution, verbose, freq, display, kwargs...)

then calling it with

    alg(; more_kwargs...)

will internally loop over an iterator constructed as

    T(; alg.kwargs..., more_kwargs...)

# Note
This constructor is not meant to be used directly: instead, algorithm-specific constructors
should be defined on top of it and exposed to the user, that set appropriate default functions
for `stop`, `solution`, `display`.

# Arguments
* `T::Type`: iterator type to use
* `maxit::Int`: maximum number of iteration
* `stop::Function`: termination condition, `stop(::T, state)` should return `true` when to stop the iteration
* `solution::Function`: solution mapping, `solution(::T, state)` should return the identified solution
* `verbose::Bool`: whether the algorithm state should be displayed
* `freq::Int`: every how many iterations to display the algorithm state
* `display::Function`: display function, `display(::Int, ::T, state)` should display a summary of the iteration state
* `kwargs...`: keyword arguments to pass on to `T` when constructing the iterator
"""
IterativeAlgorithm(T; maxit, stop, solution, verbose, freq, display, kwargs...) =
    IterativeAlgorithm{T,typeof(stop),typeof(solution),typeof(display),typeof(kwargs)}(
        maxit, stop, solution, verbose, freq, display, kwargs
    )

function (alg::IterativeAlgorithm{IteratorType})(; kwargs...) where {IteratorType}
    res = []
    iter = IteratorType(; alg.kwargs..., kwargs...)
    for (k, state) in enumerate(iter)
        push!(res, norm(state.res, Inf))
        if k >= alg.maxit || alg.stop(iter, state)
            alg.verbose && alg.display(k, iter, state)
            return state, res, k
        end
        alg.verbose && mod(k, alg.freq) == 0 && alg.display(k, iter, state)
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

end # module
