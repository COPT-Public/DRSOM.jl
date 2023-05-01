
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
    name::Union{Symbol,String}
    state::S
    trajectory::Vector{S}
    iter::I = nothing
end

function Base.show(io::IO, r::Result{I,S,Int}) where {I,S,Int}
    println(io, "Result<$(r.name)>")
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