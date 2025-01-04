__precompile__(true)

module ATRS

using LinearAlgebra
using LinearMaps
using IterativeSolvers
using Arpack
using Polynomials

include("eigenproblems.jl")
include("trust_region_boundary.jl")
include("trust_region.jl")
include("constraints.jl")
include("trust_region_small.jl")
atrs = trs
export atrs, trs_boundary, trs_small, trs_boundary_small

end