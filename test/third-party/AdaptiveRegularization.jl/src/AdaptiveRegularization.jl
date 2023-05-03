module AdaptiveRegularization

# stdlib
using LinearAlgebra, SparseArrays
# JSO
using Krylov, LinearOperators, NLPModels, SparseMatricesCOO, SolverCore
# Stopping
using Stopping, StoppingInterface

# Selective includes.
include("./utils/hessian_rep.jl")
include("./utils/pdata_struct.jl")
include("./utils/utils.jl")
include("./utils/increase_decrease.jl")

path = joinpath(dirname(@__FILE__), "SolveModel")
files = filter(x -> x[(end - 2):end] == ".jl", readdir(path))
for file in files
  include("SolveModel/" * file)
end

path = joinpath(dirname(@__FILE__), "PreProcess")
files = filter(x -> x[(end - 2):end] == ".jl", readdir(path))
for file in files
  include("PreProcess/" * file)
end

include("main.jl")

include("solvers.jl")

export ALL_solvers, NLS_solvers

ALL_solvers = keys(solvers_const)
NLS_solvers = keys(solvers_nls_const)

export TRARC

"""
    TRARC(nlp; kwargs...)

Compute a local minimum of an unconstrained optimization problem using trust-region (TR)/adaptive regularization with cubics (ARC) methods.
# Arguments
- `nlp::AbstractNLPModel`: the model solved, see `NLPModels.jl`.
The keyword arguments include
- `TR::TrustRegion`: structure with trust-region/ARC parameters, see [`TrustRegion`](@ref). Default: `TrustRegion(T(10.0))`.
- `hess_type::Type{Hess}`: Structure used to handle the hessian. The possible values are: `HessDense`, `HessSparse`, `HessSparseCOO`, `HessOp`. Default: `HessOp`.
- `pdata_type::Type{ParamData}` Structure used for the preprocessing step. Default: `PDataKARC`.
- `solve_model::Function` Function used to solve the subproblem. Default: `solve_modelKARC`.
- `robust::Bool`: `true` implements a robust evaluation of the model. Default: `true`.
- `verbose::Bool`: `true` prints iteration information. Default: `false`.
Additional `kwargs` are used for stopping criterion, see `Stopping.jl`.

# Output
The returned value is a `GenericExecutionStats`, see `SolverCore.jl`.

This implementation uses `Stopping.jl`. Therefore, it is also possible to used
        
    TRARC(stp; kwargs...)

which returns the `stp::NLPStopping` updated.

For advanced usage, the principal call to the solver uses a [`TRARCWorkspace`](@ref).

    TRARC(stp, pdata, workspace, trust_region_parameters; kwargs...)

Some variants of TRARC are already implemented and listed in `AdaptiveRegularization.ALL_solvers`.

# References
This method unifies the implementation of trust-region and adaptive regularization with cubics as described in

    Dussault, J.-P. (2020).
    A unified efficient implementation of trust-region type algorithms for unconstrained optimization.
    INFOR: Information Systems and Operational Research, 58(2), 290-309.
    10.1080/03155986.2019.1624490

# Examples

```jldoctest
using AdaptiveRegularization, ADNLPModels
nlp = ADNLPModel(x -> 100 * (x[2] - x[1]^2)^2 + (x[1] - 1)^2, [-1.2; 1.0]);
stats = TRARC(nlp)

# output

"Execution stats: first-order stationary"
```
"""
function TRARC end

function TRARC(nlp::AbstractNLPModel{T, S}; kwargs...) where {T, S}
  nlpstop = NLPStopping(nlp; optimality_check = (pb, state) -> norm(state.gx), kwargs...)
  nlpstop = TRARC(nlpstop; kwargs...)
  return stopping_to_stats(nlpstop)
end

for fun in union(keys(solvers_const), keys(solvers_nls_const))
  ht, pt, sm, ka = merge(solvers_const, solvers_nls_const)[fun]
  @eval begin
    function $fun(nlpstop::NLPStopping; kwargs...)
      kw_list = Dict{Symbol, Any}()
      if $ka != ()
        for t in $ka
          push!(kw_list, t)
        end
      end
      merge!(kw_list, Dict(kwargs))
      TRARC(nlpstop; hess_type = $ht, pdata_type = $pt, solve_model = $sm, kw_list...)
    end
  end
  @eval begin
    function $fun(nlp::AbstractNLPModel{T, S}; kwargs...) where {T, S}
      nlpstop = NLPStopping(nlp; optimality_check = (pb, state) -> norm(state.gx), kwargs...)
      nlpstop = $fun(nlpstop; kwargs...)
      return stopping_to_stats(nlpstop)
    end
  end
  @eval export $fun
end

end # module
