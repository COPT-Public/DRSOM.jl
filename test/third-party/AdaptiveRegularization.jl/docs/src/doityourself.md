# Your own way

`AdaptiveRegularization.jl` implements an unified algorithm for trust-region methods and adaptive regularization with cubics.
This package implements by default some variants, but anyone can design its own and benchmark it against existing ones.

```@example 1
using AdaptiveRegularization, Krylov
```

The implemented variants are accessible here:
```@example 1
AdaptiveRegularization.ALL_solvers
```

To make your own variant we need to implement:
- A new data structure `<: PData{T}` for some real number type `T`.
- A `preprocess(PData::TPData, H, g, gNorm2, α)` function called before each trust-region iteration.
- A `solve_model(PData::PDataST, H, g, gNorm2, n1, n2, δ::T)` function used to solve the algorithm subproblem.

In the rest of this tutorial, we implement a Steihaug-Toint trust-region method using `cg_lanczos` from [`Krylov.jl`](https://github.com/JuliaSmoothOptimizers/Krylov.jl) to solve the linear subproblem with trust-region constraint.

```@example 1
mutable struct PDataST{S,T} <: AdaptiveRegularization.TPData{T}
    d::S                      # Mandatory: solution of the subproblem
    λ::T                      # Mandatory
    ζ::T                      # Inexact Newton order parameter: stop when ||∇q|| < ξ * ||g||^(1+ζ)
    ξ::T                      # Inexact Newton order parameter: stop when ||∇q|| < ξ * ||g||^(1+ζ)
    maxtol::T                 # Largest tolerance for Inexact Newton
    mintol::T                 # Smallest tolerance for Inexact Newton
    cgatol                    # Absolute tolerance for `cg_lanczos`
    cgrtol                    # Relative tolerance for `cg_lanczos`

    OK::Bool                  # Mandatory: preprocess success
    solver::CgSolver          # Memory pre-allocation for `cg_lanczos`
end
```
The `TPData` stuctures have a unified constructor with `(::Type{S}, ::Type{T}, n)` as arguments.
```@example 1
function PDataST(
    ::Type{S},
    ::Type{T},
    n;
    ζ = T(0.5),
    ξ = T(0.01),
    maxtol = T(0.01),
    mintol = T(1.0e-8),
    cgatol = (ζ, ξ, maxtol, mintol, gNorm2) -> max(mintol, min(maxtol, ξ * gNorm2^(1 + ζ))),
    cgrtol = (ζ, ξ, maxtol, mintol, gNorm2) -> max(mintol, min(maxtol, ξ * gNorm2^ζ)),
    kwargs...,
) where {S,T}
    d = S(undef, n)
    λ = zero(T)
    OK = true
    solver = CgSolver(n, n, S)
    return PDataST(d, λ, ζ, ξ, maxtol, mintol, cgatol, cgrtol, OK, solver)
end
```
For our Steihaug-Toint implementation, we do not run any preprocess operation, so we use the default one.
```@example 1
function AdaptiveRegularization.preprocess(PData::AdaptiveRegularization.TPData, H, g, gNorm2, n1, n2, α)
    return PData
end
```
We now solve the subproblem.
```@example 1
function solve_modelST_TR(PData::PDataST, H, g, gNorm2, calls, max_calls, δ::T) where {T}
    ζ, ξ, maxtol, mintol = PData.ζ, PData.ξ, PData.maxtol, PData.mintol
    n = length(g)
    # precision = max(1e-12, min(0.5, (gNorm2^ζ)))
    # Tolerance used in Assumption 2.6b in the paper ( ξ > 0, 0 < ζ ≤ 1 )
    cgatol = PData.cgatol(ζ, ξ, maxtol, mintol, gNorm2)
    cgrtol = PData.cgrtol(ζ, ξ, maxtol, mintol, gNorm2)

    solver = PData.solver
    cg!(
        solver,
        H,
        -g,
        atol = cgatol,
        rtol = cgrtol,
        radius = δ,
        itmax = min(max_calls - sum(calls), max(2 * n, 50)),
        verbose = 0,
    )

    PData.d .= solver.x
    PData.OK = solver.stats.solved

    return PData.d, PData.λ
end
```

We can now proceed with the main solver call specifying the used `pdata_type` and `solve_model`. Since, `Krylov.cg_lanczos` only uses matrix-vector products, it is sufficient to evaluate the Hessian matrix as an operator, so we provide `hess_type = HessOp`.
```@example 1
ST_TROp(nlp; kwargs...) = TRARC(nlp, pdata_type = PDataST, solve_model = solve_modelST_TR, hess_type = HessOp; kwargs...)
```
Finally, we can apply our new method to any [`NLPModels`](https://github.com/JuliaSmoothOptimizers/NLPModels.jl).
```@example 1
using ADNLPModels, OptimizationProblems
nlp = OptimizationProblems.ADNLPProblems.arglina()
ST_TROp(nlp)
```

```@example 1
using ADNLPModels, NLPModels, OptimizationProblems, SolverBenchmark

meta = OptimizationProblems.meta
problems = meta[meta.variable_nvar .& (meta.ncon .== 0) .& .!(meta.has_bounds), :name]
n = 150
op_problems = (OptimizationProblems.ADNLPProblems.eval(Meta.parse(pb))(n = n) for pb in problems)

max_time = 120.0
max_ev = typemax(Int)
max_iter = typemax(Int)
atol = 1e-5
rtol = 1e-6

solvers = Dict(
    :ARCqKOp =>
        nlp -> ARCqKOp(
            nlp,
            verbose = false,
            atol = atol,
            rtol = rtol,
            max_time = max_time,
            max_iter = max_iter,
        ),
    :ST_TROp =>
        nlp -> ST_TROp(
            nlp,
            verbose = false,
            atol = atol,
            rtol = rtol,
            max_time = max_time,
            max_iter = max_iter,
        ),
)
stats = bmark_solvers(solvers, op_problems)
```
