# Benchmarks

## CUTEst benchmark

With a JSO-compliant solver, such as DCI, we can run the solver on a set of problems, explore the results, and compare to other JSO-compliant solvers using specialized benchmark tools. 
We are following here the tutorial in [SolverBenchmark.jl](https://juliasmoothoptimizers.github.io/SolverBenchmark.jl/v0.3/tutorial/) to run benchmarks on JSO-compliant solvers.
``` @example ex1
using CUTEst
```

To test the implementation of DCI, we use the package [CUTEst.jl](https://github.com/JuliaSmoothOptimizers/CUTEst.jl), which implements `CUTEstModel` an instance of `AbstractNLPModel`. 

``` @example ex1
using SolverBenchmark
```

Let us select unconstrained problems from CUTEst with a maximum of 300 variables.

``` @example ex1
nmax = 100
pnames = CUTEst.select(contype = "unc", max_var = nmax)

cutest_problems = (CUTEstModel(p) for p in pnames)

length(cutest_problems) # number of problems
```

We compare here AdaptiveRegularization with `trunk` from [`JSOSolvers.jl`](https://github.com/JuliaSmoothOptimizers/JSOSolvers.jl/) on a subset of CUTEst problems.

``` @example ex1
using AdaptiveRegularization, JSOSolvers

#Same time limit for all the solvers
max_time = 60. #20 minutes
atol, rtol = 1e-5, 1e-6

solvers = Dict(
  :trunk => nlp -> trunk(
    nlp,
    max_time = max_time,
    max_iter = typemax(Int64),
    max_eval = typemax(Int64),
    atol = atol,
    rtol = rtol,
  ),
  :ARCqK => nlp -> ARCqKOp(
    nlp,
    max_time = max_time,
    max_iter = typemax(Int64),
    max_eval = typemax(Int64),
    atol = atol,
    rtol = rtol,
  ),
)

stats = bmark_solvers(solvers, cutest_problems)
```
The function `bmark_solvers` return a `Dict` of `DataFrames` with detailed information on the execution. This output can be saved in a data file.
``` @example ex1
using JLD2
@save "trunk_arcqk_$(string(length(pnames))).jld2" stats
```
The result of the benchmark can be explored via tables,
``` @example ex1
pretty_stats(stats[:ARCqK])
```
or it can also be used to make performance profiles.
``` @example ex1
using Plots
gr()

legend = Dict(
  :neval_obj => "number of f evals", 
  :neval_cons => "number of c evals", 
  :neval_grad => "number of ∇f evals", 
  :neval_jac => "number of ∇c evals", 
  :neval_jprod => "number of ∇c*v evals", 
  :neval_jtprod  => "number of ∇cᵀ*v evals", 
  :neval_hess  => "number of ∇²f evals", 
  :elapsed_time => "elapsed time"
)
perf_title(col) = "Performance profile on CUTEst w.r.t. $(string(legend[col]))"

styles = [:solid,:dash,:dot,:dashdot] #[:auto, :solid, :dash, :dot, :dashdot, :dashdotdot]

function print_pp_column(col::Symbol, stats)
  
  ϵ = minimum(minimum(filter(x -> x > 0, df[!, col])) for df in values(stats))
  first_order(df) = df.status .== :first_order
  unbounded(df) = df.status .== :unbounded
  solved(df) = first_order(df) .| unbounded(df)
  cost(df) = (max.(df[!, col], ϵ) + .!solved(df) .* Inf)

  p = performance_profile(
    stats, 
    cost, 
    title=perf_title(col), 
    legend=:bottomright, 
    linestyles=styles
  )
end

print_pp_column(:elapsed_time, stats) # with respect to time
```

``` @example ex1
print_pp_column(:neval_hprod, stats) # with respect to number of Hession-vector products
```
