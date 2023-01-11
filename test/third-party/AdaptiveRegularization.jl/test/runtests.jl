# This package
using AdaptiveRegularization
# stdlib
using LinearAlgebra, SparseArrays, Test
# JSO
using ADNLPModels, NLPModels, OptimizationProblems.ADNLPProblems, SolverTest
# Stopping
using Stopping

@testset "Testing NLP solvers" begin
  @testset "$name" for name in ALL_solvers
    solver = eval(name)
    unconstrained_nlp(solver)
    multiprecision_nlp(solver, :unc, precisions = (Float32, Float64))
  end
end

@testset "Testing NLS solvers" begin
  @testset "$name" for name in union(ALL_solvers, NLS_solvers)
    solver = eval(name)
    unconstrained_nls(solver)
    multiprecision_nls(solver, :unc, precisions = (Float32, Float64))
  end
end

global nbsolver = 0
for solver in ALL_solvers
  global nbsolver += 1
  nlp = extrosnb(n = 2)
  nlpstop = NLPStopping(nlp)
  println(nbsolver, "  ", solver)
  eval(solver)(nlpstop, verbose = true)
  final_nlp_at_x, optimal = nlpstop.current_state, nlpstop.meta.optimal
  @test optimal
  reset!(nlp)
  stats = eval(solver)(nlp, verbose = false)
  @test stats.status == :first_order
  reset!(nlp)
end

if VERSION >= v"1.7.0"
  include("allocation_test.jl")
  include("allocation_test_main.jl")
end
