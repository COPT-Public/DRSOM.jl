using AdaptiveRegularization, LinearAlgebra, Test

T = Float64
S = Vector{T}
n = 1000

TR = TrustRegion(T(10))
α = T(100)

for XData in (PDataKARC(S, T, n), PDataTRK(S, T, n), PDataST(S, T, n))
  @testset "Allocation test in AdaptiveRegularization.decrease for $(typeof(XData))" begin
    alloc_decrease() = @allocated AdaptiveRegularization.decrease(XData, α, TR)
    alloc_decrease()
    @test alloc_decrease() <= 16
  end
  @testset "Allocation test in AdaptiveRegularization.decrease for $(typeof(XData))" begin
    alloc_increase() = @allocated AdaptiveRegularization.increase(XData, α, TR)
    alloc_increase()
    @test (@allocated alloc_increase()) <= 16
  end
end

using ADNLPModels, NLPModels, OptimizationProblems, Stopping

nlp = OptimizationProblems.ADNLPProblems.arglina()
n = nlp.meta.nvar
x = nlp.meta.x0
stp = NLPStopping(nlp)
H = hess(nlp, x)
g = grad(nlp, x)
ng = norm(g)
calls, max_calls = 0, 1000000

for (Data, solve, limit_solve, limit_preprocess) in (
  (PDataKARC, :solve_modelKARC, 112, 0),
  (PDataTRK, :solve_modelTRK, 112, 0),
  (PDataST, :solve_modelST_TR, 192, 0),
)
  @testset "Allocation test in preprocess with $(Data)" begin
    XData = Data(S, T, n)
    alloc_preprocess(XData, H, g, ng, calls, max_calls, α) =
      @allocated AdaptiveRegularization.preprocess(XData, H, g, ng, calls, max_calls, α)
    alloc_preprocess(XData, H, g, ng, calls, max_calls, α)
    @test alloc_preprocess(XData, H, g, ng, calls, max_calls, α) <= limit_preprocess
    @show alloc_preprocess(XData, H, g, ng, calls, max_calls, α)
  end

  @testset "Allocation test in $solve with $(Data)" begin
    XData = Data(S, T, n)
    alloc_solve_model(XData, H, g, ng, calls, max_calls, α) =
      @allocated AdaptiveRegularization.eval(solve)(XData, H, g, ng, calls, max_calls, α)
    alloc_solve_model(XData, H, g, ng, calls, max_calls, α)
    @test alloc_solve_model(XData, H, g, ng, calls, max_calls, α) <= limit_solve
    @show alloc_solve_model(XData, H, g, ng, calls, max_calls, α)
  end
end
