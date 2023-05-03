using NLPModels, Stopping, AdaptiveRegularization

mutable struct EmptyNLPModel{T, S} <: AbstractNLPModel{T, S}
  meta::NLPModelMeta{T, S}
  counters::Counters
end

NLPModels.obj(::EmptyNLPModel{T, S}, x) where {T, S} = one(T)
function NLPModels.grad!(::EmptyNLPModel{T, S}, x, gx::S) where {T, S}
  gx .= one(T)
  return gx
end
function NLPModels.hprod!(
  ::EmptyNLPModel,
  x::AbstractVector{T},
  v::AbstractVector{T},
  Hv::AbstractVector{T};
  obj_weight = one(T),
) where {T}
  Hv .= zero(T)
  return Hv
end
function NLPModels.hess_structure!(
  ::EmptyNLPModel{T, S},
  rows::AbstractVector,
  cols::AbstractVector,
) where {T, S}
  rows .= one(T)
  cols .= one(T)
  return rows, cols
end
function NLPModels.hess_coord!(
  ::EmptyNLPModel{T, S},
  x::AbstractVector{T},
  vals::AbstractVector;
  obj_weight = one(T),
) where {T, S}
  vals .= zero(T)
  return vals
end

import NLPModels: hess, hess_coord!, hess_coord, hess_op, hess_op!, hprod, hprod!

mutable struct NothingPData{T} <: AdaptiveRegularization.TPData{T}
  OK::Any
  d::Any
  λ::Any
end

n = 100

function NothingPData(::Type{S}, ::Type{T}, n; kwargs...) where {T, S}
  return NothingPData{T}(true, rand(T, n), one(T))
end

function solve_nothing(X::NothingPData, H, g, ng, calls, max_calls, α::T) where {T}
  X.d .= g
  X.λ = zero(T)
  return X.d, X.λ
end

x0 = rand(n)
nlp = EmptyNLPModel{eltype(x0), typeof(x0)}(NLPModelMeta(n), Counters())

for (Workspace, limit) in (
  (HessDense, 286624),
  (HessSparse, 84016),
  (HessSparseCOO, 0), # independent of `n`
  (HessOp, 2304), # independent of `n`
)
  who = Workspace(nlp, n)
  alloc_hessian(who, nlp, x0) = @allocated AdaptiveRegularization.hessian!(who, nlp, x0)
  alloc_hessian(who, nlp, x0)
  @test (alloc_hessian(who, nlp, x0)) <= limit
  @show alloc_hessian(who, nlp, x0)
end

stp = NLPStopping(nlp)
stp.meta.max_iter = 5
TRnothing = TrustRegion(10.0, max_unsuccinarow = 2)

nlp = stp.pb
S, T = typeof(x0), eltype(x0)
PData = NothingPData(S, T, nlp.meta.nvar)
workspace = AdaptiveRegularization.TRARCWorkspace(nlp, HessOp)

alloc_AdaptiveRegularization(stp, PData, workspace, TRnothing, solve_nothing) =
  @allocated TRARC(stp, PData, workspace, TRnothing, solve_model = solve_nothing)
alloc_AdaptiveRegularization(stp, PData, workspace, TRnothing, solve_nothing)
alloc_AdaptiveRegularization(stp, PData, workspace, TRnothing, solve_nothing)
@show alloc_AdaptiveRegularization(stp, PData, workspace, TRnothing, solve_nothing) # 11192

workspace = AdaptiveRegularization.TRARCWorkspace(nlp, HessSparseCOO)

alloc_AdaptiveRegularization_COO(stp, PData, workspace, TRnothing, solve_nothing) =
  @allocated TRARC(stp, PData, workspace, TRnothing, solve_model = solve_nothing)
alloc_AdaptiveRegularization_COO(stp, PData, workspace, TRnothing, solve_nothing)
alloc_AdaptiveRegularization_COO(stp, PData, workspace, TRnothing, solve_nothing)
@show alloc_AdaptiveRegularization_COO(stp, PData, workspace, TRnothing, solve_nothing) # 11192
