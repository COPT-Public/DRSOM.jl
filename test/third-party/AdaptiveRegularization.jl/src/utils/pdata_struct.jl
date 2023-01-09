export PDataTRK, PDataKARC, PDataST

abstract type TPData{T} end

abstract type PDataFact{T} <: TPData{T} end # Variants using matricial factorization

abstract type PDataIter{T} <: TPData{T} end # Variants using iterative (Krylov) solvers

abstract type PDataIterLS{T} <: TPData{T} end # Variants using iterative (Krylov) solvers for least-square subproblem

"""
    preprocess(PData::TPData, H, g, gNorm2, n1, n2, α)

Function called in the `TRARC` algorithm every time a new iterate has been accepted.
# Arguments
- `PData::TPData`: data structure used for preprocessing.
- `H`: current Hessian matrix.
- `g`: current gradient.
- `gNorm2`: 2-norm of the gradient.
- `n1`: Current count on the number of Hessian-vector products.
- `n2`: Maximum number of Hessian-vector products accepted.
- `α`: current value of the TR/ARC parameter.

It returns `PData`.
"""
function preprocess(PData::TPData, H, g, gNorm2, n1, n2, α)
  return PData
end

"""
    solve_model(PData::TPData, H, g, gNorm2, n1, n2, α)

Function called in the `TRARC` algorithm to solve the subproblem.
# Arguments
- `PData::TPData`: data structure used for preprocessing.
- `H`: current Hessian matrix.
- `g`: current gradient.
- `gNorm2`: 2-norm of the gradient.
- `n1`: Current count on the number of Hessian-vector products.
- `n2`: Maximum number of Hessian-vector products accepted.
- `α`: current value of the TR/ARC parameter.

It returns a couple `(PData.d, PData.λ)`. Current implementations include: `solve_modelKARC`, `solve_modelTRK`, `solve_modelST_TR`.
"""
function solve_model(X::TPData, H, g, gNorm2, n1, n2, α) end

"""
    PDataKARC(::Type{S}, ::Type{T}, n)
Return a structure used for the preprocessing of ARCqK methods.
"""
mutable struct PDataKARC{S, T, Fatol, Frtol} <: PDataIter{T}
  d::S            # (H+λI)\g ; on first call = g
  λ::T                      # "active" value of λ; on first call = 0
  ζ::T                      # Inexact Newton order parameter: stop when ||∇q|| < ξ * ||g||^(1+ζ)
  ξ::T                      # Inexact Newton order parameter: stop when ||∇q|| < ξ * ||g||^(1+ζ)
  maxtol::T                 # Largest tolerance for Inexact Newton
  mintol::T                 # Smallest tolerance for Inexact Newton
  cgatol::Fatol
  cgrtol::Frtol

  indmin::Int               # index of best shift value  within "positive". On first call = 0

  positives::Array{Bool, 1}   # indices of the shift values yielding (H+λI)⪰0
  xShift::Array{S, 1}        # solutions for each shifted system
  shifts::Array{T, 1}        # values of the shifts
  nshifts::Int              # number of shifts
  norm_dirs::Array{T, 1}     # norms of xShifts
  OK::Bool                  # preprocess success

  solver::CgLanczosShiftSolver{T, T, S}
end

function PDataKARC(
  ::Type{S},
  ::Type{T},
  n;
  ζ = T(0.5),
  ξ = T(0.01),
  maxtol = T(0.01),
  mintol = sqrt(eps(T)),
  cgatol = (ζ, ξ, maxtol, mintol, gNorm2) -> max(mintol, min(maxtol, ξ * gNorm2^(1 + ζ))),
  cgrtol = (ζ, ξ, maxtol, mintol, gNorm2) -> max(mintol, min(maxtol, ξ * gNorm2^ζ)),
  shifts = 10.0 .^ collect(-20.0:1.0:20.0),
  kwargs...,
) where {S, T}
  d = S(undef, n)
  λ = zero(T)
  indmin = 1
  nshifts = length(shifts)
  positives = Array{Bool, 1}(undef, nshifts)
  xShift = Array{S, 1}(undef, nshifts)
  for i = 1:nshifts
    xShift[i] = S(undef, n)
  end
  norm_dirs = S(undef, nshifts)
  OK = true
  solver = CgLanczosShiftSolver(n, n, nshifts, S)
  return PDataKARC(
    d,
    λ,
    ζ,
    ξ,
    maxtol,
    mintol,
    cgatol,
    cgrtol,
    indmin,
    positives,
    xShift,
    T.(shifts),
    nshifts,
    norm_dirs,
    OK,
    solver,
  )
end

"""
    PDataTRK(::Type{S}, ::Type{T}, n)
Return a structure used for the preprocessing of TRK methods.
"""
mutable struct PDataTRK{S, T, Fatol, Frtol} <: PDataIter{T}
  d::S            # (H+λI)\g ; on first call = g
  λ::T                      # "active" value of λ; on first call = 0
  ζ::T                      # Inexact Newton order parameter: stop when ||∇q|| < ξ * ||g||^(1+ζ)
  ξ::T                      # Inexact Newton order parameter: stop when ||∇q|| < ξ * ||g||^(1+ζ)
  maxtol::T                 # Largest tolerance for Inexact Newton
  mintol::T                 # Smallest tolerance for Inexact Newton
  cgatol::Fatol
  cgrtol::Frtol

  indmin::Int               # index of best shift value  within "positive". On first call = 0

  positives::Array{Bool, 1}   # indices of the shift values yielding (H+λI)⪰0
  xShift::Array{S, 1}        # solutions for each shifted system
  shifts::Array{T, 1}        # values of the shifts
  nshifts::Int              # number of shifts
  norm_dirs::Array{T, 1}     # norms of xShifts
  OK::Bool                  # preprocess success

  solver::CgLanczosShiftSolver{T, T, S}
end

function PDataTRK(
  ::Type{S},
  ::Type{T},
  n;
  ζ = T(0.5),
  ξ = T(0.01),
  maxtol = T(0.01),
  mintol = sqrt(eps(T)),
  cgatol = (ζ, ξ, maxtol, mintol, gNorm2) -> max(mintol, min(maxtol, ξ * gNorm2^(1 + ζ))),
  cgrtol = (ζ, ξ, maxtol, mintol, gNorm2) -> max(mintol, min(maxtol, ξ * gNorm2^ζ)),
  shifts = T[0.0; 10.0 .^ (collect(-20.0:1.0:20.0))],
  kwargs...,
) where {S, T}
  d = S(undef, n)
  λ = zero(T)
  indmin = 1
  nshifts = length(shifts)
  positives = Array{Bool, 1}(undef, nshifts)
  xShift = Array{S, 1}(undef, nshifts)
  for i = 1:nshifts
    xShift[i] = S(undef, n)
  end
  norm_dirs = S(undef, nshifts)
  OK = true
  solver = CgLanczosShiftSolver(n, n, nshifts, S)
  return PDataTRK(
    d,
    λ,
    ζ,
    ξ,
    maxtol,
    mintol,
    cgatol,
    cgrtol,
    indmin,
    positives,
    xShift,
    shifts,
    nshifts,
    norm_dirs,
    OK,
    solver,
  )
end

"""
    PDataST(::Type{S}, ::Type{T}, n)
Return a structure used for the preprocessing of Steihaug-Toint methods.
"""
mutable struct PDataST{S, T, Fatol, Frtol} <: PDataIter{T}
  d::S
  λ::T
  ζ::T                      # Inexact Newton order parameter: stop when ||∇q|| < ξ * ||g||^(1+ζ)
  ξ::T                      # Inexact Newton order parameter: stop when ||∇q|| < ξ * ||g||^(1+ζ)
  maxtol::T                 # Largest tolerance for Inexact Newton
  mintol::T                 # Smallest tolerance for Inexact Newton
  cgatol::Fatol
  cgrtol::Frtol

  OK::Bool    # preprocess success
  solver::CgSolver{T, T, S}
end

function PDataST(
  ::Type{S},
  ::Type{T},
  n;
  ζ = T(0.5),
  ξ = T(0.01),
  maxtol = T(0.01),
  mintol = sqrt(eps(T)),
  cgatol = (ζ, ξ, maxtol, mintol, gNorm2) -> max(mintol, min(maxtol, ξ * gNorm2^(1 + ζ))),
  cgrtol = (ζ, ξ, maxtol, mintol, gNorm2) -> max(mintol, min(maxtol, ξ * gNorm2^ζ)),
  kwargs...,
) where {S, T}
  d = S(undef, n)
  λ = zero(T)
  OK = true
  solver = CgSolver(n, n, S)
  return PDataST(d, λ, ζ, ξ, maxtol, mintol, cgatol, cgrtol, OK, solver)
end

"""
    PDataNLSST(::Type{S}, ::Type{T}, n)
Return a structure used for the preprocessing of Steihaug-Toint methods for Gauss-Newton approximation of nonlinear least squares.
"""
mutable struct PDataNLSST{S, T, Fatol, Frtol} <: PDataIterLS{T}
  d::S
  λ::T
  ζ::T                      # Inexact Newton order parameter: stop when ||∇q|| < ξ * ||g||^(1+ζ)
  ξ::T                      # Inexact Newton order parameter: stop when ||∇q|| < ξ * ||g||^(1+ζ)
  maxtol::T                 # Largest tolerance for Inexact Newton
  mintol::T                 # Smallest tolerance for Inexact Newton
  cgatol::Fatol
  cgrtol::Frtol

  OK::Bool    # preprocess success
  solver::Union{CglsSolver{T, T, S}, LsqrSolver{T, T, S}}
end

function PDataNLSST(
  ::Type{S},
  ::Type{T},
  n,
  m;
  ζ = T(0.5),
  ξ = T(0.01),
  maxtol = T(0.01),
  mintol = sqrt(eps(T)),
  cgatol = (ζ, ξ, maxtol, mintol, gNorm2) -> max(mintol, min(maxtol, ξ * gNorm2^(1 + ζ))),
  cgrtol = (ζ, ξ, maxtol, mintol, gNorm2) -> max(mintol, min(maxtol, ξ * gNorm2^ζ)),
  solver_method = :cgls,
  kwargs...,
) where {S, T}
  d = S(undef, n)
  λ = zero(T)
  OK = true
  solver = if solver_method == :cgls
    CglsSolver(m, n, S)
  else
    LsqrSolver(m, n, S)
  end
  return PDataNLSST(d, λ, ζ, ξ, maxtol, mintol, cgatol, cgrtol, OK, solver)
end
