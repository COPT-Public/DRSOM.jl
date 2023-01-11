function solve_modelNLSST_TR(PData::PDataNLSST, Jx, Fx, norm_∇f, calls, max_calls, δ::T) where {T}
  # cas particulier Steihaug-Toint
  # ϵ = sqrt(eps(T)) # * 100.0 # old
  # cgtol = max(ϵ, min(cgtol, 9 * cgtol / 10, 0.01 * norm(g)^(1.0 + PData.ζ))) # old

  ζ, ξ, maxtol, mintol = PData.ζ, PData.ξ, PData.maxtol, PData.mintol
  n = length(Fx)
  # precision = max(1e-12, min(0.5, (norm_∇f^ζ)))
  # Tolerance used in Assumption 2.6b in the paper ( ξ > 0, 0 < ζ ≤ 1 )
  cgatol = PData.cgatol(ζ, ξ, maxtol, mintol, norm_∇f)
  cgrtol = PData.cgrtol(ζ, ξ, maxtol, mintol, norm_∇f)

  solver = PData.solver
  Krylov.solve!(
    solver,
    Jx,
    Fx,
    atol = cgatol,
    rtol = cgrtol,
    radius = δ,
    itmax = min(max_calls - sum(calls), max(2 * n, 50)),
    verbose = 0,
  )

  @. PData.d = -solver.x
  PData.OK = solver.stats.solved

  return PData.d, PData.λ
end
