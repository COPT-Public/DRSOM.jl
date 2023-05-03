function preprocess(PData::PDataTRK, Hop, g, gNorm2, calls, max_calls, α)
  ζ, ξ, maxtol, mintol = PData.ζ, PData.ξ, PData.maxtol, PData.mintol
  nshifts = PData.nshifts
  shifts = PData.shifts

  n = length(g)
  # Tolerance used in Assumption 2.6b in the paper ( ξ > 0, 0 < ζ ≤ 1 )
  cgatol = PData.cgatol(ζ, ξ, maxtol, mintol, gNorm2)
  cgrtol = PData.cgrtol(ζ, ξ, maxtol, mintol, gNorm2)

  nshifts = length(shifts)
  solver = PData.solver

  cb = (slv) -> begin
    if slv.converged[1]
      return norm(slv.x[1]) ≤ α
    else
      for i = 1:length(shifts)
        if !slv.not_cv[i] && (norm(slv.x[i]) - α > 0)
          return true
        end
      end
    end
    return false
  end
  cg_lanczos_shift!(
    solver,
    Hop,
    g,
    shifts,
    itmax = min(max_calls - sum(calls), 2 * n),
    atol = cgatol,
    rtol = cgrtol,
    verbose = 0,
    check_curvature = true,
    callback = cb,
  )

  PData.indmin = 0
  PData.positives .= solver.converged
  for i = 1:nshifts
    @. PData.xShift[i] = -solver.x[i]
    PData.norm_dirs[i] = norm(solver.x[i]) # Get it from cg_lanczos?
  end
  PData.shifts .= shifts
  PData.nshifts = nshifts
  PData.OK = sum(solver.converged) != 0 # at least one system was solved

  return PData
end
