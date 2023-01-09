function solve_modelTRK(X::PDataTRK, H, g, gNorm2, calls, max_calls, α::T) where {T}
  # target value should be close to satisfy α=||d||
  start = findfirst(X.positives)
  if isnothing(start)
    start = 1
  end
  if VERSION < v"1.7.0"
    positives = collect(start:length(X.positives))
    target = [(abs(α * X.shifts[i] - X.norm_dirs[i])) for i in positives]
  else
    positives = start:length(X.positives)
    target = ((abs(α * X.shifts[i] - X.norm_dirs[i])) for i in positives)
  end
  # pick the closest shift to the target within positive definite H+λI
  # before, check the shift = 0 for direction within the trust region

  if (X.indmin == 0)  #  try Newton's direction
    X.indmin = 1
    if (start == 1) & (X.norm_dirs[1] <= α)
      X.d .= X.xShift[1]
      X.λ = zero(T)
      return X.d, X.λ
    end
  end
  indmin = argmin(target)
  X.indmin = start + indmin - 1
  p_imin = X.indmin
  X.d .= X.xShift[p_imin]
  # Refine this rough constrained direction
  X.λ = X.shifts[p_imin]

  return X.d, X.λ
end

# To replace.
#    1. Form the signed ( α - X.norm_dirs[i]) ) within positives of the form +++...----
#    2. If starts with a +, interpolate the interval where sign changes
#       else (hard case) obtain from Krylov shift solver negative curvature directions
#            and extrapolate to the boundary the short (longest) direction
#            in the descent negative curvature direction
#
# Keep on treating λ=0 special for Newton's direction (no extrapolation to boundary)
