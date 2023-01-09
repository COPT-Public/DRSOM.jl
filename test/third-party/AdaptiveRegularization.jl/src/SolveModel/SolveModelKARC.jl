function solve_modelKARC(X::PDataKARC, H, g, gNorm2, calls, max_calls, α::T) where {T}
  # target value should be close to satisfy αλ=||d||
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
  indmin = argmin(target)
  X.indmin = start + indmin - 1
  p_imin = X.indmin
  X.d .= X.xShift[p_imin]
  X.λ = X.shifts[p_imin]

  return X.d, X.λ
end
