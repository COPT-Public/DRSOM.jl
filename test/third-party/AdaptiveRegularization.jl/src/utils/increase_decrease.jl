"""
    decrease(X::TPData, α::T, TR::TrustRegion)

Return a decreased `α`.
"""
function decrease(X::TPData, α::T, TR::TrustRegion) where {T}
  return α * TR.decrease_factor
end

"""
    increase(X::TPData, α::T, TR::TrustRegion)

Return an increased `α`.
"""
function increase(::TPData, α::T, TR::TrustRegion) where {T}
  return min(α * TR.increase_factor, TR.max_α)
end

# X.indmin is between 1 and nshifts
function decrease(X::PDataKARC, α::T, TR::TrustRegion) where {T}
  X.indmin += 1 # the step wasn't successful so we need to change something
  α2 = max(X.norm_dirs[X.indmin] / X.shifts[X.indmin], eps(T))

  targetα = α * TR.decrease_factor

  # fix α to its "ideal" value to satisfy αλ=||d||
  # while ensuring α decreases enough
  last = findlast(X.positives)
  while !isnothing(last) && α2 > targetα && X.indmin < last
    X.indmin += 1
    α2 = max(X.norm_dirs[X.indmin] / X.shifts[X.indmin], eps(T))
  end

  if X.indmin == last & (α2 > targetα)
    @warn "PreProcessKARC failure: α2=$α2"
  end

  X.d .= X.xShift[X.indmin]
  X.λ = X.shifts[X.indmin]

  return α2
end

function decrease(X::PDataTRK, α::T, TR::TrustRegion) where {T}
  X.indmin += 1
  α2 = X.norm_dirs[X.indmin]

  # fix α to its "ideal" value to satisfy α=||d||
  # while ensuring α decreases enough
  targetα = α * TR.decrease_factor

  last = findlast(X.positives)
  while !isnothing(last) && α2 > targetα && X.indmin < last
    X.indmin += 1
    α2 = X.norm_dirs[X.indmin]
  end

  if X.indmin == last
    @warn "PreProcessTRK failure: α2=$α2"
  end

  X.d .= X.xShift[X.indmin]
  X.λ = X.shifts[X.indmin]

  return α2
end
