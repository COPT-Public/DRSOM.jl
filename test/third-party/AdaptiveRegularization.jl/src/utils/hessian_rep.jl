"""
    HessDense(::AbstractNLPModel{T,S}, n)
Return a structure used for the evaluation of dense Hessian matrix.
"""
struct HessDense{T}
  H::Matrix{T}
  function HessDense(::AbstractNLPModel{T, S}, n) where {T, S}
    H = Matrix{Float64}(undef, n, n)
    return new{T}(H)
  end
end

"""
    HessSparse(::AbstractNLPModel{T,S}, n)
Return a structure used for the evaluation of sparse Hessian matrix.
"""
struct HessSparse{T, S, Vi, It <: Integer}
  rows::Vi
  cols::Vi
  vals::S
  H::Symmetric{T, SparseMatrixCSC{T, It}}
  function HessSparse(nlp::AbstractNLPModel{T, S}, n) where {T, S}
    rows, cols = hess_structure(nlp)
    vals = S(undef, nlp.meta.nnzh)
    H = Symmetric(spzeros(T, n, n), :L)
    return new{T, S, typeof(rows), eltype(rows)}(rows, cols, vals, H)
  end
end

"""
    HessSparseCOO(::AbstractNLPModel{T,S}, n)
Return a structure used for the evaluation of sparse Hessian matrix in COO-format.
"""
struct HessSparseCOO{Tv, Ti}
  H::Symmetric{Tv, SparseMatrixCOO{Tv, Ti}}
end

function HessSparseCOO(nlp::AbstractNLPModel{T, S}, n) where {T, S}
  rows, cols = hess_structure(nlp)
  vals = S(undef, nlp.meta.nnzh)
  H = Symmetric(SparseMatrixCOO(n, n, rows, cols, vals), :L)
  return HessSparseCOO(H)
end

"""
    HessOp(::AbstractNLPModel{T,S}, n)
Return a structure used for the evaluation of the Hessian matrix as an operator.
"""
mutable struct HessOp{S}
  Hv::S
  H
  function HessOp(::AbstractNLPModel{T, S}, n) where {T, S}
    H = LinearOperator{T}(n, n, true, true, v -> v, v -> v, v -> v)
    return new{S}(S(undef, n), H)
  end
end

"""
    HessGaussNewtonOp(::AbstractNLSModel{T,S}, n)
Return a structure used for the evaluation of the Hessian matrix as an operator.
"""
mutable struct HessGaussNewtonOp{S}
  Jv::S
  Jtv::S
  H
  function HessGaussNewtonOp(nls::AbstractNLSModel{T, S}, n) where {T, S}
    Jx = LinearOperator{T}(nls.nls_meta.nequ, n, false, false, v -> v, v -> v, v -> v)
    return new{S}(S(undef, nls.nls_meta.nequ), S(undef, n), Jx' * Jx)
  end
end

export HessDense, HessSparse, HessSparseCOO, HessOp, HessGaussNewtonOp

"""
    init(::Type{Hess}, nlp::AbstractNLPModel{T,S}, n)

Return the hessian structure `Hess` and its composite type.
"""
function init(::Type{Hess}, nlp::AbstractNLPModel{T, S}, n) where {Hess, T, S}
  Hstruct = Hess(nlp, n)
  return Hstruct, typeof(Hstruct)
end

"""
    hessian!(workspace::HessDense, nlp, x)

Return the Hessian matrix of `nlp` at `x` in-place with memory update of `workspace`.
"""
function hessian! end

function hessian!(workspace::HessDense, nlp, x)
  workspace.H .= Matrix(hess(nlp, x))
  return workspace.H
end

function hessian!(workspace::HessOp, nlp, x)
  workspace.H = hess_op!(nlp, x, workspace.Hv)
  return workspace.H
end

function hessian!(workspace::HessGaussNewtonOp, nlp, x)
  Jx = jac_op_residual!(nlp, x, workspace.Jv, workspace.Jtv)
  workspace.H = Jx' * Jx
  return workspace.H
end

function hessian!(workspace::HessSparse, nlp, x)
  hess_coord!(nlp, x, workspace.vals)
  n = nlp.meta.nvar
  workspace.H .= Symmetric(sparse(workspace.rows, workspace.cols, workspace.vals, n, n), :L)
  return workspace.H
end

function hessian!(workspace::HessSparseCOO, nlp, x)
  hess_coord!(nlp, x, workspace.H.data.vals)
  return workspace.H
end
