const solvers_const = Dict(
  :ARCqKdense =>
    (HessDense, PDataKARC, solve_modelKARC, [(:shifts => 10.0 .^ (collect(-20.0:1.0:20.0)))]),
  :ARCqKOp =>
    (HessOp, PDataKARC, solve_modelKARC, [:shifts => 10.0 .^ (collect(-20.0:1.0:20.0))]),
  :ARCqKsparse =>
    (HessSparse, PDataKARC, solve_modelKARC, [:shifts => 10.0 .^ (collect(-20.0:1.0:20.0))]),
  :ARCqKCOO =>
    (HessSparseCOO, PDataKARC, solve_modelKARC, [:shifts => 10.0 .^ (collect(-20.0:1.0:20.0))]),
  :ST_TRdense => (HessDense, PDataST, solve_modelST_TR, ()),
  :ST_TROp => (HessOp, PDataST, solve_modelST_TR, ()),
  :ST_TRsparse => (HessSparse, PDataST, solve_modelST_TR, ()),
  :TRKdense => (HessDense, PDataTRK, solve_modelTRK, ()),
  :TRKOp => (HessOp, PDataTRK, solve_modelTRK, ()),
  :TRKsparse => (HessSparse, PDataTRK, solve_modelTRK, ()),
)

const solvers_nls_const = Dict(
  :ARCqKOpGN => (
    HessGaussNewtonOp,
    PDataKARC,
    solve_modelKARC,
    [:shifts => 10.0 .^ (collect(-10.0:0.5:20.0))],
  ),
  :ST_TROpGN => (HessGaussNewtonOp, PDataST, solve_modelST_TR, ()),
  :ST_TROpGNLSCgls =>
    (HessGaussNewtonOp, PDataNLSST, solve_modelNLSST_TR, [:solver_method => :cgls]),
  :ST_TROpGNLSLsqr =>
    (HessGaussNewtonOp, PDataNLSST, solve_modelNLSST_TR, [:solver_method => :lsqr]),
  :ST_TROpLS => (HessOp, PDataNLSST, solve_modelNLSST_TR, ()),
)
