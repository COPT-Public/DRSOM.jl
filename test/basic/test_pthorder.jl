
using DRSOM


using LinearAlgebra
using SparseArrays
using Random

Random.seed!(1)


A = Symmetric(sprand(20, 20, 0.75), :U)
b = rand(20)
M = 1.3

xd, ld, _ = DRSOM.CubicSubpCholesky(A, b, M; Δϵ=1e-19)

@info begin
    """\n
    Direct
    dual*2/M : $(ld * 2/M)
    size     : $(xd |> norm)
    residual : $(2*ld/M - (xd |> norm))
    """
end
