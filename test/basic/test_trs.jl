
using DRSOM


using LinearAlgebra
using SparseArrays
using Random

Random.seed!(1)


A = Symmetric(sprand(20, 20, 0.75), :U)
b = rand(20)

Δ = 3.0

x, l, k = DRSOM.LanczosTrustRegionBisect(A, -b, Δ)

@info begin
    """\n
    Exact
    dual    : $l
    size    : $(x |> norm) $(Δ)
    residual: $(abs2.((A + l*I)*x + b) |> maximum)
    """
end


lc = DRSOM.Lanczos(20, 30, b)
x1, l1, info = DRSOM.InexactLanczosTrustRegionBisect(A, -b, Δ, lc; k=21)


@info begin
    """\n
    Inexact
    dual    : $l1
    size    : $(x1 |> norm) $(Δ)
    residual: $(abs2.((A + l1*I)*x1 + b) |> maximum)
    """
end


xd, ld, _ = DRSOM.TrustRegionCholesky(A, b, Δ)

@info begin
    """\n
    Direct
    dual    : $ld
    size    : $(xd |> norm) $(Δ)
    residual: $(abs2.((A + ld*I)*xd + b) |> maximum)
    """
end
