# increasing the dim of DRSOM subproblems

using DRSOM, DataFrames, CSV
using AdaptiveRegularization
using Test
using Optim


include("../tools.jl")
include("../lp.jl")
using .LP
options = Optim.Options(
    g_tol=1e-6,
    iterations=5000,
    store_trace=true,
    show_trace=true,
    show_warnings=false,
    show_every=50
)

params = LP.LPMinimizationParams(
    n=500, m=200, p=0.5, nnz=0.25
)
Random.seed!(2)
A, v, b = LP.create_random_lp(params)
x₀ = zeros(params.m)
ϵ = 1e-2
# reset λ
params.λ = norm(A' * b, Inf) / 20

f(x) = 0.5 * norm(A * x - b)^2 + LP.huberlike(params.λ, ϵ, params.p, x)
g(x) = (A'*(A*x-b))[:] + LP.huberlikeg(params.λ, ϵ, params.p, x)
H(x) = A' * A + LP.huberlikeh(params.λ, ϵ, params.p, x)
hvp(x, v, buff) = copyto!(buff, (A'*A*v)[:] + LP.huberlikeh(params.λ, ϵ, params.p, x) * v)

alg = DRSOM2()
algex = DRSOMEx()
r = alg(x0=copy(x₀), f=f, g=g, hvp=hvp, sog=:hvp, maxiter=500, freq=1)
rex = algex(x0=copy(x₀), f=f, g=g, hvp=hvp, sog=:hvp, maxiter=500, freq=1, dimmax=5)
rex = algex(x0=copy(x₀), f=f, g=g, hvp=hvp, sog=:hvp, maxiter=500, freq=1, dimmax=10)

