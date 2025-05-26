

using DRSOM, DataFrames, CSV
using AdaptiveRegularization
using Test
using Optim


bool_plot = true
bool_calc = true
iterations = []
names = []
times = []
if bool_calc
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
        n=1000, m=500, p=0.5, nnz=0.25
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
    @testset "grad & hess" begin
        r = alg(x0=copy(x₀), f=f, g=g, H=H, sog=:hess)
        push!(times, r.state.t)
        push!(iterations, r.k)
        push!(names, L"g \& H")
    end
    @testset "grad & hvp" begin
        r = alg(x0=copy(x₀), f=f, g=g, hvp=hvp, sog=:hvp)
        push!(times, r.state.t)
        push!(iterations, r.k)
        push!(names, L"g \& \textrm{hvp}")
    end
    @testset "grad & direct" begin
        r = alg(x0=copy(x₀), f=f, g=g, sog=:direct)
        push!(times, r.state.t)
        push!(iterations, r.k)
        push!(names, L"g \& \textrm{interpolation}")
    end
    @testset "forward direct" begin
        r = alg(x0=copy(x₀), f=f, fog=:forward, sog=:direct)
        push!(times, r.state.t)
        push!(iterations, r.k)
        push!(names, L"\textrm{AD (forward)} \& \textrm{interpolation}")
    end
    @testset "forward forward" begin
        r = alg(x0=copy(x₀), f=f, fog=:forward, sog=:forward)
        push!(times, r.state.t)
        push!(iterations, r.k)
        push!(names, L"\textrm{AD (forward)} \& \textrm{AD (forward)}")
    end
    @testset "backward direct" begin
        r = alg(x0=copy(x₀), f=f, fog=:backward, sog=:direct)
        push!(times, r.state.t)
        push!(iterations, r.k)
        push!(names, L"\textrm{AD (backward)} \& \textrm{interpolation}")
    end
    @testset "backward backward" begin
        r = alg(x0=copy(x₀), f=f, fog=:backward, sog=:backward)
        push!(times, r.state.t)
        push!(iterations, r.k)
        push!(names, L"\textrm{AD (backward)} \& \textrm{AD (backward)")
    end
end