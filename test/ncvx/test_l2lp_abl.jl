

using DRSOM, DataFrames, CSV
using AdaptiveRegularization
using Test
using Optim


bool_plot = true
bool_calc = false

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
    params.λ = norm(A' * b, Inf) / 10

    f(x) = 0.5 * norm(A * x - b)^2 + LP.huberlike(params.λ, ϵ, params.p, x)
    g(x) = (A'*(A*x-b))[:] + LP.huberlikeg(params.λ, ϵ, params.p, x)
    H(x) = A' * A + LP.huberlikeh(params.λ, ϵ, params.p, x)
    hvp(x, v, buff) = copyto!(buff, (A'*A*v)[:] + LP.huberlikeh(params.λ, ϵ, params.p, x) * v)

    alg = DRSOM2()
    r = alg(x0=copy(x₀), f=f, g=g, hvp=hvp, sog=:hvp)
    algex = DRSOMEx()
    dimmax = 10
    rex = algex(x0=copy(x₀), f=f, g=g, hvp=hvp, sog=:hvp, maxiter=500, freq=1, dimmax=dimmax)
    r_gd = Optim.optimize(
        f, g, x₀,
        GradientDescent(;),
        options;
        inplace=false
    )
    r_lbfgs10 = Optim.optimize(
        f, g, x₀,
        LBFGS(;
            m=10, alphaguess=LineSearches.InitialStatic(),
            linesearch=LineSearches.HagerZhang(),
        ),
        options;
        inplace=false,
    )
    r_tr = Optim.optimize(f, g, x₀, NewtonTrustRegion(), options; inplace=false)

end
if bool_plot
    results = [
        optim_to_result(r_gd, "GD"),
        r,
        rex,
        optim_to_result(r_lbfgs10, "LBFGS (10)"),
        optim_to_result(r_tr, "Newton-TR"),
    ]
    method_names = getname.(results)
    method_names[3] = "DRSOM ($dimmax)"
    # for metric in (:ϵ, :fx)
    # for metric in [:ϵ]
    metric = :ϵ
    truncoff = 1
    method_objval_ragged = rstack([
            getresultfield.(results, metric)...
        ]; fill=NaN
    )


    @printf("plotting results\n")

    pgfplotsx()
    title = L"$\ell_2-\ell_p$ minimization $p=1/2$"
    fig = plot(
        (1:(method_objval_ragged|>size|>first))[truncoff:end],
        method_objval_ragged[truncoff:end, :],
        label=permutedims(method_names),
        xscale=:log10,
        yscale=:log10,
        xlabel="Iteration",
        ylabel=metric == :ϵ ? L"\|\nabla f\|" : L"f(x)",
        # title=title,
        size=(900, 500),
        # yticks=[1e-7, 1e-6, 1e-4, 1e-2, 1e-1, 1e0, 1e1],
        xticks=[10, 100, 200, 500, 1000, 10000, 100000, 1e6],
        dpi=500,
        labelfontsize=18,
        xtickfont=font(18),
        ytickfont=font(18),
        leg=:bottomleft,
        legendfontsize=18,
        legendfontfamily="sans-serif",
        titlefontsize=24,
        # legendfontalign=:left,
        tex_output_standalone=true,
    )

    savefig(fig, "/tmp/$metric-l2lp.pdf")
end