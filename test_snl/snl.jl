###############
# file: snl.jl
# project: SNL
# created Date: Apr 2022
# author: Chuwen Zhang
# -----
# last Modified: Wed Apr 20 2022
# modified By: Chuwen Zhang
# -----
# (c) 2022 Chuwen Zhang, Yinyu Ye
# -----
# HISTORY:
# Date      	By	Comments
# ----------	---	---------------------------------------------------------

# Module for modeling Sensor Network Localization using SDP relaxation and RSOM to minimize the second-stage nonlinear least-square.
# We suggest the user to go through [1] for complete description of this problem.
#
# References:
# 1. Wang, Z., Zheng, S., Ye, Y., Boyd, S.: Further relaxations of the semidefinite programming approach to sensor network localization. SIAM Journal on Optimization. 19, 655–673 (2008)
# 2. Biswas, P., Lian, T.-C., Wang, T.-C., Ye, Y.: Semidefinite programming based algorithms for sensor network localization. ACM Transactions on Sensor Networks (TOSN). 2, 188–220 (2006)
# 3. Biswas, P., Liang, T.-C., Toh, K.-C., Ye, Y., Wang, T.-C.: Semidefinite programming approaches for sensor network localization with noisy distance measurements. IEEE transactions on automation science and engineering. 3, 360–371 (2006)
# 4. Biswas, P., Ye, Y.: Semidefinite programming for ad hoc wireless sensor network localization. In: Third International Symposium on Information Processing in Sensor Networks, IPSN 2004. pp. 46–54 (2004)
# 
###############



__precompile__()

module SNL
using MAT
using DRSOM
using Random
using Printf
using LinearAlgebra
using ProximalAlgorithms
using Optim
using LineSearches
using ReverseDiff
using JuMP
using ArgParse
try
    using MosekTools
    function SDR(n, m, nf::Real, pp::Matrix{Float64}, Nx::NeighborVector, edges::Dict)::Tuple
        # we build standard SDP relaxation
        solver = Mosek.Optimizer
        model = Model(solver)
        dd = n - m
        e = [ones(dd, 1); sum(pp[:, dd+1:n], dims=2)] / sqrt(n)
        C = 1.4 * nf * (e * e' - I(dd + 2))
        @variable(model, Z[1:dd+2, 1:dd+2], PSD)
        @variable(model, wp[keys(edges)] >= 0)
        @variable(model, wn[keys(edges)] >= 0)
        @constraint(model, Z[1:2, 1:2] .== [1 0; 0 1])
        for nx in Nx
            @constraint(model,
                LinearAlgebra.tr(Z * nx.vec * nx.vec') + wp[nx.edge] - wn[nx.edge] == nx.distn)
        end
        @objective(model, Min, sum(wp) + sum(wn) + tr(C * Z))
        @printf("SDR built finished \n")
        optimize!(model)

        Zv = value.(Z)
        Yv = Zv[3:end, 3:end]
        Xv = Zv[1:2, 3:end]

        @printf("SDR residual trace(Y - X'X): %.2e\n", Yv - Xv' * Xv |> LinearAlgebra.tr)
        return Zv, Yv, Xv
    end
catch e
    @warn("Unable to import JuMP and MosekTools, in which case you cannot use SDP relaxation")
    @warn(" If you intend to use SDR, then you should add JuMP and MosekTools as requirements")
end


Base.@kwdef mutable struct Neighbor
    edge::Tuple{Int,Int}
    type::Char
    vec::Vector{Float64}
    dist::Float64
    distn::Float64
end

NeighborVector = Vector{Neighbor}

function ei(i, n)
    ev = zeros(n)
    ev[i] = 1
    return ev
end

function create_snl_data(param::Dict)
    # create random data
    snldata = Dict()
    Random.seed!(1) # for reproducibility
    snldata["m"] = param[:m]
    snldata["n"] = param[:n]
    snldata["PP"] = rand(Float64, (2, param[:n])) .- 0.5
    snldata["r"] = param[:r]
    snldata["nf"] = param[:nf]
    snldata["deg"] = param[:degree]
    matwrite(@sprintf("/tmp/test%d-%d.mat", param[:m], param[:n]), snldata)
    return snldata
end

function create_neighborhood(n, m, pp, radius, nf, degree)
    Random.seed!(2) # reset seed
    Nx = NeighborVector()
    if Threads.nthreads() > 1
        Threads.@threads for i::Int = 1:n-m
            flag = 0
            for j::Int = i+1:n
                rr = norm(pp[:, i] - pp[:, j])
                distn = rr * sqrt(max(0, (1 + randn() * nf)))
                if rr < radius && flag < degree
                    flag = flag + 1
                    if j <= n - m
                        nxv = [0; 0; ei(i, n - m) - ei(j, n - m)]
                        push!(Nx, Neighbor(edge=(i, j), type='x', vec=nxv, dist=rr, distn=distn))
                    else
                        nav = [-pp[:, j]; ei(i, n - m)]
                        push!(Nx, Neighbor(edge=(i, j), type='a', vec=nav, dist=rr, distn=distn))
                    end
                end
            end
        end
        return Nx
    else
        for i::Int = 1:n-m
            flag = 0
            for j::Int = i+1:n
                rr = norm(pp[:, i] - pp[:, j])
                distn = rr * sqrt(max(0, (1 + randn() * nf)))
                if rr < radius && flag < degree
                    flag = flag + 1
                    if j <= n - m
                        nxv = [0; 0; ei(i, n - m) - ei(j, n - m)]
                        push!(Nx, Neighbor(edge=(i, j), type='x', vec=nxv, dist=rr, distn=distn))
                    else
                        nav = [-pp[:, j]; ei(i, n - m)]
                        push!(Nx, Neighbor(edge=(i, j), type='a', vec=nav, dist=rr, distn=distn))
                    end
                end
            end
        end
        return Nx
    end

end

function least_square(n, m, points, pp, Nx::NeighborVector)
    val = 0
    for nx in Nx
        i, j = nx.edge
        xij = j < n - m + 1 ? points[:, i] - points[:, j] : points[:, i] - pp[:, j]
        dh = xij .^ 2 |> sum
        val += (dh - nx.distn^2)^2
    end
    return val
end

function drsom_nls_legacy(n, m, pp, Nx::NeighborVector, Xv::Matrix{Float64}, tol::Float64, max_iter::Real, verbose::Bool)
    function loss(x::AbstractVector{T}) where {T}
        xv = reshape(x, 2, n - m)
        return least_square(n, m, xv, pp, Nx)
    end
    x0 = vec(Xv)
    f_tape = ReverseDiff.GradientTape(loss, x0)
    f_tape_compiled = ReverseDiff.compile(f_tape)
    @printf("compile finished\n")
    iter = DRSOM.DRSOMFreeIteration(x0=x0, rh=DRSOM.hessba, f=loss, tp=f_tape_compiled, mode=:backward)
    rb = nothing
    for (k, state::DRSOM.DRSOMFreeState) in enumerate(iter)

        if k >= max_iter || DRSOM.drsom_stopping_criterion(tol, state)
            rb = state, k
            break
        end
        verbose && mod(k, 1) == 0 && DRSOM.drsom_display(k, state)
    end
    return rb
end

function drsom_nls(n, m, pp, Nx::NeighborVector, Xv::Matrix{Float64}, tol::Float64, max_iter::Real, verbose::Bool, max_time::Real=100.0, freq=10)
    function loss(x::AbstractVector{T}) where {T}
        xv = reshape(x, 2, n - m)
        return least_square(n, m, xv, pp, Nx)
    end
    x0 = vec(Xv)
    g(x) = DRSOM.ReverseDiff.gradient(loss, x)
    iter = DRSOM.DRSOMIteration(x0=x0, f=loss, g=g, H=Nothing(), mode=:direct)
    rb = nothing
    for (k, state::DRSOM.DRSOMState) in enumerate(iter)

        if k >= max_iter || state.t >= max_time || DRSOM.drsom_stopping_criterion(tol, state)
            rb = (state, k)
            DRSOM.drsom_display(k, state)
            break
        end
        verbose && (k == 1 || mod(k, freq) == 0) && DRSOM.drsom_display(k, state)
    end
    @printf("finished with iter: %.3e, objval: %.3e\n", rb[2], rb[1].fx)
    return rb
end



function gd_nls(n, m, pp, Nx::NeighborVector, Xv::Matrix{Float64}, tol::Float64, max_iter::Real, verbose::Bool, max_time::Real=100.0, freq=20)
    function loss(x::AbstractVector{T}) where {T}
        xv = reshape(x, 2, n - m)
        return least_square(n, m, xv, pp, Nx)
    end
    x0 = vec(Xv)
    g(x) = DRSOM.ReverseDiff.gradient(loss, x)
    options = Optim.Options(
        g_tol=tol,
        iterations=round(Int, max_iter),
        store_trace=true,
        show_trace=true,
        show_every=freq,
        time_limit=max_time
    )
    res1 = Optim.optimize(
        loss, g, x0,
        GradientDescent(; alphaguess=LineSearches.InitialHagerZhang(),
            linesearch=LineSearches.StrongWolfe()
        ),
        options;
        inplace=false
    )
    return res1, res1
end


function fista_nls(n, m, pp, Nx::NeighborVector, Xv::Matrix{Float64}, tol::Float64, max_iter::Real, verbose::Bool)
    function loss(x::AbstractVector{T}) where {T}
        xv = reshape(x, 2, n - m)
        return least_square(n, m, xv, pp, Nx)
    end
    x0 = vec(Xv)

    iter = ProximalAlgorithms.FastForwardBackwardIteration(x0=x0, f=loss)
    rb = nothing
    for (k, state::ProximalAlgorithms.FastForwardBackwardState) in enumerate(iter)

        if k >= max_iter || ProximalAlgorithms.default_stopping_criterion(tol, iter, state)
            rb = state, k
            break
        end
        verbose && mod(k, 40) == 1 && DRSOM.default_display(k, state.f_x, state.gamma, norm(state.res, Inf))
    end
    return rb
end



s = ArgParseSettings(
    description="""
        A script for the Sensor Network Localization 
            using optionally SDP relaxation 
          and DRSOM to minimize the second-stage nonlinear least-square
          see snl.jl for more descriptions.
        if you create a instance that is very large, it is suggested that you use:
            `julia -t` to invoke this script
    """,
    exit_after_help=true,
    preformatted_description=true,
    preformatted_epilog=true
)

@add_arg_table s begin
    "--c"
    help = "if true, then create a new instance"
    arg_type = Int
    default = 1
    "--seed"
    arg_type = Int
    default = 1
    help = "random seed if you create a new instance"
    "--fp"
    arg_type = String
    help = "read the instance"
    "--n"
    arg_type = Int
    default = 80
    required = true
    help = "total number of sensors (including the anchors)"
    "--m"
    arg_type = Int
    default = 5
    help = "total number of anchors, (suggested #: n/30)"
    "--degree"
    arg_type = Int
    default = 5
    help = "degree, the number of edges for a sensor. (suggested #: n/20)"
    "--r"
    arg_type = Float64
    default = 0.5
    "--nf"
    arg_type = Float64
    default = 0.05
    "--option_plot_js"
    arg_type = Int
    default = 1
    "--option_use_sdr"
    arg_type = Int
    default = 0
    "--timelimit"
    arg_type = Int
    default = 300
end


end # module
