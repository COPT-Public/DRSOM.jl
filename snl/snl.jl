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

using DRSOM
using Random
using Printf
using JuMP
using MosekTools
using LinearAlgebra
using ProximalAlgorithms
using Optim
using LineSearches
using ReverseDiff

#######################
# plot defaults
#######################

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


function create_neighborhood(n, m, pp, radius, nf, degree)
    Random.seed!(2) # reset seed
    Nx = NeighborVector()
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


"""
least_square
"""

# function least_square(n, m, points, pp, edges::Dict)
#     val = 0
#     for (i, j) in keys(edges)
#         xij = j < n - m + 1 ? points[:, i] - points[:, j] : points[:, i] - pp[:, j]
#         dh = sqrt(xij .^ 2 |> sum)
#         val += (dh - edges[i, j])^2
#     end
#     return val
# end

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


# function rsom_nls(n, m, pp, edges, Xv::Matrix{Float64}, tol::Float64, max_iter::Real, verbose::Bool)
#     function loss(x::AbstractVector{T}) where {T}
#         xv = reshape(x, 2, n - m)
#         return least_square(n, m, xv, pp, edges)
#     end
#     x0 = vec(Xv)
#     iter = DRSOM.DRSOMFreeIteration(x0=x0, f=loss)
#     rb = nothing
#     for (k, state::DRSOM.DRSOMFreeState) in enumerate(iter)

#         if k >= max_iter || DRSOM.drsom_stopping_criterion(tol, state)
#             rb = state, k
#             break
#         end
#         verbose && mod(k, 1) == 0 && DRSOM.drsom_display(k, state)
#     end
#     return rb
# end


function rsom_nls(n, m, pp, Nx::NeighborVector, Xv::Matrix{Float64}, tol::Float64, max_iter::Real, verbose::Bool)
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


function gd_nls(n, m, pp, Nx::NeighborVector, Xv::Matrix{Float64}, tol::Float64, max_iter::Real, verbose::Bool)
    function loss(x::AbstractVector{T}) where {T}
        xv = reshape(x, 2, n - m)
        return least_square(n, m, xv, pp, Nx)
    end
    x0 = vec(Xv)
    options = Optim.Options(
        g_tol=tol,
        iterations=round(Int, max_iter),
        store_trace=true,
        show_trace=true,
        show_every=50,
    )
    res1 = Optim.optimize(loss, x0, GradientDescent(; alphaguess=LineSearches.InitialHagerZhang(),
            linesearch=LineSearches.StrongWolfe()), options; autodiff=:forward)
    return res1, res1
end


end # module
