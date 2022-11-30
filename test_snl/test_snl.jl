###############
# file: test_snl.jl
# project: test
# created Date: Mo Apr yyyy
# author: <<author>
# -----
# last Modified: Tue Apr 19 2022
# modified By: Chuwen Zhang
# -----
# (c) 2022 Chuwen Zhang
# -----
# HISTORY:
# Date      	By	Comments
# ----------	---	---------------------------------------------------------
# Test script for the Sensor Network Localization using optionally SDP relaxation and RSOM to minimize the second-stage nonlinear least-square
# see snl.jl for more descriptions.


###############
include("snl.jl")

using MAT
using Plots
using Printf
using Random
using .SNL

# create the data for SNL
option = 1
option_plot_js = true
option_use_sdr = false

size = parse(Int, ARGS[1])
timelimit = parse(Int, ARGS[2])
# create random data
snldata = Dict()
n = size
Random.seed!(1) # for reproducibility
snldata["m"] = m = Int(n / 30 |> round)
snldata["PP"] = pp = rand(Float64, (2, n)) .- 0.5
snldata["r"] = radius = 0.5
snldata["nf"] = nf = 0.05
snldata["deg"] = degree = Int(n / 20 |> round)
matwrite(@sprintf("/tmp/test%d-%d.mat", m, n), snldata)


Nx = SNL.create_neighborhood(n, m, pp, radius, nf, degree)
edges = Dict(nx.edge => nx.distn for nx in Nx)
@printf("neighborhood created with size: %.3e\n", length(Nx))

if option_use_sdr
    Zv, Yv, Xv = SNL.SDR(n, m, nf, pp, Nx, edges)
else
    Xv = zeros(2, n - m)
end

state_drsom, k = SNL.drsom_nls(n, m, pp, Nx, Xv, 1e-6, 3e2, true, timelimit, 10)
state_grad, k = SNL.gd_nls(n, m, pp, Nx, Xv, 1e-6, 3e3, true, timelimit, 30)

Xvr = reshape(state_drsom.x, 2, n - m)
# Xvf = reshape(state_fista.x, 2, n - m)
Xvg = reshape(state_grad.minimizer, 2, n - m)


if option_plot_js
    # js + html backend
    plotly()
    fig = scatter(
        Xv[1, :], Xv[2, :],
        markerstrokecolor=[:red],
        markercolor="grey99",
        fillstyle=nothing,
        markershape=:circle,
        label="SDR",
        size=(1080, 960),
    )
    scatter!(
        fig, pp[1, 1:n-m], pp[2, 1:n-m], markershape=:xcross, markerstrokecolor=[:black], markersize=4, markerstrokewidth=0.1, label="Truth"
    )

    scatter!(fig, pp[1, n-m+1:n], pp[2, n-m+1:n], markershape=:utriangle, markersize=4, markerstrokewidth=0.1, label="Anchors")


    scatter!(
        fig, Xvr[1, :], Xvr[2, :], markerstrokecolor=[:black],
        markercolor="grey99", fillstyle=nothing, markershape=:circle, label="DRSOM"
    )

    # scatter!(
    #     fig, Xvf[1, :], Xvf[2, :], markerstrokecolor=[:purple],
    #     markercolor="grey99", fillstyle=nothing, markershape=:circle, label="FISTA"
    # )

    scatter!(
        fig, Xvg[1, :], Xvg[2, :], markerstrokecolor=[:purple],
        markercolor="grey99", fillstyle=nothing, markershape=:circle, label="GD"
    )
    name = @sprintf("rsom_snl_%d_%d_%d", n, m, option_use_sdr)
    savefig(fig, @sprintf("/tmp/%s.html", name))
else
    # publication
    pgfplotsx()
    fig = scatter(
        pp[1, 1:n-m], pp[2, 1:n-m],
        markershape=:xcross,
        markerstrokecolor=[:black],
        markersize=8,
        markerstrokewidth=1.5,
        label="Truth",
        legendfontsize=24,
        tickfontsize=16,
        size=(1080, 960),
    )

    if option_use_sdr
        scatter!(
            fig,
            Xv[1, :], Xv[2, :],
            markerstrokecolor=[:blue],
            markercolor="grey99",
            fillstyle=nothing,
            markershape=:circle,
            label="SDR")
    end

    scatter!(
        fig, pp[1, n-m+1:n], pp[2, n-m+1:n],
        markershape=:rect,
        markersize=8,
        markercolor=[:green],
        markerstrokewidth=0.1,
        label="Anchors"
    )


    scatter!(
        fig, Xvr[1, :], Xvr[2, :],
        markerstrokecolor=[:red], markeralpha=0, markersize=8, markerstrokealpha=1,
        markercolor="grey99", fillstyle=nothing, markershape=:circle, label="DRSOM"
    )

    name = @sprintf("rsom_snl_%d_%d_%d", n, m, option_use_sdr)
    savefig(fig, @sprintf("/tmp/%s.tikz", name))
    savefig(fig, @sprintf("/tmp/%s.tex", name))
    savefig(fig, @sprintf("/tmp/%s.pdf", name))
    savefig(fig, @sprintf("/tmp/%s.png", name))

    fig = scatter(
        pp[1, 1:n-m], pp[2, 1:n-m],
        markershape=:xcross,
        markerstrokecolor=[:black],
        markersize=8,
        markerstrokewidth=1.5,
        label="Truth",
        legendfontsize=24,
        tickfontsize=16,
        size=(1080, 960),
    )

    if option_use_sdr
        scatter!(
            fig,
            Xv[1, :], Xv[2, :],
            markerstrokecolor=[:blue],
            markercolor="grey99",
            fillstyle=nothing,
            markershape=:circle,
            label="SDR",)
    end

    scatter!(
        fig, pp[1, n-m+1:n], pp[2, n-m+1:n],
        markershape=:rect,
        markersize=8,
        markercolor=[:green],
        markerstrokewidth=0.1,
        label="Anchors"
    )

    scatter!(
        fig, Xvg[1, :], Xvg[2, :],
        markerstrokecolor=[:red], markeralpha=0, markersize=8, markerstrokealpha=1,
        markercolor="grey99", fillstyle=nothing, markershape=:circle, label="GD"
    )
    name = @sprintf("gd_snl_%d_%d_%d", n, m, option_use_sdr)
    savefig(fig, @sprintf("/tmp/%s.tikz", name))
    savefig(fig, @sprintf("/tmp/%s.tex", name))
    savefig(fig, @sprintf("/tmp/%s.pdf", name))
    savefig(fig, @sprintf("/tmp/%s.png", name))

end