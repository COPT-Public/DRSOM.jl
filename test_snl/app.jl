###############
# @date: Tue Apr 19 2022
# @author: Chuwen Zhang
# -----
# (c) 2022 Chuwen Zhang
# -----
# A script for the Sensor Network Localization using optionally SDP relaxation 
#    and DRSOM to minimize the second-stage nonlinear least-square
# see snl.jl for more descriptions.


###############

include("src/snl.jl")

using Plots
using Printf
using Random
using ArgParse
using .SNL


function parse_cmd()
    _args = parse_args(SNL.s, as_symbols=true)
    display(_args)
    return _args
end


function julia_main()::Cint
    _args = parse_cmd()
    if _args[:c] == 1
        snldata = SNL.create_snl_data(_args)
    else
        snldata = matread(_args[:fp])
    end
    n, m, nf, pp = snldata["n"], snldata["m"], snldata["nf"], snldata["PP"]
    @printf("finished loading")
    Nx = SNL.create_neighborhood(
        snldata["n"], snldata["m"], snldata["PP"], snldata["r"], snldata["nf"], snldata["deg"]
    )
    edges = Dict(nx.edge => nx.distn for nx in Nx)
    @printf("neighborhood created with size: %.3e\n", length(Nx))

    # PHASE-I: initialzation
    if _args[:option_use_sdr] == 1
        # if start with SDP relaxation
        Zv, Yv, Xv = SNL.SDR(n, m, nf, pp, Nx, edges)
    else
        # else simply start at all 0s
        Xv = zeros(2, n - m)
    end
    # PHASE-II: 
    state_drsom, k = SNL.drsom_nls(
        n, m, pp, Nx, Xv,
        1e-5, 1e5, true,
        _args[:timelimit], 20
    )

    Xvr = reshape(state_drsom.x, 2, n - m)
    md = _args[:option_set_comparison]
    if length(md) >= 1
        othermd = md[1]
    else
        othermd = 0
    end
    if "gd" ∈ othermd
        state_grad_gd, k = SNL.gd_nls(
            n, m, pp, Nx, Xv,
            1e-5, 1e5, true,
            _args[:timelimit], 30
        )
        println(state_grad_gd)
        Xvg = reshape(state_grad_gd.minimizer, 2, n - m)
    end

    if "cg" ∈ othermd
        state_grad_cg, k = SNL.cg_nls(
            n, m, pp, Nx, Xv,
            1e-5, 1e5, true,
            _args[:timelimit], 30
        )

        Xvc = reshape(state_grad_cg.minimizer, 2, n - m)
        println(state_grad_cg)
    end


    if _args[:option_plot_js] == 1
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
            markercolor="grey99", markeralpha=0, markerstrokealpha=1, markershape=:circle, label="DRSOM"
        )

        if "gd" ∈ othermd
            scatter!(
                fig, Xvg[1, :], Xvg[2, :], markerstrokecolor=[:purple],
                markercolor="grey99", markeralpha=0, markerstrokealpha=1, markershape=:circle, label="GD"
            )
        end
        if "cg" ∈ othermd
            scatter!(
                fig, Xvc[1, :], Xvc[2, :], markerstrokecolor=[:green],
                markercolor="grey99", markeralpha=0, markerstrokealpha=1, markershape=:circle, label="CG"
            )
        end

        name = @sprintf("drsom_snl_%d_%d_%d", n, m, _args[:option_use_sdr])
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

        if _args[:option_use_sdr] == 1
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

        name = @sprintf("drsom_snl_%d_%d_%d", n, m, _args[:option_use_sdr])
        savefig(fig, @sprintf("/tmp/%s.tikz", name))
        savefig(fig, @sprintf("/tmp/%s.tex", name))
        savefig(fig, @sprintf("/tmp/%s.pdf", name))
        savefig(fig, @sprintf("/tmp/%s.png", name))

        if "gd" ∈ othermd
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


            if _args[:option_use_sdr] == 1
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
            name = @sprintf("gd_snl_%d_%d_%d", n, m, _args[:option_use_sdr])
            savefig(fig, @sprintf("/tmp/%s.tikz", name))
            savefig(fig, @sprintf("/tmp/%s.tex", name))
            savefig(fig, @sprintf("/tmp/%s.pdf", name))
            savefig(fig, @sprintf("/tmp/%s.png", name))
        end
        if "cg" ∈ othermd
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


            if _args[:option_use_sdr] == 1
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
                fig, Xvc[1, :], Xvc[2, :],
                markerstrokecolor=[:red], markeralpha=0, markersize=8, markerstrokealpha=1,
                markercolor="grey99", fillstyle=nothing, markershape=:circle, label="GD"
            )
            name = @sprintf("cg_snl_%d_%d_%d", n, m, _args[:option_use_sdr])
            savefig(fig, @sprintf("/tmp/%s.tikz", name))
            savefig(fig, @sprintf("/tmp/%s.tex", name))
            savefig(fig, @sprintf("/tmp/%s.pdf", name))
            savefig(fig, @sprintf("/tmp/%s.png", name))
        end
    end

    return 0

end

julia_main()
