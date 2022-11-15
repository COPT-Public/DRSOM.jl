module drsom_helper_plus
using ProximalOperators
using DRSOM
using ProximalAlgorithms
using Plots
using Printf
using LazyStack
using HTTP
using LaTeXStrings
using LinearAlgebra
using Statistics
using Optim
using ForwardDiff
using ReverseDiff
using LineSearches

Base.@kwdef mutable struct Result{StateType,Int}
    name::String
    state::StateType
    k::Int
    traj::Vector{StateType}
end


basename = "DRSOMPlus"
naming(dirtype, numdir) = @sprintf("%s(%s,%s)", basename, dirtype, numdir)
########################################################
# the DRSOM runner
########################################################
# provide 4 types of run:
# - run_drsomf: (forward), run DRSOM with forward automatic differentiation
# - run_drsomb: (backward), run DRSOM with backward automatic differentiation (recommended)
# - run_drsomd: (direct mode), run DRSOM with provided g(⋅) and H(⋅)
# - run_drsomd_traj: (direct mode) run add save trajactory

function run_drsomb(x0, f_composite; tol=1e-6, maxiter=100, maxtime=100, freq=1, record=true, direction=:gaussian, direction_num=1)
    ########################################################
    arr = Vector{DRSOM.DRSOMPlusState}()
    rb = nothing
    name = naming(direction, direction_num)
    @printf("%s\n", '#'^60)
    @printf("running: %s with tol: %.3e\n", name, tol)
    f_tape = ReverseDiff.GradientTape(f_composite, x0)
    f_tape_compiled = ReverseDiff.compile(f_tape)
    @printf("compile finished\n")
    @printf("%s\n", '#'^60)
    iter = DRSOM.DRSOMPlusIteration(x0=x0, rh=DRSOM.hessba, f=f_composite, tp=f_tape_compiled, mode=:backward, direction=direction, direction_num=direction_num)
    for (k, state::DRSOM.DRSOMPlusState) in enumerate(iter)
        (record) && push!(arr, copy(state))
        if k >= maxiter || state.t >= maxtime || DRSOM.drsom_stopping_criterion(tol, state)
            rb = (state, k)
            DRSOM.drsom_display(k, state)
            break
        end
        (k == 1 || mod(k, freq) == 0) && DRSOM.drsom_display(k, state)
    end
    @printf("finished with iter: %.3e, objval: %.3e\n", rb[2], rb[1].fx)
    return Result(name=name, state=rb[1], k=rb[2], traj=arr)
end

function run_drsomd(x0, f_composite, g, H; tol=1e-6, maxiter=100, maxtime=100, freq=1, record=true, direction=:gaussian, direction_num=1)
    ########################################################
    name = naming(direction, direction_num)
    arr = Vector{DRSOM.DRSOMPlusState}()
    rb = nothing
    @printf("%s\n", '#'^60)
    @printf("running: %s with tol: %.3e\n", name, tol)
    iter = DRSOM.DRSOMPlusIteration(x0=x0, f=f_composite, g=g, H=H, mode=:direct, direction=direction, direction_num=direction_num)
    for (k, state::DRSOM.DRSOMPlusState) in enumerate(iter)
        (record) && push!(arr, copy(state))
        if k >= maxiter || state.t >= maxtime || DRSOM.drsom_stopping_criterion(tol, state)
            rb = (state, k)
            DRSOM.drsom_display(k, state)
            break
        end
        (k == 1 || mod(k, freq) == 0) && DRSOM.drsom_display(k, state)
    end
    @printf("finished with iter: %.3e, objval: %.3e\n", rb[2], rb[1].fx)
    return Result(name=name, state=rb[1], k=rb[2], traj=arr)
end


# general options for Optim
# GD and LBFGS, Trust Region Newton,
options = Optim.Options(
    g_tol=1e-6,
    iterations=10000,
    store_trace=true,
    show_trace=true,
    show_every=50,
)

end