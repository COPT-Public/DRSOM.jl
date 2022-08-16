module drsom_helper_l
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
    traj::Vector{Float64}
end


basename = "DRSOML"
naming(mode, rank) = @sprintf("%s(%s,+%s)", basename, mode, rank)
########################################################
# the DRSOM runner
########################################################
# provide 4 types of run:
# - run_drsomf: (forward), run DRSOM with forward automatic differentiation
# - run_drsomb: (backward), run DRSOM with backward automatic differentiation (recommended)
# - run_drsomd: (direct mode), run DRSOM with provided g(⋅) and H(⋅)
# - run_drsomd_traj: (direct mode) run add save trajactory

function run_drsomb(
    x0, f_composite; tol=1e-6, maxiter=100, freq=1,
    direction=:gaussian, direction_num=1,
    hessian=:sr1, hessian_rank=length(x0)
)
    ########################################################
    arr = Vector{Float64}()
    rb = nothing
    name = naming(hessian, hessian_rank)
    @printf("%s\n", '#'^60)
    @printf("running: %s with tol: %.3e\n", name, tol)
    f_tape = ReverseDiff.GradientTape(f_composite, x0)
    f_tape_compiled = ReverseDiff.compile(f_tape)
    @printf("compile finished\n")
    @printf("%s\n", '#'^60)

    iter = DRSOM.DRSOMLIteration(
        x0=x0, f=f_composite, tp=f_tape_compiled, mode=:backward,
        direction=direction, direction_num=direction_num,
        hessian=hessian, hessian_rank=hessian_rank
    )
    for (k, state::DRSOM.DRSOMLState) in enumerate(iter)
        push!(arr, state.ϵ)
        if k >= maxiter || DRSOM.drsom_stopping_criterion(tol, state)
            rb = (state, k)
            DRSOM.drsom_display(k, state)
            break
        end
        (k == 1 || mod(k, freq) == 0) && DRSOM.drsom_display(k, state)
    end
    @printf("finished with iter: %.3e, objval: %.3e\n", rb[2], arr[end])
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