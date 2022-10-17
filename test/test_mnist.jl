

include("helper.jl")

using MLDatasets
using Flux
using LinearAlgebra
using Statistics
using Flux: logitcrossentropy, normalise, onecold, onehotbatch
using DRSOM
using ProximalAlgorithms
using Plots
using Printf
using LazyStack
using ProximalOperators

import .drsom_helper


function logit_model(wbv, x)
    wb = reshape(wbv, 10, :)
    return wb[:, 1:end-1] * x' .+ wb[:, end]
end

# options
option_plot = false
# load full training set
x_train, y_train = MNIST.traintensor(Float64), MNIST.trainlabels()
x_train = permutedims(x_train, [3, 1, 2])
n_train, d, _ = size(x_train)
x_train = reshape(x_train, n_train, :)
y_all = unique(y_train)
yc_train = Flux.onehotbatch(y_train, y_all)
# load full test set
x_test, y_test = MNIST.testtensor(Float64), MNIST.testlabels()
x_test = permutedims(x_test, [3, 1, 2])
n_test, _, _ = size(x_test)
x_test = reshape(x_test, n_test, :)
yc_test = Flux.onehotbatch(y_test, y_all)


loss_train(wb) = Flux.logitcrossentropy(logit_model(wb, x_train), yc_train)
loss_test(wb) = Flux.logitcrossentropy(logit_model(wb, x_test), yc_test)
accuracy(x, y, wbv) = mean(onecold(logit_model(wbv, x)) .== onecold(y))

f = loss_train
g = ProximalOperators.Zero()
f_composite(x) = f(x)

iter_scale = 0
w0 = ones(7850)
method_objval = Dict{String,AbstractArray{Float64}}()
method_state = Dict{String,Any}()
########################################################
# rsom
r1 = drsom_helper.run_drsomb(copy(w0), f_composite; maxiter=10)

@printf("train error. %.4f\n", 1 - accuracy(x_train, yc_train, r1.state.x))
@printf("test error. %.4f\n", 1 - accuracy(x_test, yc_test, r1.state.x))


r1 = drsom_helper.run_drsomb(copy(w0), f_composite; maxiter=40)
method_objval[name] = copy(arr)
iter_scale = max(iter_scale, k) # compute max plot scale
@printf("train error. %.4f\n", 1 - accuracy(x_train, yc_train, r1.state.x))
@printf("test error. %.4f\n", 1 - accuracy(x_test, yc_test, r1.state.x))

r1 = drsom_helper.run_drsomb(copy(w0), f_composite; maxiter=100)
method_objval[name] = copy(arr)
iter_scale = max(iter_scale, k) # compute max plot scale
@printf("train error. %.4f\n", 1 - accuracy(x_train, yc_train, r1.state.x))
@printf("test error. %.4f\n", 1 - accuracy(x_test, yc_test, r1.state.x))

# method_objval_ragged = rstack([values(method_objval)...]; fill=NaN)

# if option_plot
#     pgfplotsx()

#     pl_residual = plot(
#         1:iter_scale,
#         method_objval_ragged,
#         label=permutedims([keys(method_objval)...]),
#         xscale=:log2,
#         yscale=:log10,
#         xlabel="Iteration", ylabel=L"Objective: $f$",
#         title=L"\textrm{Least Squares with }  \mathcal{L}_p \textrm{ Regularization}: 
#         \frac{1}{2}\|Ax-b\|^2 + \|x\|_p^p, p = 2",
#         size=(1080, 720),
#         # yticks=[1e-2, 1, 100, 1000, 1000],
#         # xticks=[1, 10, 100, 200, 500, 1000, 10000],
#         dpi=500
#     )

#     name = @sprintf("rsom_logistic_mnist")
#     savefig(pl_residual, @sprintf("/tmp/%s.tikz", name))
#     savefig(pl_residual, @sprintf("/tmp/%s.tex", name))
#     savefig(pl_residual, @sprintf("/tmp/%s.pdf", name))
#     savefig(pl_residual, @sprintf("/tmp/%s.png", name))
# end