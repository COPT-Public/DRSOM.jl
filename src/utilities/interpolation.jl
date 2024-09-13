using Base.Iterators
using LinearAlgebra
using Printf
using Dates
using KrylovKit

using Random


rng = MersenneTwister(1234)

function generate_sample_sphere(m)
    a = randn(rng, Float64, m)
    return a / norm(a)
end

function directional_interpolation(iter, state, V::Tv, c::Tc) where {Tv<:VecOrMat,Tc<:VecOrMat}

    m = length(c)
    l = m * (m + 1) / 2 |> Int
    a = [generate_sample_sphere(m) for i = 1:l] .* 1e-4
    A = hcat([build_natural_basis(z) for z in a]...)'
    d = [c' * z for z in a]
    # trial points
    x_n(_a) = (map((x, y) -> x * y, V, _a) |> sum) + state.x
    xs = [x_n(_a) for _a in a]
    b = [(x |> iter.f) - state.fx for x in xs] # rhs
    q = A \ (b - d)
    Q = Symmetric(
        [i <= j ? q[Int(j * (j - 1) / 2)+i] : 0
         for i = 1:m, j = 1:m], :U
    )

    # sanity check
    # H = iter.H(state.x); C = hcat(V...); Qc = C' * H * C
    return Q
end

function build_natural_basis(v)
    _len = length(v)
    a = [i == j ? v[i] * v[j] / 2 : v[i] * v[j]
         for i = 1:_len
         for j = i:_len]
    return a
end
