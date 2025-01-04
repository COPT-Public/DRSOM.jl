using JuMP, NLPModels, NLPModelsJuMP, Random, Distributions, LinearAlgebra, Test, Optim, DataFrames, StatsBase, CSV

Random.seed!(0)
const CAT_SOLVER = "CAT"
const NEWTON_TRUST_REGION_SOLVER = "NewtonTrustRegion"

function formulateMatrixCompletionProblem(M::Matrix, Ω::Matrix{Int64}, r::Int64, λ_1::Float64, λ_2::Float64)
    @show "Creating model"
    model = Model()

    D = transpose(M)
    n_1 = size(D)[1]
    n_2 = size(D)[2]
    Ω = transpose(Ω)
    temp_D = Ω .* D
    μ = mean(temp_D)
    @show "Creating variables"
    A = ones(n_1, r)
    B = ones(n_2, r)
    @variable(model, P[i=1:n_1, j=1:r], start = A[i, j])
    @variable(model, Q[i=1:n_2, j=1:r], start = B[i, j])

    @NLexpression(model, sum_observer_deviation_rows_squared, sum(((1 / n_2) * sum(sum(P[i, k] * transpose(Q)[k, j] for k in 1:r) for j in 1:n_2) - μ)^2 for i in 1:n_1))

    @NLexpression(model, sum_observer_deviation_columns_squared, sum(((1 / n_1) * sum(sum(P[i, k] * transpose(Q)[k, j] for k in 1:r) for i in 1:n_1) - μ)^2 for j in 1:n_2))

    @NLexpression(model, frobeniusNorm_P, sum(sum(P[i, j]^2 for j in 1:r) for i in 1:n_1))

    @NLexpression(model, frobeniusNorm_Q, sum(sum(Q[i, j]^2 for j in 1:r) for i in 1:n_2))

    @NLexpression(model, square_loss, 0.5 * (sum(sum(Ω[i, j] * (D[i, j] - μ - ((1 / n_2) * sum(sum(P[i, k] * transpose(Q)[k, j] for k in 1:r) for j in 1:n_2) - μ) - ((1 / n_1) * sum(sum(P[i, k] * transpose(Q)[k, j] for k in 1:r) for i in 1:n_1) - μ) - sum(P[i, k] * transpose(Q)[k, j] for k in 1:r))^2 for j in 1:n_2) for i in 1:n_1)))

    @show "Defining objective function"
    @NLobjective(model, Min, square_loss + λ_1 * (sum_observer_deviation_rows_squared + sum_observer_deviation_columns_squared) + λ_2 * (frobeniusNorm_P + frobeniusNorm_Q))
    return model
end

function getData(directoryName::String, fileName::String, rows::Int64, columns::Int64)
    M = prepareData(directoryName, fileName, rows, columns)
    Ω = sampleData(M)
    return M, Ω
end

function getData(directoryName::String, fileName::String, rows::Int64, columns::Int64, i::Int64, j::Int64)
    M = prepareData(directoryName, fileName, rows, columns, i, j)
    Ω = sampleData(M)
    return M, Ω
end

function prepareData(directoryName::String, fileName::String, rows::Int64, columns::Int64)
    filePath = string(directoryName, "/", fileName)
    df = DataFrame(CSV.File(filePath))
    for i in 1:size(df)[1]
        for j in 1:size(df)[2]
            if typeof(df[i, j]) == Missing
                df[i, j] = 1.0
            end
        end
    end

    df = Missings.replace(df, 1.0)
    df = df[2:(2+rows-1), 5:(5+columns-1)]
    M = Matrix(df)
    return transpose(M)
end

function prepareData(directoryName::String, fileName::String, rows::Int64, columns::Int64, i::Int64, j::Int64)
    @show "Reading file: $fileName"
    filePath = string(directoryName, "/", fileName)
    df = DataFrame(CSV.File(filePath))
    @show "Replacing missing values"
    for i in 1:size(df)[1]
        for j in 1:size(df)[2]
            if typeof(df[i, j]) == Missing
                df[i, j] = 1.0
            end
        end
    end

    df = Missings.replace(df, 1.0)
    @show "Creating Matrix M"
    df = df[2:size(df)[1], 5:size(df)[2]]
    df = df[1+rows*(i-1):rows*i, 1+columns*(j-1):columns*j]
    M = Matrix(df)
    return M
end

function sampleData(M::Matrix)
    rows = size(M)[1]
    columns = size(M)[2]
    T = rows * columns
    Ω = rand(DiscreteUniform(0, 1), rows, columns)
    return Ω
end