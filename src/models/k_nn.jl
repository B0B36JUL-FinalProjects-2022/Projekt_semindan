export K_nn, Metric

@enum Metric l1 l2 lmax

"""
    K_nn(;n=3, metric=l2, train_data_X=nothing, train_data_y=nothing)

An implementation of K-Nearest Neighbors algorithm.
"""
mutable struct K_nn <: Model
    n::Int
    metric::Metric
    train_data_X::DataFrame
    train_data_y::AbstractArray
    function K_nn(; n = 3, metric = l2, train_data_X = nothing, train_data_y = nothing)
        n <= 0 && throw(ErrorException("n must be positive"))
        model = new(n, metric)
        isnothing(findfirst(==(metric), instances(Metric))) &&
            throw(ErrorException("Invalid metric"))
        !isnothing(train_data_X) &&
            !isnothing(train_data_y) &&
            model_fit!(model, train_data_X, train_data_y)
        model
    end
end

"""
    model_fit!(model, X, y)

Fits the `model` to training data.
"""
function model_fit!(model::K_nn, X::DataFrame, y::AbstractArray)
    nrow(X) == length(y) ||
        throw(ErrorException("Number of X's rows must be equal to length(y), i.e. $(length(y)), found $(nrow(X))"))
    model.train_data_X = X
    model.train_data_y = y
end

"""
    model_predict(model, tst)

Predicts the output for test data `tst` using the `model`.
"""
function model_predict(model::K_nn, tst::DataFrame)
    distances = [metric_dist(model.metric, Matrix(model.train_data_X)', x)
                 for x in eachcol(Matrix(tst)')]
    dists_matrix = reduce(vcat, distances)
    predictions = [most_common(model.train_data_y[sortperm(row)][1:(model.n)])
                   for
                   row in eachrow(dists_matrix)]
    predictions
end
"""
    metric_dist(metric, x1, x2)

Computes the distance between vectors x1 and x2 according to the specified metric (l1, l2 or lmax).
"""
function metric_dist(metric::Metric, x1::AbstractArray, x2::AbstractArray)
    metric == Titanic.l2 && return sqrt.(sum((x1 .- x2) .^ 2, dims = 1))
    metric == Titanic.l1 && return sum(abs.(x1 .- x2), dims = 1)
    metric == Titanic.lmax && return maximum(x1 .- x2, dims = 1)
end
