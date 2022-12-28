abstract type Model end

mutable struct K_nn <: Model
    n
    metric
    train_data_X
    train_data_y
    K_nn(;n=3, metric="l2") = n < 0 ? throw(ErrorException("n must be positive")) : new(n, metric)
end

function model_fit(model::K_nn, X::DataFrame, y::AbstractArray) 
    model.train_data_X = Matrix(X)'
    model.train_data_y = y
    nothing
end

function model_fit(model::K_nn, X::Matrix, y::AbstractArray) 
    size(X)[2] == length(y) || throw(ErrorException("X must be of shape (features, entries)"))

    model.train_data_X = X
    model.train_data_y = y
end

function model_predict(model::K_nn, tst::DataFrame) 
    distances = [metric_dist(model.metric, model.train_data_X, x) for x in eachcol(Matrix(tst)')]
    dists_matrix = reduce(vcat, distances)
    predictions = [most_common(model.train_data_y[sortperm(row)][1:model.n]) for row in eachrow(dists_matrix)]
    predictions
end

function metric_dist(metric::String, x1, x2)
    metric == "l2" && return sqrt.(sum((x1 .- x2).^2, dims=1))
    metric == "l1" && return sum(abs.(x1 .- x2), dims=1)
end