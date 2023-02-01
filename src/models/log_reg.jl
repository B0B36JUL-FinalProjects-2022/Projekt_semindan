export Log_reg

"""
    Log_reg(; lr = 0.1, max_iter = 10000, w = nothing, epsilon = 1e-08)

Creates the logistic regression model instance.
The learning rate `lr` must be positive, `max_iter` is greater than 1 and the `epsilon` is positive.
"""
mutable struct Log_reg <: Model
    lr::Real
    max_iter::Int
    w::Union{AbstractArray, Nothing}
    epsilon::Real
    function Log_reg(; lr = 0.1, max_iter = 10000, w = nothing, epsilon = 1e-08)
        lr <= 0 && throw(ErrorException("n must be positive"))
        max_iter < 1 && throw(ErrorException("max iters should be greater than 1"))
        epsilon <= 0 && throw(ErrorException("epsilon must be positive"))
        new(lr, max_iter, w, epsilon)
    end
end

"""
    model_fit!(model, X, y)

Fits the logreg model to the training data, `X` is a DataFrame and `y` is an array containing the target labels.
"""
function model_fit!(model::Log_reg, X::DataFrame, y::AbstractArray)
    X_matrix = Matrix(X)'
    y_minus_plus = replace(y, 0 => -1)
    f = w -> logistic_loss(X_matrix, y_minus_plus, w)
    g = w -> logistic_loss_grad(X_matrix, y_minus_plus, w)
    w = ones(size(X_matrix)[1])
    model.w = gradient_descent(g, w; lr = model.lr, max_iter = model.max_iter)
end
"""
    gradient_descent(g, w; lr = 0.01, max_iter = 10000)

Performs gradient descent optimization on the given function `g` with initial parameters `w`. The learning rate `lr` and maximum number of iterations `max_iter` are optional parameters. Returns the optimized parameters `w`.
"""
function gradient_descent(g::Function, w::AbstractArray; lr = 0.01, max_iter = 10000)
    for _ in 1:max_iter
        w = w - lr * g(w)
    end
    w
end
"""
    model_predict(model, tst)

Predicts the outputs of the logistic regression model `model` on the test data `tst`. Returns an array of the predicted outputs.
"""
function model_predict(model::Log_reg, tst::DataFrame)
    replace(map(sign, Matrix(tst) * model.w), -1 => 0)
end
"""
    logistic_loss(X, y, w)

Calculates the logistic loss of the given data `X`, ground truth `y` and model parameters `w`. Returns the logistic loss.
"""
logistic_loss(X::AbstractMatrix, y::AbstractArray, w::AbstractArray) = length(y)^-1 * sum(log.(1 .+ exp.(-y .* (X' * w))))

"""
    logistic_loss_grad(X, y, w)

Calculates the gradient of the logistic loss with respect to the model parameters `w`, for the given data `X`, ground truth `y`. Returns the gradient of the logistic loss.
"""
function logistic_loss_grad(X::AbstractMatrix, y::AbstractArray, w::AbstractArray)
    -length(y)^-1 * ((1 ./ (1 .+ exp.(y .* (X' * w))))' * (X' .* y))'
end
