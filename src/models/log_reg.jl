export Log_reg

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

function model_fit!(model::Log_reg, X::DataFrame, y::AbstractArray)
    X_matrix = Matrix(Matrix(X)')
    y_minus_plus = replace(y, 0 => -1)
    f = w -> logistic_loss(X_matrix, y_minus_plus, w)
    g = w -> logistic_loss_grad(X_matrix, y_minus_plus, w)
    w = ones(size(X_matrix)[1])
    model.w = gradient_descent(f, g, w; lr = model.lr, max_iter = model.max_iter)
end

function gradient_descent(f, g, w; lr = 0.01, max_iter = 10000)
    for _ in 1:max_iter
        w = w - lr * g(w)
    end
    w
end

function model_predict(model::Log_reg, tst::DataFrame)
    replace(map(sign, Matrix(tst) * model.w), -1 => 0)
end
logistic_loss(X, y, w) = length(y)^-1 * sum(log.(1 .+ exp.(-y .* (X' * w))))
function logistic_loss_grad(X, y, w)
    -length(y)^-1 * ((1 ./ (1 .+ exp.(y .* (X' * w))))' * (X' .* y))'
end
