abstract type Model end

function model_fit end
function model_predict end

# function model_fit(model::T, X::DataFrame, y::AbstractArray) where T <: Model
#     # size(X)[2] == length(y) || throw(ErrorException("X must be of shape (features, entries), i.e. (features, $(length(y))), found (features, $(size(X)[2]))"))
#     # model_fit(model, Matrix(Matrix(X)'), y)
#     # model_fit(model, Matrix(Matrix(X)'), y)
# end

@enum Metric l1 l2 lmax

mutable struct K_nn <: Model
    n::Int
    metric::Metric
    train_data_X::Union{Matrix, DataFrame}
    train_data_y::AbstractArray
    function K_nn(; n = 3, metric = l2, train_data_X = nothing, train_data_y = nothing)
        n <= 0 && throw(ErrorException("n must be positive"))
        model = new(n, metric)
        isnothing(findfirst(==(metric), instances(Metric))) && throw(ErrorException("Invalid metric"))
        !isnothing(train_data_X) && !isnothing(train_data_y) && model_fit(model, train_data_X, train_data_y)
        model
    end
end

function model_fit(model::K_nn, X::Matrix, y::AbstractArray)
    size(X)[2] == length(y) || throw(ErrorException("X must be of shape (features, entries), i.e. (features, $(length(y))), found (features, $(size(X)[2]))"))
    model.train_data_X = X
    model.train_data_y = y
end

function model_predict(model::K_nn, tst::DataFrame)
    distances = [metric_dist(model.metric, model.train_data_X, x) for x in eachcol(Matrix(tst)')]
    dists_matrix = reduce(vcat, distances)
    predictions = [
        most_common(model.train_data_y[sortperm(row)][1:model.n]) for
        row in eachrow(dists_matrix)
    ]
    predictions
end

function metric_dist(metric::Metric, x1, x2)
    metric == Titanic.l2 && return sqrt.(sum((x1 .- x2) .^ 2, dims = 1))
    metric == Titanic.l1 && return sum(abs.(x1 .- x2), dims = 1)
end

mutable struct Log_reg <: Model
    lr::Real
    max_iter::Int
    w::Union{AbstractArray, Nothing}
    epsilon::Real
    function Log_reg(; lr = 0.1, max_iter = 10000, w = nothing, epsilon=1e-08)
        lr <= 0 && throw(ErrorException("n must be positive"))
        max_iter < 1 && throw(ErrorException("max iters should be greater than 1"))
        epsilon <= 0 && throw(ErrorException("epsilon must be positive"))
        new(lr, max_iter, w, epsilon)
    end
end


# model_fit(model::Log_reg, X::DataFrame, y::AbstractArray, w_init; args...) = model_fit(model, Matrix(Matrix(X)'), y, w_init)
function model_fit(
    model::Log_reg,
    X::Matrix,
    y::AbstractArray
)
    # y = replace(y, 0 => -1)
    y[y .== 0] .= -1
    # w, L, g = w_init, logistic_loss(X, y, w_init), logistic_loss_grad(X, y, w_init)
    # wt, Lt = Matrix(w'), [L]
    diff = Inf
    f = w -> logistic_loss(X,y,w)
    g = w -> logistic_loss_grad(X,y,w)
    w = ones(size(X)[1])
    model.w = gradient_descent(f, g, w; lr=model.lr, max_iter=model.max_iter)
end

function gradient_descent(f, g, w; lr=0.01, max_iter=10000)
    for _ in 1:max_iter
        w = w - lr * g(w)
    end
    return w
end

    # y[y .== 0] = -1
    # w, L, g = w_init, logistic_loss(X, y, w_init), logistic_loss_grad(X, y, w_init)
    # wt, Lt = Matrix(w'), [L]
    # diff = Inf
    # for i = 1:max_iter
    #     diff <= epsilon && break
    #     w_new = w - lr * g
    #     L_new, g_new = logistic_loss(X, y, w_new), logistic_loss_grad(X, y, w_new)
    #     if L_new < L
    #         wt = vcat(wt, w_new')
    #         Lt = vcat(Lt, L_new)
    #         w, L, g = w_new, L_new, g_new
    #         lr *= 2
    #     else
    #         lr /= 2
    #     end

    #     if size(wt)[begin] > 1
    #         diff = norm(wt[end, :] - wt[end-1, :])
    #     end
    # end
    # model.w = w
    # return w, wt, Lt

function model_predict(model::Log_reg, tst::DataFrame)
    return replace(map(sign, Matrix(tst) * model.w), -1 => 0)
end
logistic_loss(X, y, w) = length(y)^-1 * sum(log.(1 .+ exp.(-y .* (X' * w))))
logistic_loss_grad(X, y, w) = -length(y)^-1 * ((1 ./ (1 .+ exp.(y .* (X' * w))))' * (X' .* y))'


mutable struct Neural_network <: Model
    args::Any
    m::Any
    Neural_network(; args = Args(), m = nothing) = new(args, m)
end

@with_kw mutable struct Args
    η::Float64 = 3e-4       # learning rate
    batchsize::Int = 64   # batch size
    epochs::Int = 10        # number of epochs
    device::Function = cpu  # set as gpu, if gpu available
end


function model_fit(model::Neural_network, X::DataFrame, y::AbstractArray)
    X[!, :Target] = y
    trn, val = random_split(X, [0.6, 0.4])
    batched_trn = batch(Matrix(trn[!, Not(:Target)])', trn[!, :Target], model.args)
    batched_val = batch(Matrix(val[!, Not(:Target)])', val[!, :Target], model.args)
    model.m = build_model(length(names(X)) - 1)
    train(model, batched_trn, batched_val, model.args)
end

function model_predict(model::Neural_network, tst::DataFrame)
    batched_tst = batch(Matrix(tst)', zeros(nrow(tst)),model.args)
    predictions = []
    for (x, y) in batched_tst
        append!(predictions, onecold(cpu(model.m(x))))
    end
    return predictions .- 1
end


function model_predict(model::Neural_network, tst::DataLoader)
    predictions = []
    for (x, y) in tst
        append!(predictions, onecold(cpu(model.m(x))))
    end
    return predictions .- 1
end

function batch(X, y, args; shuffle = false)
    y = onehotbatch(y, 0:1)
    batched_data = DataLoader((X, y), batchsize = args.batchsize, shuffle = shuffle)
    return batched_data
end

function build_model(in_size; nclasses = 2)
    return Chain(Dense(in_size, 500, relu), Dense(500, 32, relu), Dense(32, nclasses))
end


function loss_all(dataloader, model)
    mean(map(pair -> logitcrossentropy(model(pair[begin]), pair[end], dims=1), dataloader))
end


function nn_accuracy(dataloader, model)
    acc = 0
    for (x, y) in dataloader
        acc += sum(onecold(cpu(model(x))) .== onecold(cpu(y))) * 1 / size(x, 2)
    end
    acc / length(dataloader)
end


function train(model, train_data, val_data, args)
    m = model.m
    train_data = args.device.(train_data)
    val_data = args.device.(val_data)
    m = args.device(m)
    loss(x, y) = logitcrossentropy(m(x), y)
    opt = AdamW(args.η)
    for epoch in 1:args.epochs
        Flux.train!(loss, Flux.params(m), train_data, opt)
        @show epoch
        @show(loss_all(train_data, m))
        @show accuracy_my(model_predict(model, val_data), val_data)
    end
    return m
end

mutable struct Decision_tree <: Model
    max_depth::Any
    criterion::Any
    root::Any
    Decision_tree(; max_depth = 3, criterion = gini, root = nothing) =
        new(max_depth, criterion, root)
end

mutable struct Node
    left::Any
    right::Any
    depth::Any
    decision_function::Any
    feature_name::Any
    threshold::Any
    label::Any
    Node(;
        depth = 0,
        left = nothing,
        right = nothing,
        decision_function = identity,
        feature_name = nothing,
        label = nothing,
        threshold = nothing,
    ) = new(left, right, depth, decision_function, feature_name, threshold, label)
end


function model_fit(model::Decision_tree, X::DataFrame, y::AbstractArray)
    model.root = build_tree(X, y; max_depth = model.max_depth, criterion = model.criterion)
end


function model_predict(model::Decision_tree, tst::DataFrame)
    predictions = []
    for entry in eachrow(tst)
        root = model.root
        label = root.label
        while !isnothing(root)
            feature_split = threshold_decision(entry[root.feature_name], root.threshold)
            label = root.label
            root = feature_split == true ? root.right : root.left
        end
        push!(predictions, label)
    end
    return predictions
end

function build_tree(X, y; max_depth = 4, criterion = gini, cur_depth = 0)
    cur_depth >= max_depth && return nothing
    left_split, left_labels, right_split, right_labels, feature_name, gain, t =
        split_by_best_feature(X, y)
    isnothing(left_labels) && return nothing
    isnothing(right_labels) && return nothing

    root = Node(;
        depth = cur_depth,
        label = most_common(y),
        threshold = t,
        feature_name = feature_name,
        left = build_tree(
            left_split,
            left_labels;
            max_depth = max_depth,
            criterion = criterion,
            cur_depth = cur_depth + 1,
        ),
        right = build_tree(
            right_split,
            right_labels;
            max_depth = max_depth,
            criterion = criterion,
            cur_depth = cur_depth + 1,
        ),
    )
    return root
end

function split_by_best_feature(X::DataFrame, y; criterion = gini)
    original_entropy = criterion(y)
    best_gain = 0
    best_feature = nothing
    best_left = nothing
    best_right = nothing
    best_left_labels = nothing
    best_right_labels = nothing
    best_t = nothing
    for feature_name in names(X)
        feature = X[!, feature_name]
        values = sort(unique(feature))
        thresholds = [(values[i-1] + values[i]) / 2 for i = 2:length(values)]
        for t in thresholds
            feature_split = map(entry -> entry >= t, feature)
            left, right = y[feature_split], y[Not(feature_split)]
            (isempty(left) || isempty(right)) && continue

            if (
                cur_gain =
                    original_entropy -
                    split_entropy_sum(y, [y[feature_split], y[.~feature_split]], criterion)
            ) >= best_gain
                best_left = X[feature_split, :]
                best_left_labels = y[feature_split]
                best_right = X[.~feature_split, :]
                best_right_labels = y[.~feature_split]
                best_gain = cur_gain
                best_feature = feature_name
                best_t = t
            end
        end
    end

    best_left,
    best_left_labels,
    best_right,
    best_right_labels,
    best_feature,
    best_gain,
    best_t
end

threshold_decision(data, t) = map(entry -> entry >= t, data)
information_gain(root, children, criterion) = criterion(root) - sum(criterion.(children))
split_entropy_sum(parent, children, criterion) =
    sum(classes_rate(parent, children) .* criterion.(children))
gini(data) = 1 - sum(classes_rate(data) .^ 2)
entropy(data) = -sum(classes_rate(data) .* log.(classes_rate(data)))
classes_rate(data) =
    map(class_count -> class_count / length(data), values(sort(countmap(data))))
classes_rate(parent, children) = length.(children) ./ length(parent)
