export Neural_network, Args, batch
"""
    Neural_network(; args = Args(), m = nothing)

The `Neural_network` type. The user can pass in arguments of type `Args` and a custom model `m` (which is set to `nothing` by default).
"""
mutable struct Neural_network <: Model
    args::Any
    m::Any
    Neural_network(; args = Args(), m = nothing) = new(args, m)
end

@with_kw mutable struct Args
    lr::Float64 = 3e-4
    batchsize::Int = 64
    epochs::Int = 30
    ratios::AbstractArray = [0.8, 0.2]
    device::Function = cpu
end

"""
    model_fit!(model, X, y, X_val, y_val; verbose = false)

Fits the model `model` to training data `X` and `y` and validation data `X_val` and `y_val`.
"""
function model_fit!(model::Neural_network,
                    X::DataFrame,
                    y::AbstractArray,
                    X_val::DataFrame,
                    y_val::AbstractArray;
                    verbose = false)
    batched_trn = batch(Matrix(X)', y, model.args; shuffle = true)
    batched_val = batch(Matrix(X_val)', y_val, model.args)
    isnothing(model.m) && (model.m = build_model(length(names(X))))
    train!(model, batched_trn, batched_val, model.args; verbose = verbose)
end
"""
    model_fit!(model, X, y; verbose = false)

Fits the `model` to training data `X` and `y` by randomly splitting the data into training and validation sets using the `ratios` specified in `args` of the `model`.
"""
function model_fit!(model::Neural_network, X::DataFrame, y::AbstractArray; verbose = false)
    X_local = insertcols(X, :Target => y)
    trn, val = random_split(X_local, model.args.ratios)
    model_fit!(model,
               trn[!, Not(:Target)],
               trn[!, :Target],
               val[!, Not(:Target)],
               val[!, :Target];
               verbose = verbose)
end

"""
    model_predict(model, tst)

Predicts the output for test data `tst` using the `model`.
"""
function model_predict(model::Neural_network, tst::DataFrame)
    batched_tst = batch(Matrix(tst)', zeros(nrow(tst)), model.args)
    model_predict(model, batched_tst)
end

"""
    model_predict(model, tst)

Predicts the output for test data `tst` using the `model`.
"""
function model_predict(model::Neural_network, tst::Any)
    predictions = []
    for (x, y) in tst
        predictions = vcat(predictions, onecold(cpu(model.m(x))))
    end
    predictions .- 1
end

"""
    batch(X, y, args; shuffle = false)

Batches the input data `X` and labels `y` into a `DataLoader` object. If `shuffle` is `false` (default), the data is not shuffled, otherwise it is shuffled. The batch size is determined by `batchsize` in `args`.
"""
function batch(X::AbstractMatrix, y::AbstractArray, args; shuffle = false)
    y_onehot = onehotbatch(y, 0:1)
    DataLoader((X, y_onehot), batchsize = args.batchsize, shuffle = shuffle)
end

"""
    build_model(in_size; nclasses = 2)

Builds a neural network with a pre-defined simple architecture.
"""
function build_model(in_size::Int; nclasses = 2)
    Chain(Dense(in_size, 500, relu), Dense(500, 32, relu), Dense(32, nclasses))
end

"""
    loss_all(dataloader, model)

Calculates the cross-entropy loss.
"""
function loss_all(dataloader::DataLoader, model::Neural_network)
    mean(map(pair -> logitcrossentropy(model.m(pair[begin]), pair[end], dims = 1),
             dataloader))
end
"""
    labels_from_batched(batched_data::DataLoader)

Extracts the labels from the `DataLoader` object.
"""
function labels_from_batched(batched_data::DataLoader)
    collect(Iterators.flatten(map(pair -> onecold(pair[end]), batched_data))) .- 1
end

"""
    train!(model, train_data, val_data, args; verbose = false)

Trains the neural network on the training data.
If `verbose` is `false` (default), the function runs silently, otherwise it outputs the loss and accuracy for each epoch.
"""
function train!(model::Neural_network, train_data::DataLoader, val_data::DataLoader, args::Args; verbose = false)
    m = model.m
    train_data = args.device.(train_data)
    val_data = args.device.(val_data)
    m = args.device(m)
    loss(x, y) = logitcrossentropy(m(x), y)
    opt = AdamW(args.lr)
    for epoch in 1:(args.epochs)
        Flux.train!(loss, Flux.params(m), train_data, opt)
        verbose || continue
        println("epoch: $(epoch)")
        println("loss: $(loss_all(train_data, model))")
        println("val accuracy: $(accuracy(model_predict(model, val_data), labels_from_batched(val_data)))")
    end
    m
end
