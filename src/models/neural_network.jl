export Neural_network, Args, batch

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

function model_fit!(model::Neural_network,
                    X::DataFrame,
                    y::AbstractArray,
                    X_val::DataFrame,
                    y_val::AbstractArray;
                    verbose = false)
    batched_trn = batch(Matrix(X)', y, model.args)
    batched_val = batch(Matrix(X_val)', y_val, model.args)
    isnothing(model.m) && (model.m = build_model(length(names(X))))
    train(model, batched_trn, batched_val, model.args; verbose = verbose)
end

function model_fit!(model::Neural_network, X::DataFrame, y::AbstractArray; verbose = false)
    X[!, :Target] = y
    trn, val = random_split(X, model.ratios)
    model_fit!(model,
               trn[!, Not(:Target)],
               trn[!, :Target],
               val[!, Not(:Target)],
               val[!, :Target];
               verbose = verbose)
end

function model_predict(model::Neural_network, tst::DataFrame)
    batched_tst = batch(Matrix(tst)', zeros(nrow(tst)), model.args)
    model_predict(model, batched_tst)
end

function model_predict(model::Neural_network, tst)
    predictions = []
    for (x, y) in tst
        predictions = vcat(predictions, onecold(cpu(model.m(x))))
    end
    predictions .- 1
end

function batch(X, y, args; shuffle = false)
    y = onehotbatch(y, 0:1)
    DataLoader((X, y), batchsize = args.batchsize, shuffle = shuffle)
end

function build_model(in_size; nclasses = 2)
    Chain(Dense(in_size, 500, relu), Dense(500, 32, relu), Dense(32, nclasses))
end

function loss_all(dataloader, model)
    mean(map(pair -> logitcrossentropy(model(pair[begin]), pair[end], dims = 1),
             dataloader))
end

function labels_from_batched(batched_data)
    collect(Iterators.flatten(map(pair -> onecold(pair[end]), batched_data))) .- 1
end

function train(model, train_data, val_data, args; verbose = false)
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
        println("loss: $(loss_all(train_data, m))")
        println("val accuracy: $(accuracy(model_predict(model, val_data), labels_from_batched(val_data)))")
    end
    m
end
