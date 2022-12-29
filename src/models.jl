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


mutable struct Log_reg <: Model
    lr
    max_iter
    w
    Log_reg(;lr=0.1, max_iter=10000) = new(lr, max_iter)
end


model_fit(model::Log_reg, X::DataFrame, y::AbstractArray, w_init; args...) = model_fit(model, Matrix(Matrix(X)'), y, w_init)
function model_fit(model::Log_reg, X::Matrix, y::AbstractArray, w_init; max_iter = 10000, lr = 1.0, epsilon=1e-08)
    y = replace(y, 0 => -1)
    w, L, g = w_init, logistic_loss(X, y, w_init), logistic_loss_grad(X, y, w_init)
    wt, Lt = Matrix(w'), [L]
    diff = Inf
    for i in 1:max_iter
        diff <= epsilon && break
        w_new = w - lr * g
        L_new, g_new = logistic_loss(X, y, w_new), logistic_loss_grad(X, y, w_new)
        if L_new < L
            wt = vcat(wt, w_new')
            Lt = vcat(Lt, L_new)
            w, L, g = w_new, L_new, g_new
            lr *= 2
        else
            lr /= 2
        end

        if size(wt)[begin] > 1
            diff = norm(wt[end, :] - wt[end-1, :])
        end
    end    
    model.w = w   
    return w, wt, Lt
end


function model_predict(model::Log_reg, tst::DataFrame) 
    return map(entry -> entry > 0 ? 1 : 0, Matrix(tst) * model.w)
end

logistic_loss(X, y, w) = length(y)^-1 * sum(log.(1 .+ exp.(-y .* (X' * w))))
logistic_loss_grad(X, y, w) = -length(y)^-1 * ((1 ./ (1 .+ exp.(y .* (X' * w))))' * (X' .*  y))'



@with_kw mutable struct Args
    η::Float64 = 3e-4       # learning rate
    batchsize::Int = 64   # batch size
    epochs::Int = 10        # number of epochs
    device::Function = gpu  # set as gpu, if gpu available
end

function to_batches(trn::DataFrame, val::DataFrame, tst::DataFrame, args)
    ENV["DATADEPS_ALWAYS_ACCEPT"] = "true"
	
    trn_X, trn_y = Matrix(trn[!, Not(:Survived)])', trn[!, :Survived]
    val_X, val_y = Matrix(val[!, Not(:Survived)])', val[!, :Survived]
    tst_X, tst_y = Matrix(tst[!, Not(:Survived)])', tst[!, :Survived]

    trn_y, val_y, tst_y = onehotbatch(trn_y, 0:1), onehotbatch(val_y, 0:1),onehotbatch(tst_y, 0:1) 

    trn_data = DataLoader((trn_X, trn_y), batchsize=args.batchsize, shuffle=true)
    val_data = DataLoader((val_X, val_y), batchsize=args.batchsize)
    tst_data = DataLoader((tst_X, tst_y), batchsize=args.batchsize)

    return trn_data, val_data, tst_data
end

function build_model(in_size ; nclasses=2)
    return Chain(
 	    Dense(in_size, 500, relu),
        Dense(500, 32, relu),
            Dense(32, nclasses))
end


function loss_all(dataloader, model)
    l = 0f0
    for (x,y) in dataloader
        l += logitcrossentropy(model(x), y, dims=1)
    end
    l/length(dataloader)
end


function accuracy(data_loader, model)
    acc = 0
    for (x,y) in data_loader
        acc += sum(onecold(cpu(model(x))) .== onecold(cpu(y)))*1 / size(x,2)
    end
    acc/length(data_loader)
end


function train(train_data, val_data; kws...)
    # Initializing Model parameters 
    args = Args(; kws...)

    # Load Data


    # # Construct model
    m = build_model(9)
    train_data = args.device.(train_data)
    val_data = args.device.(val_data)
    m = args.device(m)
    loss(x,y) = logitcrossentropy(m(x), y)
    loss(train_data[1][1], train_data[1][2])
    evalcb = () -> ()
    # @show(loss_all(train_data, m))
    opt = ADAM(args.η)
		
    # @epochs args.epochs 
    Flux.train!(loss, Flux.params(m), train_data, opt, cb = evalcb)

    @show accuracy(train_data, m)

    @show accuracy(val_data, m)

    return m
end