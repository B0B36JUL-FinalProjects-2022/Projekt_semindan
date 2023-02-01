export model_predict, model_fit!

abstract type Model end

"""
    model_fit!(model, X, y)
    
Trains the model on the provided data.
"""
function model_fit! end

"""
    model_predict(model, X)
    
Makes a prediction on the data using the provided model.
"""
function model_predict end

include("k_nn.jl")
include("log_reg.jl")
include("neural_network.jl")
include("decision_tree.jl")
