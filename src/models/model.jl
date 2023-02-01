export model_predict, model_fit!

abstract type Model end

@doc """
    model_fit!(model, X, y)
    
trains the model on the provided data
"""
function model_fit! end
@doc """
    model_predict(model, X)
    
makes a prediction on the data using the provided model
"""
function model_predict end

include("k_nn.jl")
include("log_reg.jl")
include("neural_network.jl")
include("decision_tree.jl")
