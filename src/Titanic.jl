module Titanic

using CSV
using DataFrames
using Statistics
using StatsBase
using StatsPlots
using Random
using LinearAlgebra
using Flux
using Flux.Data: DataLoader
using Flux: onehotbatch, onecold, logitcrossentropy
using Parameters: @with_kw

include("utils.jl")
include("models/model.jl")
end
