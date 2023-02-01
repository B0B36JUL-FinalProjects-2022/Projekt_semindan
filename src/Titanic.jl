module Titanic

using CSV
using DataFrames
using Statistics
using StatsBase
using StatsPlots
using Random
using LinearAlgebra
using Flux
using Flux, Statistics
using Flux.Data: DataLoader
using Flux: onehotbatch, onecold, logitcrossentropy, throttle, @epochs
using Base.Iterators: repeated
using Parameters: @with_kw

include("utils.jl")
include("models/model.jl")
end
