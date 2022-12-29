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

export read_csv_data,
       replace_in_cols!,
       replace_in_cols,
       apply_to_cols,
       categorize,
       standartize,
       most_common,
       strip_cabin_numbers,
       get_title_groups,
       replace_names_with_title_categories,
       random_split,
       K_nn,
       model_predict,
       model_fit,
       metric_dist,
       accuracy_my,
       Log_reg,
       logistic_loss,
       logistic_loss_grad,
       train,
       to_batches,
       Args

include("utils.jl")
include("models.jl")

end
