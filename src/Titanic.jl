module Titanic

using CSV
using DataFrames
using Statistics
using StatsBase
using Random
using LinearAlgebra

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
       accuracy

include("utils.jl")
include("models.jl")

end
