module Titanic

using CSV
using DataFrames
using Statistics

export read_csv_data, replace_in_cols!, replace_in_cols, apply_to_cols, to_categorical, standartize

include("utils.jl")

end
