using Revise
using Titanic
using DataFrames
using DataStructures
using Statistics
using StatsBase

df = read_csv_data("data/train.csv")
trn, val, tst = random_split(df, [0.5, 0.3, 0.2])
knn = K_nn(;n=2, metric="l2")


