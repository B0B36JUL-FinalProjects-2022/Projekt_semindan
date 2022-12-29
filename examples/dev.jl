using Revise
using Titanic
using DataFrames
using DataStructures
using Statistics
using StatsBase

df = read_csv_data("data/train.csv")
knn = K_nn(;n=2, metric="l2")
col_median = apply_to_cols(df, :Age, median)
df = replace_in_cols(df, :Age, missing, col_median)
trn, val, tst = random_split(df, [0.5, 0.3, 0.2])
val_X, val_y = val[!, [:Fare, :Age]], val[!, :Survived]
Matrix(val_X)
val_y
logistic_loss_grad(Matrix(val_X)', val_y, [1.0,1.0])
X = Matrix(val_X)
y = val_y
w = [1.0, 1.0]


X = [[1 1 1];[1 2 3]]
y = [1, -1, -1]
w = [1.5, -0.5]
-length(y)^-1 * ((1 ./ (1 .+ exp.(y .* (X' * w))))' * (X' .*  y))'








logreg = Log_reg()
Matrix(val_X)

trn_X, trn_y = trn[!, [:Fare, :Age]], trn[!, :Survived]
w, wt = model_fit(logreg, trn_X, trn_y)
w
val_y
accuracy(model_predict(logreg, val_X), val_y)