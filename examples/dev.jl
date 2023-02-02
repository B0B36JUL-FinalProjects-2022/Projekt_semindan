1 + 1
using Revise
using Titanic
using DataFrames
using Statistics
using StatsBase
using Flux
using Flux, Statistics
using Flux.Data: DataLoader
using Flux: onehotbatch, onecold, logitcrossentropy, throttle, @epochs

df = read_csv_data("data/train.csv")

col_median = apply_to_cols(df, :Age, median)
df = replace_in_cols(df, :Age, missing, col_median)
df
col_city = apply_to_cols(df, :Embarked, most_common)
df = replace_in_cols(df, :Embarked, missing, col_city)

df = replace_in_cols(df, :Cabin, missing, "N")
col_cabin = apply_to_cols(df, :Cabin, strip_cabin_numbers)
df[!, :Cabin] = col_cabin
groups = get_title_groups()
df = replace_names_with_title_categories(df, groups)
df = categorize(df)
df = df[!, Not(:PassengerId)]
survived = df[!, :Survived]
df = standartize(df)
df.Survived = survived

trn, val, tst = random_split(df, [0.6, 0.2, 0.2])
y = trn[!, :Survived]
X = trn[!, Not(:Survived)]
tst_y = tst[!, :Survived]
tst_X = tst[!, Not(:Survived)]

dt = Decision_tree(max_depth = 15)
model_fit!(dt, X, y)
preds = model_predict(dt, tst_X)
println(accuracy(tst[!, :Survived], preds))

nn = Neural_network()
model_fit!(nn, X, y)
preds = model_predict(nn, tst_X)
println(accuracy(tst[!, :Survived], preds))

knn = K_nn(; metric = Titanic.lmax)
model_fit!(knn, X, y)
preds = model_predict(knn, tst_X)
println(accuracy(tst[!, :Survived], preds))

logreg = Log_reg()
model_fit!(logreg, X, y)
preds = model_predict(logreg, tst_X)
println(accuracy(tst[!, :Survived], preds))
