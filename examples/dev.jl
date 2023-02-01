1+1
using Revise
using Titanic
using DataFrames
using DataStructures
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
df = to_onehot(df; remove_original=true)
survived = df[!, :Survived]
df = standartize(df)
df.Survived = survived

dt = Decision_tree(max_depth=15)

trn, val, tst = random_split(df, [0.6, 0.2, 0.2])
y = trn[!, :Survived]
X = trn[!, Not(:Survived)]
# X = Matrix(Matrix(trn[!, Not(:Survived)])')

tst_y = tst[!, :Survived]
tst_X = tst[!, Not(:Survived)]

model_fit!(dt, X, y)
preds = model_predict(dt, tst_X)

nn = Neural_network()

# y[y .== -1] .= 0

X[!, :Target] = y
trn, val = random_split(X, [0.6, 0.4])
batched_trn = batch(Matrix(trn[!, Not(:Target)])', trn[!, :Target], nn.args)
batched_trn

collect(Iterators.flatten(map(pair -> onecold(pair[end]), batched_trn)))

model_fit!(nn, X, y)
preds = model_predict(nn, tst_X)
preds = model_predict(knn, tst_X)

knn = K_nn()
model_fit(knn, X, y)
preds = model_predict(knn, tst_X)
knn.train_data_X
countmap(preds)
countmap(y)

logreg = Log_reg()
model_fit(logreg, X, y)
preds = model_predict(logreg, tst_X)
preds[preds .== -1] .= 0

dt
model_fit(dt, X, y)
dt.root.feature_name
tst
tst_X = Matrix(Matrix(tst[!, Not(:Survived)])')
preds = model_predict(dt, tst)
accuracy(tst[!, :Survived], preds)
preds = model_predict(dt, X)
accuracy(trn[!, :Survived], preds)
tst

