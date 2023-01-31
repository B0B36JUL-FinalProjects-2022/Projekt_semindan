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

col_city = apply_to_cols(df, :Embarked, most_common)
df = replace_in_cols(df, :Embarked, missing, col_city)

df = replace_in_cols(df, :Cabin, missing, "N") 
col_cabin = apply_to_cols(df, :Cabin, strip_cabin_numbers)
df[!, :Cabin] = col_cabin

groups = get_title_groups()
df = replace_names_with_title_categories(df, groups)

df = categorize(df)

survived = df[!, :Survived]
df = standartize(df)
df.Survived = survived

dt = Decision_tree(max_depth=5)

trn, val, tst = random_split(df, [0.6, 0.2, 0.2])
y = trn[!, :Survived]
X = trn[!, Not(:Survived)]
# X = Matrix(Matrix(trn[!, Not(:Survived)])')

tst_y = tst[!, :Survived]
tst_X = tst[!, Not(:Survived)]

nn = Neural_network()

# y[y .== -1] .= 0

X[!, :Target] = y
trn, val = random_split(X, [0.6, 0.4])
batched_trn = batch(Matrix(trn[!, Not(:Target)])', trn[!, :Target], nn.args)
batched_trn
trn

for (x,y) in batched_trn
    println(onecold(y))
    break
end

collect(Iterators.flatten(map(pair -> onecold(pair[end]), batched_trn)))

model_fit(nn, X, y)
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
accuracy_my(tst[!, :Survived], preds)
tst


dt.root.left.left.label
dt.root.label
orig_entropy = gini(y)
left, left_labels, right, right_labels, feature, gain, t = build_node(X, y, orig_entropy)
left
left_labels
right
right_labels
y
X
Matrix(Matrix(X)')
Matrix(Matrix(X)')
gain
unique(trn[!, :Age])

onehot(trn[!, :Sex], unique(trn[!, :Sex]))

model_fit(dt, X, y)
dt.root.

left, right, feature, gain = build_node(df)
countmap(a)

X = df

    root = X[!, :Survived]
    best_gain = -1
    best_feature = nothing
    best_children = []
    for feature in eachcol(X)
        left, right = split_by_function(feature, feature_func)
        if (cur_gain = information_gain(root, [left, right], criterion)) > best_gain
            best_children = [left, right]
            best_gain = cur_gain
            best_feature = feature
        end
    end
    best_children[begin], best_children[end]


data=df
col= :Name

i = 1
class = 0
function to_onehot(data, col)
    classes = unique(data[!, col])
    data_onehot = falses(nrow(data), length(unique(data[!, col])))
    for (i, class) in enumerate(classes)
        data_onehot[data[!, col] .== class, i] .= 1
        data[!, string(col, "_", class)] = data_onehot[:, i]
    end
    return data
end
length(unique(df[!, :Name]))
nrow(df)
to_onehot(df, :Name)
dff = DataFrame(b = zeros(891))
hcat(df, dff)
df[!, Not([:a, :Name, :Age])]
