using Revise
using Titanic
using DataFrames
using DataStructures
using Statistics
using StatsBase

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


dt = Decision_tree()
dt
dt.root.left

trn, val, tst = random_split(df, [0.6, 0.2, 0.2])
y = trn[!, :Survived]
X = Matrix(Matrix(trn[!, Not(:Survived)])')
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