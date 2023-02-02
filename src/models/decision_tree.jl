export Decision_tree, gini, entropy_local

"""
    Node(; depth = 0, left = nothing, right = nothing, decision_function = identity, feature_name = nothing, label = nothing, threshold = nothing)

Decision node.
"""
mutable struct Node
    left::Union{Node, Nothing}
    right::Union{Node, Nothing}
    depth::Int
    decision_function::Function
    feature_name::Union{String, Symbol}
    threshold::Number
    label::Int
    function Node(;
                  depth = 0,
                  left = nothing,
                  right = nothing,
                  decision_function = identity,
                  feature_name = nothing,
                  label = nothing,
                  threshold = nothing)
        new(left, right, depth, decision_function, feature_name, threshold, label)
    end
end

"""
    Decision_tree(; max_depth = 3, criterion = gini)

Binary decision tree implementation.
"""
mutable struct Decision_tree <: Model
    max_depth::Int
    criterion::Function
    root::Union{Node, Nothing}
    function Decision_tree(; max_depth = 3, criterion = gini)
        new(max_depth, criterion, nothing)
    end
end

"""
    model_fit!(model, X, y)

Fits the model `model` to training data.
"""
function model_fit!(model::Decision_tree, X::DataFrame, y::AbstractArray)
    model.root = build_tree(X, y; max_depth = model.max_depth, criterion = model.criterion)
end

"""
    model_predict(model, tst)

Predicts the output for test data `tst` using the `model`.
"""
function model_predict(model::Decision_tree, tst::DataFrame)
    predictions = []
    for entry in eachrow(tst)
        root = model.root
        label = root.label
        while !isnothing(root)
            feature_split = threshold_decision(entry[root.feature_name], root.threshold)
            label = root.label
            root = feature_split == true ? root.left : root.right
        end
        predictions = vcat(predictions, label)
    end
    predictions
end

"""
    build_tree(X, y; max_depth = 4, criterion = gini, cur_depth = 0)

Builds the decision tree and returns the root.
"""
function build_tree(X::DataFrame, y::AbstractArray; max_depth = 4, criterion = gini,
                    cur_depth = 0)
    cur_depth >= max_depth && return nothing
    left_split, left_labels, right_split, right_labels, feature_name, gain, t = split_by_best_feature(X,
                                                                                                      y)
    isnothing(left_labels) && return nothing
    isnothing(right_labels) && return nothing

    root = Node(;
                depth = cur_depth,
                label = most_common(y),
                threshold = t,
                feature_name = feature_name,
                left = build_tree(left_split,
                                  left_labels;
                                  max_depth = max_depth,
                                  criterion = criterion,
                                  cur_depth = cur_depth + 1),
                right = build_tree(right_split,
                                   right_labels;
                                   max_depth = max_depth,
                                   criterion = criterion,
                                   cur_depth = cur_depth + 1))
    root
end

"""
    function split_by_best_feature(X, y; criterion = gini)

Splits the data by the feature and threshold yielding the highest information gain.
"""
function split_by_best_feature(X::DataFrame, y::AbstractArray; criterion = gini)
    original_entropy = criterion(y)
    best_gain = 0
    best_feature = nothing
    best_left = nothing
    best_right = nothing
    best_left_labels = nothing
    best_right_labels = nothing
    best_t = nothing
    for feature_name in names(X)
        feature = X[!, feature_name]
        values = sort(unique(feature))
        thresholds = [(values[i - 1] + values[i]) / 2 for i in 2:length(values)]
        for t in thresholds
            feature_split = map(entry -> entry >= t, feature)
            left, right = y[feature_split], y[Not(feature_split)]
            (isempty(left) || isempty(right)) && continue

            if (cur_gain = original_entropy -
                           split_entropy_sum(y, [y[feature_split], y[.~feature_split]],
                                             criterion)) >= best_gain
                best_left = X[feature_split, :]
                best_left_labels = y[feature_split]
                best_right = X[.~feature_split, :]
                best_right_labels = y[.~feature_split]
                best_gain = cur_gain
                best_feature = feature_name
                best_t = t
            end
        end
    end

    best_left,
    best_left_labels,
    best_right,
    best_right_labels,
    best_feature,
    best_gain,
    best_t
end

threshold_decision(data::Real, t::Real) = data >= t
threshold_decision(data::AbstractArray, t::Real) = map(entry -> entry .>= t, data)
function information_gain(root::AbstractArray, children::AbstractArray, criterion::Function)
    criterion(root) - sum(criterion.(children))
end
function split_entropy_sum(parent::AbstractArray, children::AbstractArray,
                           criterion::Function)
    sum(classes_rate(parent, children) .* criterion.(children))
end
gini(data::AbstractArray) = 1 - sum(classes_rate(data) .^ 2)
entropy_local(data::AbstractArray) = -sum(classes_rate(data) .* log.(classes_rate(data)))
classes_rate(data::AbstractArray) = values(sort(countmap(data))) ./ length(data)
function classes_rate(parent::AbstractArray, children::AbstractArray)
    length.(children) ./ length(parent)
end
