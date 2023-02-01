export Decision_tree, gini, entropy_local

mutable struct Decision_tree <: Model
    max_depth::Any
    criterion::Any
    root::Any
    function Decision_tree(; max_depth = 3, criterion = gini, root = nothing)
        new(max_depth, criterion, root)
    end
end

mutable struct Node
    left::Any
    right::Any
    depth::Any
    decision_function::Any
    feature_name::Any
    threshold::Any
    label::Any
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

function model_fit!(model::Decision_tree, X::DataFrame, y::AbstractArray)
    model.root = build_tree(X, y; max_depth = model.max_depth, criterion = model.criterion)
end

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

function build_tree(X, y; max_depth = 4, criterion = gini, cur_depth = 0)
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

function split_by_best_feature(X::DataFrame, y; criterion = gini)
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

threshold_decision(data, t) = map(entry -> entry .>= t, data)
information_gain(root, children, criterion) = criterion(root) - sum(criterion.(children))
function split_entropy_sum(parent, children, criterion)
    sum(classes_rate(parent, children) .* criterion.(children))
end
gini(data) = 1 - sum(classes_rate(data) .^ 2)
entropy_local(data) = -sum(classes_rate(data) .* log.(classes_rate(data)))
classes_rate(data) = values(sort(countmap(data))) ./ length(data)
classes_rate(parent, children) = length.(children) ./ length(parent)
