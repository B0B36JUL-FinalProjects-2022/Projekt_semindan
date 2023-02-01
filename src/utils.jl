
@doc """
    read_csv_data(name)

reads a csv file and returns a DataFrame
"""
function read_csv_data(name::String)
    ispath(name) || throw(ErrorException("Invalid data path"))
    return CSV.read(name, DataFrame; header = true)
end

replace_eq(value::Any, src::Missing, dst::Any) = ismissing(value) ? dst : value # coalesce
replace_eq(value::Any, src::Any, dst::Any) = value == src ? dst : value
replace_eq(value::Any, src::AbstractArray, dst::Any) =
    !isnothing(findfirst(entry -> entry == value, src)) ? dst : value # many-to-one 
replace_eq(value::Any, src::AbstractArray, dst::AbstractArray) =
    !isnothing(findfirst(entry -> entry == value, src)) ?
    dst[findfirst(entry -> entry == value, src)] : value # many-to-many

@doc """
    replace_in_cols(df, cols, src, dst)

replaces entries in the DataFrame columns, src values are replaced with dst values
"""
function replace_in_cols(
    df::DataFrame,
    cols::Union{String,Symbol,Vector{<:Union{String,Symbol}}},
    src::Any,
    dst::Any,
)
    return transform(df, cols .=> ByRow(entry -> replace_eq(entry, src, dst)) .=> cols)
end

function replace_in_cols!(
    df::DataFrame,
    cols::Union{String,Symbol,Vector{<:Union{String,Symbol}}},
    src::Any,
    dst::Any,
)
    return transform!(df, cols .=> ByRow(entry -> replace_eq(entry, src, dst)) .=> cols)
end

function replace_in_cols(df::DataFrame, src::Any, dst::Any)
    return replace_in_cols(df, names(df), src, dst)
end
function replace_in_cols!(df::DataFrame, src::Any, dst::Any)
    return replace_in_cols!(df, names(df), src, dst)
end

@doc """
    categorize(df, cols)

categorizes the chosen columns in a DataFrame
"""
function categorize(df::DataFrame, cols::Vector{<:Union{String,Symbol}})
    return transform(
        df,
        cols .=> (col -> map(entry -> col_to_categorical(col, entry), col)) .=> cols,
    )
end

@doc """
    categorize(df)

categorizes all columns in a DataFrame
"""
categorize(df::DataFrame) = categorize(df, names(df))
categorize(df::DataFrame, cols::Union{Symbol,String}) = categorize(df, [cols])
function col_to_categorical(col::AbstractArray, value::Any)
    return findfirst(entry -> entry == value, unique(col)) - 1
end
col_to_categorical(col::AbstractArray, value::Number) = value


@doc """
    standartize(df, cols)

normalizes the chosen columns in a DataFrame
"""
function standartize(df::DataFrame, cols::Vector{<:Union{String,Symbol}})
    return transform(
        df,
        cols .=> (col -> map(entry -> col_to_norm(col, entry), col)) .=> cols,
    )
end

col_to_norm(col::AbstractArray{<:Number}, value::Number) = (value - mean(col)) / std(col)

@doc """
    standartize(df)

standartizes all columns in a DataFrame
"""
standartize(df::DataFrame) = standartize(df, names(df))
standartize(df::DataFrame, cols::Union{Symbol,String}) = standartize(df, [cols])

@doc """
    apply_to_cols(df, cols, func)

applies a function to the chosen columns in a DataFrame
"""
function apply_to_cols(df::DataFrame, cols::Union{Symbol,String}, func::Function)
    return func(collect(skipmissing(df[!, cols])))
end
function apply_to_cols(df::DataFrame, cols::Vector{<:Union{String,Symbol}}, func::Function)
    return map(col -> apply_to_cols(df, col, func), cols)
end # we DO NOT have Point{Float64} <: Point{Real}
apply_to_cols(df::DataFrame, func::Function) = apply_to_cols(df, names(df), func)

most_common(arr::AbstractArray) = argmax(countmap(arr))

@doc  """
    strip_cabin_numbers(arr)

given an array of strings, where a string represents a cabin number, removes the string's content past the first character
"""
strip_cabin_numbers(arr::AbstractArray{<:AbstractString}) = map(entry -> entry[begin], arr)

@doc """
    get_title_groups()

groups names based on title
"""
function get_title_groups()
    default_titles = ["Mr.", "Mrs.", "Mlle.", "Miss.", "Mme.", "Ms."]
    royal_titles = ["Master.", "Don.", "Lady.", "Sir.", "Countess.", "Jonkheer."]
    other_titles = ["Major.", "Col.", "Capt.", "Dr.", "Rev."]

    return [(default_titles, "D"), (royal_titles, "R"), (other_titles, "O")]
end


function replace_names_with_title_categories(df, groups)
    for group in groups
        transform!(
            df,
            :Name =>
                ByRow(entry -> any(occursin.(group[begin], entry)) ? group[end] : entry) =>
                    :Name,
        )
    end
    return df
end

@doc """
    random_split(df, ratios; seed = 42)
    
creates an array with data splits, where sizes of the splits are determined by their ratios w.r.t the original data size
"""
function random_split(df::DataFrame, ratios::AbstractArray{<:Number}; seed = 42)
    isempty(ratios) && throw(ErrorException("Array of ratios is empty"))
    (all(0.0 .<= ratios .<= 1.0) &&
        sum(ratios) == 1.0) ||
        throw(ErrorException("Invalid size ratios"))
    rng = MersenneTwister(seed)
    df_shuffled = shuffle(rng, df)
    splits = []
    rest = copy(df_shuffled)
    for ratio in ratios
        pivot = Int(floor(ratio * nrow(df)))
        split = rest[begin:pivot, :]
        rest = rest[(pivot+1):end, :]
        splits = vcat(splits, split)
    end
    return splits
end


@doc """
    to_onehot(df, cols; remove_original = false)
    
translates the specified columns of a DataFrame to one-hot encoding  
specify remove_original = true to keep the original column
"""

to_onehot(df::DataFrame; remove_original=false) = to_onehot(df, names(df); remove_original = remove_original)
to_onehot(df::DataFrame, cols::Union{String, Symbol}; remove_original=false) = to_onehot(df, [cols]; remove_original = remove_original)
function to_onehot(df::DataFrame, cols; remove_original=false)
    isempty(cols) && throw(ErrorException("Array of cols is empty"))
    df_copy = copy(df)
    for col in cols
        df_copy = to_onehot!(df_copy, col; remove_original = remove_original)
    end
    return df_copy
end
function to_onehot!(data::DataFrame, col::Union{String, Symbol}; remove_original = false)
    classes = unique(data[!, col])
    # length(classes) == 2 && return data
    data_onehot = falses(nrow(data), length(unique(data[!, col])))
    for (i, class) in enumerate(classes)
        data_onehot[data[!, col] .== class, i] .= 1
        data[!, string(col, "_", class)] = Int.(data_onehot[:, i])
    end
    return remove_original ? data[!, Not(col)] : data
end


@doc """
    accuracy(y1, y2)
    
computes the accuracy of predictions
"""
accuracy(y1, y2) = length(y1) != length(y2) ? throw(ErrorException("Arrays have different lengths")) : sum(y1 .== y2) / length(y2)
# function accuracy_my(predictions, dataloader::DataLoader)

    
