export read_csv_data,
       replace_in_cols!,
       replace_in_cols,
       apply_to_cols,
       categorize,
       standartize,
       most_common,
       strip_cabin_numbers,
       get_title_groups,
       replace_names_with_title_categories,
       random_split,
       accuracy,
       to_onehot

"""
    read_csv_data(name)

Reads a csv file and returns a DataFrame
"""
function read_csv_data(name::String)
    ispath(name) || throw(ErrorException("Invalid data path"))
    CSV.read(name, DataFrame; header = true)
end

replace_eq(value::Any, src::Missing, dst::Any) = ismissing(value) ? dst : value # coalesce
replace_eq(value::Any, src::Any, dst::Any) = value == src ? dst : value
function replace_eq(value::Any, src::AbstractArray, dst::Any)
    !isnothing(findfirst(entry -> entry == value, src)) ? dst : value
end # many-to-one 
function replace_eq(value::Any, src::AbstractArray, dst::AbstractArray)
    !isnothing(findfirst(entry -> entry == value, src)) ?
    dst[findfirst(entry -> entry == value, src)] : value # many-to-many
end # many-to-many

"""
    replace_in_cols(df, cols, src, dst)

Replaces entries in the DataFrame columns, src values are replaced with dst values
"""
function replace_in_cols(df::DataFrame,
                         cols::Union{String, Symbol, Vector{<:Union{String, Symbol}}},
                         src::Any,
                         dst::Any)
    transform(df, cols .=> ByRow(entry -> replace_eq(entry, src, dst)) .=> cols)
end

function replace_in_cols!(df::DataFrame,
                          cols::Union{String, Symbol, Vector{<:Union{String, Symbol}}},
                          src::Any,
                          dst::Any)
    transform!(df, cols .=> ByRow(entry -> replace_eq(entry, src, dst)) .=> cols)
end

function replace_in_cols(df::DataFrame, src::Any, dst::Any)
    replace_in_cols(df, names(df), src, dst)
end
function replace_in_cols!(df::DataFrame, src::Any, dst::Any)
    replace_in_cols!(df, names(df), src, dst)
end

"""
    categorize(df, cols)

Categorizes the chosen columns in a DataFrame
"""
function categorize(df::DataFrame, cols::Vector{<:Union{String, Symbol}})
    transform(df,
              cols .=>
                  (col -> map(entry -> col_to_categorical(col, entry), col)) .=>
                      cols)
end

"""
    categorize(df)

Categorizes all columns in a DataFrame
"""
categorize(df::DataFrame) = categorize(df, names(df))
categorize(df::DataFrame, cols::Union{Symbol, String}) = categorize(df, [cols])
function col_to_categorical(col::AbstractArray, value::Any)
    findfirst(entry -> entry == value, unique(col)) - 1
end
col_to_categorical(col::AbstractArray, value::Number) = value

"""
    standartize(df, cols)

Normalizes the chosen columns in a DataFrame
"""
function standartize(df::DataFrame, cols::Vector{<:Union{String, Symbol}})
    transform(df,
              cols .=> (col -> map(entry -> col_to_norm(col, entry), col)) .=> cols)
end

col_to_norm(col::AbstractArray{<:Number}, value::Number) = (value - mean(col)) / std(col)

"""
    standartize(df)

Standartizes all columns in a DataFrame
"""
standartize(df::DataFrame) = standartize(df, names(df))
standartize(df::DataFrame, cols::Union{Symbol, String}) = standartize(df, [cols])

"""
    apply_to_cols(df, cols, func)

Applies a function to the chosen columns in a DataFrame
"""
function apply_to_cols(df::DataFrame, cols::Union{Symbol, String}, func::Function)
    func(collect(skipmissing(df[!, cols])))
end
function apply_to_cols(df::DataFrame, cols::Vector{<:Union{String, Symbol}}, func::Function)
    map(col -> apply_to_cols(df, col, func), cols)
end # we DO NOT have Point{Float64} <: Point{Real}
apply_to_cols(df::DataFrame, func::Function) = apply_to_cols(df, names(df), func)

most_common(arr::AbstractArray) = argmax(countmap(arr))

"""
    strip_cabin_numbers(arr)

Given an array of strings, where a string represents a cabin number, removes the string's content past the first character
"""
strip_cabin_numbers(arr::AbstractArray{<:AbstractString}) = map(entry -> entry[begin], arr)

"""
    get_title_groups()

Groups names based on title
"""
function get_title_groups()
    default_titles = ["Mr.", "Mrs.", "Mlle.", "Miss.", "Mme.", "Ms."]
    royal_titles = ["Master.", "Don.", "Lady.", "Sir.", "Countess.", "Jonkheer."]
    other_titles = ["Major.", "Col.", "Capt.", "Dr.", "Rev."]

    [(default_titles, "D"), (royal_titles, "R"), (other_titles, "O")]
end

function replace_names_with_title_categories(df::DataFrame,
                                             groups::AbstractArray{
                                                                   Tuple{Vector{String},
                                                                         String}})
    for group in groups
        transform!(df,
                   :Name => ByRow(entry -> any(occursin.(group[begin], entry)) ? group[end] : entry) => :Name)
    end
    df
end

"""
    random_split(df, ratios; seed = 42)
    
Creates an array with data splits, where sizes of the splits are determined by their ratios w.r.t the original data size
"""
function random_split(df::DataFrame, ratios::AbstractArray{<:Number}; seed = 42)
    isempty(ratios) && throw(ErrorException("Array of ratios is empty"))
    (all(0.0 .<= ratios .<= 1.0) && sum(ratios) == 1.0) ||
        throw(ErrorException("Invalid size ratios"))
    rng = MersenneTwister(seed)
    df_shuffled = shuffle(rng, df)
    splits = []
    rest = copy(df_shuffled)
    for ratio in ratios
        pivot = Int(floor(ratio * nrow(df)))
        split = rest[begin:pivot, :]
        rest = rest[(pivot + 1):end, :]
        splits = vcat(splits, split)
    end
    splits
end

"""
    to_onehot(df, cols; remove_original = false)
    
Translates the specified columns of a DataFrame to one-hot encoding  
specify remove_original = true to keep the original column
"""

function to_onehot(df::DataFrame; remove_original = false)
    to_onehot(df, names(df); remove_original = remove_original)
end

function to_onehot(df::DataFrame, cols::Vector{<:Union{String, Symbol}};
                   remove_original = false)
    isempty(cols) && throw(ErrorException("Array of cols is empty"))
    df_local = df
    for col in cols
        df_local = to_onehot(df_local, col; remove_original = remove_original)
    end
    df_local
end

function to_onehot(df::DataFrame, col::Union{String, Symbol}; remove_original = false)
    classes = unique(df[!, col])
    data_onehot = falses(nrow(df), length(unique(df[!, col])))
    df_local = DataFrame()
    for (i, class) in enumerate(classes)
        data_onehot[df[!, col] .== class, i] .= 1
        df_local[!, string(col, "_", class)] = Int.(data_onehot[:, i])
    end
    remove_original ? [df[!, Not(col)] df_local] : [df df_local]
end

"""
    accuracy(y1, y2)
    
Computes the accuracy of predictions
"""
function accuracy(y1::AbstractArray, y2::AbstractArray)
    length(y1) != length(y2) ? throw(ErrorException("Arrays have different lengths")) :
    sum(y1 .== y2) / length(y2)
end
# function accuracy_my(predictions, dataloader::DataLoader)
