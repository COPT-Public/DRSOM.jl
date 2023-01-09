"""
Contains functions for reading and writing data from files in LIBSVM-format.
"""
module LIBSVMFileIO

using SparseArrays
using Formatting

export libsvmsize, libsvmread, libsvmwrite


"""
    libsvmsize(file)

Read the number of examples/data points and the biggest non-zero feature index
from <file>.

# Example
```julia-repl
julia> nEx, nFeat = libsvmsize("path/to/libsvm/file/a1a")
(1605, 119)
```

**Note:** The number of features is not necessarily the same the number of
features given in the data set documentation. Trailing features that are zero
for all data points are not counted (and can not be counted). For instance, the
'a1a' data set have 123 features but libsvmsize return 119.

See also: [`libsvmread`](@ref), [`libsvmwrite`](@ref), [`libsvmsize`](@ref)
"""
function libsvmsize(file)
	nEx = 0
	nFeat = 0
	for ex in eachline(file)
		nEx += 1

		offset = 1
		feat_idx_rx = r"[0-9]+ *:"

		while true
			m_idx = match(feat_idx_rx, ex, offset)
			m_idx == nothing && break

			offset = m_idx.offset + length(m_idx.match)
			idx = parse(Int, m_idx.match[1:(end-1)])
			nFeat = max(nFeat, idx)
		end
	end
	return nEx, nFeat
end


"""
    libsvmread(file; <keyword arguments>)
Read the data and labels from <file>. Return data as a `Vector` of
`SparseVector` and labels a `Vector`.

# Example
```julia-repl
julia> data, labels = libsvmread("path/to/libsvm/file/a1a");

julia> typeof(data)
Array{SparseArrays.SparseVector{Float64,Int64},1}

julia> typeof(labels)
Array{Int64,1}

julia> length(data)
1605

julia> data, labels = libsvmread("path/to/libsvm/file/a1a", selection=11:20);

julia length(data)
10
```

# Keyword Arguments
- `labeltype::Number`: Number type to parse labels as (default: `Int`).
- `valuetype::Real`: Number type to parse data values as (default: `Float64`).
- `size::Tuple{Integer,Integer}`: Size of data set given as `(<nbr of data
  points>, <nbr of examples>)`. If given, no parsing of the data set size is
  made. Useful for speeding up repeated partial reads.
- `selection`: Collection of indices. If given, only load the data points with
  indices in `selection`.
- `multilabel::Bool`: The data points of the data set have multiple labels
  (default: false). For multi-label data the labels are returned as a `Vector`
  of `NTuples` of <labeltype>.
- `dense::Bool`: Return data as ordinary dense vectors instead of
  `SparseVector` (default: false)

See also: [`libsvmread`](@ref), [`libsvmwrite`](@ref), [`libsvmsize`](@ref)
"""
function libsvmread(
		file;
		labeltype=Int, valuetype=Float64,
		size=nothing, selection=nothing, multilabel=false, dense=false)

	size == nothing ? ( (nEx,nFeat) = libsvmsize(file)) : ( (nEx,nFeat) = size)
	selection == nothing && (selection = 1:nEx)

	function parseexample(ex)
		entries = split(ex, ' ')
		
		# Labels
		if multilabel
			m = match(r" *[-+,\.0-9]+ *", ex)
			l = Tuple(parse.(labeltype, split(m.match, ',')))
		else
			m = match(r" *[-+\.0-9]+ *", ex)
			l = parse(labeltype, m.match)
		end

		# Features
		d = (dense ? zeros(valuetype,nFeat) : spzeros(valuetype,nFeat))

		offset = 1
		idx_rx = r"[0-9]+ *:"
		val_rx = r": *[+-\.0-9]+"

		while true
			m_idx = match(idx_rx, ex, offset)
			m_val = match(val_rx, ex, offset)
			(m_idx == nothing || m_val == nothing) && break
			offset = m_val.offset + length(m_val.match)

			idx = parse(Int, m_idx.match[1:(end-1)])
			val = parse(valuetype, m_val.match[2:end])
			d[idx] = val
		end
		return l, d
	end


	data_container = (dense ? Vector{valuetype} : SparseVector{valuetype,Int})
	label_container = (multilabel ? (NTuple{N,labeltype} where N) : labeltype)
	
	data = Vector{data_container}(undef,length(selection))
	labels = Vector{label_container}(undef, length(selection))

	open(file, "r") do f
		i_store = 1
		for i_ex = 1:nEx
			ex = readline(f)
			if i_ex in selection 
				labels[i_store], data[i_store] = parseexample(ex)
				i_store += 1
			end
		end
	end

	return data, labels
end

"""
    libsvmwrite(data, labels, file; precision=6)

Write <data> and <labels> to <file> in LIBSVM-format.

# Example
```julia-repl
julia> data = [randn(5) for _ = 1:10];

julia> labels = [(rand(1:2), randn()) for _ = 1:10];

julia> libsvmwrite(data, labels, "example")

julia> for l in eachline("example") println(l) end
1,1.374311 1:0.447671 2:0.151464 3:-0.251898
1,-0.455168 1:-0.782467 2:0.160167 3:-0.850168
1,1.618833 1:-0.946307 2:-1.073485 3:0.504319
2,-1.141834 1:0.545098 2:-0.15826 3:-1.271802
1,0.70919 1:0.414993 2:0.232563 3:1.5618
```

# Arguments
- `data` and `labels` are iterable collections containing iterable collections of
  numbers.
- `precision` is the maximal number of decimal places to write to `file`.
  Trailing zeros are never written. 

See also: [`libsvmread`](@ref), [`libsvmwrite`](@ref), [`libsvmsize`](@ref)
"""
function libsvmwrite(data, labels, file; precision=6)
	length(data) != length(labels) &&
		error("Number of examples and labels not the same.")

	formatnumber(x::Integer) = string(x)
	function formatnumber(x)
		fmt_spec = FormatSpec(".$(precision)f")
		str = fmt(fmt_spec, x)
		# Prune trailing zeros and decimal periods
		str = str[1:findlast(c->c!='0',str)]
		str[end] == '.' && (str = str[1:(end-1)])
		return str
	end

	function writelabels(labels, f)
		s = ""
		for l in labels
			s *= formatnumber(l)*","
		end
		write(f, s[1:(end-1)])
	end

	function writefeatures(data, f)
		s = ""
		for (i,d) in enumerate(data)
			val = formatnumber(d)
			parse(Float64, val) == 0.0 && continue # Ignore zeros
			s *= formatnumber(i)*":"*val*" "
		end
		write(f, s[1:(end-1)])
	end


	open(file, "w") do f
		for i = 1:length(data)
			writelabels(labels[i], f)
			write(f, " ")
			writefeatures(data[i], f)
			write(f, "\n")
		end
	end
end

end # Module
