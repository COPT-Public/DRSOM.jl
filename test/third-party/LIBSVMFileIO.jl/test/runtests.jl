using Test
using LIBSVMFileIO
using SparseArrays
using Random
Random.seed!(0)


function readcorrectly(data, labels, d_read, l_read, precision=1e6)
	@test length(d_read) == length(data)
	@test length(l_read) == length(labels)
	data_correct = true
	labels_correct = true
	for i = 1:length(d_read)
		data_correct =
			all(isapprox.(d_read[i], round.(data[i].*precision)./precision)) &&
			data_correct
		labels_correct =
			all(isapprox.(l_read[i], round.(labels[i].*precision)./precision))&&
			labels_correct
	end
	@test data_correct
	@test labels_correct
end



@testset "Basics" begin
	println("Testing Basics")
	data = [20 .*randn(5) for _ = 1:10]
	labels = [rand(-10:10) for _ = 1:10]
	file = "tmp.libsvm"


	libsvmwrite(data, labels, file)
	for l in eachline(file)
		println(l)
	end

	nEx, nFeat = libsvmsize(file)
	@test nEx == 10
	@test nFeat == 5

	d_read, l_read = libsvmread(file)
	readcorrectly(data, labels, d_read, l_read)



	libsvmwrite(data, labels, file, precision=2)
	
	nEx, nFeat = libsvmsize(file)
	@test nEx == 10
	@test nFeat == 5
	
	d_read, l_read = libsvmread(file)
	readcorrectly(data, labels, d_read, l_read, 1e2)
end



@testset "Sparse Data" begin
	println("Testing Sparse Data")
	data = [20 .*sprandn(50, 0.4) for _ = 1:20]
	data[1][end] = 1.0
	labels = [rand(-10:10) for _ = 1:20]
	file = "tmp.libsvm"

	libsvmwrite(data, labels, file, precision=5)
	
	nEx, nFeat = libsvmsize(file)
	@test nEx == 20
	@test nFeat == 50
	
	d_read, l_read = libsvmread(file)
	readcorrectly(data, labels, d_read, l_read, 1e5)


	# Zeros out last data points
	nTrailing = 3
	for d in data
		d[(end-nTrailing):end] .= 0.0
	end
	
	libsvmwrite(data, labels, file)
	
	nEx, nFeat = libsvmsize(file)
	@test nEx == 20
	@test nFeat == 50-nTrailing-1

	data_ctrl = [d[1:(end-nTrailing-1)] for d in data]
	d_read, l_read = libsvmread(file)
	readcorrectly(data_ctrl, labels, d_read, l_read)
end



@testset "Parsing Types" begin
	println("Testing Parsing Types")
	data1 = [20 .*randn(5) for _ = 1:10]
	data2 = [rand(-10:10,5) for _ = 1:10]
	data = [data1;data2]
	labels = [rand(-10:10) for _ = 1:20]
	labels_f = randn(20)
	file = "tmp.libsvm"

	

	libsvmwrite(data, labels, file)
	d_read, l_read = libsvmread(file, valuetype=Float32)
	@test eltype(eltype(d_read)) == Float32
	@test d_read[1] isa SparseVector
	@test eltype(l_read) == Int
	readcorrectly(data, labels, d_read, l_read)
	
	d_read, l_read = libsvmread(file, labeltype=Float64)
	@test eltype(eltype(d_read)) == Float64
	@test d_read[1] isa SparseVector
	@test eltype(l_read) == Float64
	readcorrectly(data, labels, d_read, l_read)

	

	libsvmwrite(data, labels_f, file)
	for l in eachline(file)
		println(l)
	end
	d_read, l_read = libsvmread(file, labeltype=Float32, valuetype=Float32)
	@test eltype(eltype(d_read)) == Float32
	@test d_read[1] isa SparseVector
	@test eltype(l_read) == Float32
	readcorrectly(data, labels_f, d_read, l_read)
	
	d_read, l_read = libsvmread(file, labeltype=Float32, dense=true)
	@test eltype(eltype(d_read)) == Float64
	@test d_read[1] isa Vector
	@test eltype(l_read) == Float32
	readcorrectly(data, labels_f, d_read, l_read)
end



@testset "Selection" begin
	println("Testing Selection")
	data = [20 .*randn(10) for i = 1:30]
	labels = rand(-10:10, 30)
	file = "tmp.libsvm"

	libsvmwrite(data, labels, file)

	nEx, nFeat = libsvmsize(file)
	@test nEx == 30
	@test nFeat == 10
	
	selection = 1:nEx
	d_read, l_read = libsvmread(file, selection=selection, size=(nEx,nFeat))
	readcorrectly(data, labels, d_read, l_read)

	selection = 7:3:16
	d_read, l_read = libsvmread(file, selection=selection)
	readcorrectly(data[selection], labels[selection], d_read, l_read)

	selection = 13:13
	d_read, l_read = libsvmread(file, selection=selection, size=(nEx,nFeat))
	readcorrectly(data[selection], labels[selection], d_read, l_read)
	
	selection = 13:12
	d_read, l_read = libsvmread(file, selection=selection, size=(nEx,nFeat))
	readcorrectly(data[selection], labels[selection], d_read, l_read)
	

	selection = 2:2:21
	d_read, l_read = libsvmread(file, selection=selection, valuetype=Float32,
								labeltype=Float64, size=(nEx,nFeat))
	readcorrectly(data[selection], labels[selection], d_read, l_read)
	
end



@testset "Multi-label" begin
	println("Testing Multi-label")
	data = [20 .*randn(5) for i = 1:15]
	labels = [Tuple(randn(Float64,rand(1:6))) for _ = 1:15]
	file = "tmp.libsvm"

	libsvmwrite(data, labels, file, precision=2)

	for l in eachline(file)
		println(l)
	end

	nEx, nFeat = libsvmsize(file)
	@test nEx == 15
	@test nFeat == 5

	d_read, l_read = libsvmread(file, multilabel=true, labeltype=Float64)
	readcorrectly(data, labels, d_read, l_read, 1e2)
end
