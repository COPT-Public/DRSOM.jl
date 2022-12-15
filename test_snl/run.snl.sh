for i in 500,50 \
	1000,80 \
	2000,120 \
	3000,150 \
	4000,400 \
	5000,500 \
	6000,600 ; do 
IFS=',' read n m <<< "${i}";
cmd="julia -t 1 --project=./test_snl test_snl/app.jl --n $n --m $m --nf 0 --degree 50 --timelimit 3000 --option_plot_js 1 --option_set_comparison gd cg &> $n.$m-snl.log &";
echo $cmd
done
