julia -t 1 --project=./test_snl test_snl/app.jl --n 2000 --nf 0 --m 150 --degree 50 --timelimit 1000 --option_plot_js 1 --option_set_comparison gd
julia -t 1 --project=./test_snl test_snl/app.jl --n 3000 --nf 0 --m 200 --degree 50 --timelimit 3000 --option_plot_js 1 --option_set_comparison gd
julia -t 1 --project=./test_snl test_snl/app.jl --n 4000 --nf 0 --m 300 --degree 50 --timelimit 3000 --option_plot_js 1 --option_set_comparison gd
julia -t 1 --project=./test_snl test_snl/app.jl --n 5000 --nf 0 --m 500 --degree 50 --timelimit 5000 --option_plot_js 1 --option_set_comparison gd
julia -t 1 --project=./test_snl test_snl/app.jl --n 6000 --nf 0 --m 600 --degree 50 --timelimit 5000 --option_plot_js 1 --option_set_comparison gd
julia -t 1 --project=./test_snl test_snl/app.jl --n 7000 --nf 0 --m 700 --degree 50 --timelimit 5000 --option_plot_js 1 --option_set_comparison gd
