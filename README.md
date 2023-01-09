# DRSOM: A Dimension-Reduced Second-Order Method for Convex and Nonconvex Optimization
<!-- | **Documentation** | | -->
[![docs-stable][docs-stable-img]][docs-stable-url] [![docs-dev][docs-dev-img]][docs-dev-url] 

[docs-stable-img]: https://img.shields.io/badge/docs-stable-blue.svg
[docs-dev-img]: https://img.shields.io/badge/docs-dev-purple.svg
<!-- [docs-stable-url]: https://JuliaSmoothOptimizers.github.io/LinearOperators.jl/stable -->
<!-- [docs-dev-url]: https://JuliaSmoothOptimizers.github.io/LinearOperators.jl/dev -->
[docs-stable-url]: https://copt-public.github.io/DRSOM.jl/stable
[docs-dev-url]: https://copt-public.github.io/DRSOM.jl/dev


DRSOM.jl is a Julia implementation of the Dimension-Reduced Second-Order Method for unconstrained smooth optimization. 

DRSOM.jl now includes a bunch of algorithms, including the variants of original DRSOM and the HSODM: *Homogeneous Second-order Descent Method*.

## Known issues
`DRSOM.jl` is still under active development. Please add issues on GitHub.

## License
`DRSOM.jl` is licensed under the MIT License. Check `LICENSE` for more details

## Acknowledgment

- Special thanks go to the COPT team and Tianyi Lin [(Darren)](https://tydlin.github.io/) for helpful suggestions.

## Developer

- Chuwen Zhang <chuwzhang@gmail.com>
- Yinyu Ye     <yyye@stanford.edu>

## Reference
You are welcome to cite our paper on DRSOM :)
```bibtex
@misc{zhang_drsom_2023,
	title = {{DRSOM}: {A} {Dimension} {Reduced} {Second}-{Order} {Method}},
	url = {http://arxiv.org/abs/2208.00208},
	doi = {10.48550/arXiv.2208.00208},
	author = {Zhang, Chuwen and Ge, Dongdong and He, Chang and Jiang, Bo and Jiang, Yuntian and Ye, Yinyu},
	month = jan,
	year = {2023},
	note = {arXiv:2208.00208 [cs, math]},
}
```
and HSODM,
```
@misc{zhang_homogenous_2022,
	title = {A {Homogenous} {Second}-{Order} {Descent} {Method} for {Nonconvex} {Optimization}},
	url = {http://arxiv.org/abs/2211.08212},
	publisher = {arXiv},
	author = {Zhang, Chuwen and Ge, Dongdong and He, Chang and Jiang, Bo and Jiang, Yuntian and Xue, Chenyu and Ye, Yinyu},
	month = nov,
	year = {2022},
	note = {arXiv:2211.08212 [math]},
	keywords = {Mathematics - Optimization and Control}
}
```