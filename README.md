# DRSOM: A Dimension-Reduced Second-Order Method for Convex and Nonconvex Optimization

[docs-dev-img]: https://img.shields.io/badge/docs-dev-purple.svg
<!-- [docs-dev-url]: https://JuliaSmoothOptimizers.github.io/AdaptiveRegularization.jl/dev -->


DRSOM.jl is a Julia implementation of the Dimension-Reduced Second-Order Method for unconstrained smooth optimization. The DRSOM works with the following iteration:

$$
        x_{k+1}     = x_k- \alpha_k^1 g_k +\alpha_k^2 d_k, \\
        \alpha_k  = \arg \min m_k^\alpha(\alpha), 
$$

where  $m_k^\alpha(\alpha)$ is a 2-dimensional quadratic approximation to $f(x)$ using gradient $g_k$ and Hessian information $H_k$.

- The differentiation is done by `ForwardDiff` and `ReverseDiff` using finite-difference.
- Notably, **DRSOM does not have to compute** Hessian $H_k$; instead, it only requires Hessian-vector products or uses interpolation to contruct the quadratic model.
- Of course, you may provide $g_k, H_k$ directly. 

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
You are welcome to cite our paper :), [see](https://arxiv.org/abs/2208.00208)
```bibtex

@misc{zhang_drsom_2022,
	title = {{DRSOM}: {A} {Dimension} {Reduced} {Second}-{Order} {Method} and {Preliminary} {Analyses}},
	copyright = {All rights reserved},
	shorttitle = {{DRSOM}},
	url = {http://arxiv.org/abs/2208.00208},
	language = {en},
	urldate = {2022-08-12},
	publisher = {arXiv},
	author = {Zhang, Chuwen and Ge, Dongdong and Jiang, Bo and Ye, Yinyu},
	month = jul,
	year = {2022},
	note = {arXiv:2208.00208 [cs, math]},
	keywords = {Computer Science - Machine Learning, Mathematics - Optimization and Control},
}


@misc{zhang_homogenous_2022,
	title = {A {Homogenous} {Second}-{Order} {Descent} {Method} for {Nonconvex} {Optimization}},
	url = {http://arxiv.org/abs/2211.08212},
	urldate = {2022-11-16},
	publisher = {arXiv},
	author = {Zhang, Chuwen and Ge, Dongdong and He, Chang and Jiang, Bo and Jiang, Yuntian and Xue, Chenyu and Ye, Yinyu},
	month = nov,
	year = {2022},
	note = {arXiv:2211.08212 [math]},
	keywords = {Mathematics - Optimization and Control}
}
```