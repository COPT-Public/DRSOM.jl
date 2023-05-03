# DRSOM.jl

DRSOM.jl is a Julia implementation of the Dimension-Reduced Second-Order Method for unconstrained smooth optimization. The DRSOM works with the following iteration:

```math
        x_{k+1}     = x_k- \alpha_k^1 g_k +\alpha_k^2 d_k, \\
```
where  $m_k^\alpha(\alpha)$ is a 2-dimensional quadratic approximation to $f(x)$ using gradient $g_k$ and Hessian information $H_k$, namely,
```math
  \begin{aligned}
    \min_{\alpha \in\mathbb{R}^2} &~m_k(\alpha) := f(x_k) + (c_k)^{T} \alpha+\frac{1}{2} \alpha^{T} Q_k \alpha\\
    ~\mathrm{s.t.}~& \|\alpha\|_{G_k}:= \sqrt{\alpha^T G_k \alpha} \le \Delta, \quad \textrm{with}\quad G_k=\left[\begin{array}{cc}
\left(g_k\right)^T g_k & -\left(g_k\right)^T d_k \\
-\left(g_k\right)^T d_k & \left(d_k\right)^T d_k
\end{array}\right],\;   
  \end{aligned} 
```
and
```math

Q_k =\begin{bmatrix}
	( g_k)^{T} H_k g_k  & -( d_k)^{T} H_k g_k \\
	-( d_k)^{T} H_k g_k & ( d_k)^{T} H_k d_k
	\end{bmatrix} \in \mathcal S^{2},\; 
c_k =\begin{bmatrix}
	-\left\| g_k\right\|^{2} \\
	( g_k)^{T} d_k
	\end{bmatrix} \in \mathbb{R}^{2}.
```

- The differentiation is done by `ForwardDiff` and `ReverseDiff` 
- The subproblem is very easy to solve.

!!! note
	Notably, **DRSOM does not have to compute** $n$-by-$n$ Hessian $H_k$ directly (of course, it is perfect if you can provide!).
	Instead, it requires Hessian-vector products (HVPs) or `interpolation` to contruct the quadratic model. The latter approach is now preferred.
	

DRSOM.jl is now a suite of algorithms, including the variants of original DRSOM and the HSODM: a *Homogeneous Second-order Descent Method*.



## Install DRSOM.jl
!!! tip
    To try your own ideas with `DRSOM.jl`, 
	use local path mode:
	```
	(v1.8) pkg> add path-to-DRSOM.jl
	``` 
	or the `dev` command:
	```
	(v1.8) pkg> dev DRSOM
	``` 



## API Reference

If you just want help on a specific function, see the [API Reference](@ref api_reference_list) page.

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
@misc{zhang_drsom_2022,
	title = {{DRSOM}: {A} {Dimension} {Reduced} {Second}-{Order} {Method} and {Preliminary} {Analyses}},
	url = {http://arxiv.org/abs/2208.00208},
	publisher = {arXiv},
	author = {Zhang, Chuwen and Ge, Dongdong and Jiang, Bo and Ye, Yinyu},
	month = jul,
	year = {2022},
	note = {arXiv:2208.00208 [cs, math]},
	keywords = {Computer Science - Machine Learning, Mathematics - Optimization and Control},
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