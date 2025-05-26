# DRSOM.jl

DRSOM.jl is a Julia implementation of the Dimension-Reduced Second-Order Method for unconstrained smooth optimization. 

!!! note
	DRSOM.jl is now a suite of second-order algorithms, including the variants of original DRSOM and the HSODM: a *Homogeneous Second-order Descent Method*.


The original 2-dimensional DRSOM works with the following iteration:

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
	




## Install DRSOM.jl

DRSOM is now available at JuliaRegistries, simply try
```
(v1.8) pkg> add DRSOM
```
try the `dev` branch (most up-to-date)
```
(v1.8) pkg> add DRSOM#dev
```

### Try your own ideas
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



## Algorithms

If you just want help on a specific algorithm, see the [Algorithm Reference](@ref alg_reference_list) page.

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
@article{heHomogeneousSecondorderDescent2025,
  title = {Homogeneous second-order descent framework: a fast alternative to {{Newton-type}} methods},
  shorttitle = {Homogeneous second-order descent framework},
  author = {He, Chang and Jiang, Yuntian and Zhang, Chuwen and Ge, Dongdong and Jiang, Bo and Ye, Yinyu},
  year = {2025},
  month = may,
  journal = {Mathematical Programming},
  issn = {1436-4646},
  doi = {10.1007/s10107-025-02230-3},
  urldate = {2025-05-15},
  langid = {english}
}
@misc{jiangNonconvexityUniversalTrustregion2023,
  title = {Beyond nonconvexity: a universal trust-region method with new analyses},
  shorttitle = {Beyond nonconvexity},
  author = {Jiang, Yuntian and He, Chang and Zhang, Chuwen and Ge, Dongdong and Jiang, Bo and Ye, Yinyu},
  year = {2023},
  number = {arXiv:2311.11489},
  eprint = {2311.11489},
  primaryclass = {math},
  publisher = {arXiv},
  doi = {10.48550/arXiv.2311.11489},
  urldate = {2025-02-02},
  archiveprefix = {arXiv}
}
@misc{zhangDRSOMDimensionReduced2022,
  title = {{{DRSOM}}: a dimension reduced second-order method},
  shorttitle = {{{DRSOM}}},
  author = {Zhang, Chuwen and Ge, Dongdong and He, Chang and Jiang, Bo and Jiang, Yuntian and Ye, Yinyu},
  year = {2022},
  month = may,
  number = {arXiv:2208.00208},
  eprint = {2208.00208},
  primaryclass = {cs, math},
  publisher = {arXiv},
  urldate = {2023-07-24},
  archiveprefix = {arXiv}
}
@article{zhangHomogeneousSecondorderDescent2025,
  title = {A homogeneous second-order descent method for nonconvex optimization},
  author = {Zhang, Chuwen and He, Chang and Jiang, Yuntian and Xue, Chenyu and Jiang, Bo and Ge, Dongdong and Ye, Yinyu},
  year = {2025},
  month = may,
  journal = {Mathematics of Operations Research},
  publisher = {INFORMS},
  issn = {0364-765X},
  doi = {10.1287/moor.2023.0132},
  urldate = {2025-05-19}
}
```