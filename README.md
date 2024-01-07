# DRSOM.jl: A Second-Order Optimization Package for Nonlinear Programming
<!-- | **Documentation** | | -->
[![docs-stable][docs-stable-img]][docs-stable-url] [![docs-dev][docs-dev-img]][docs-dev-url] 

[docs-stable-img]: https://img.shields.io/badge/docs-stable-blue.svg
[docs-dev-img]: https://img.shields.io/badge/docs-dev-purple.svg
<!-- [docs-stable-url]: https://JuliaSmoothOptimizers.github.io/LinearOperators.jl/stable -->
<!-- [docs-dev-url]: https://JuliaSmoothOptimizers.github.io/LinearOperators.jl/dev -->
[docs-stable-url]: https://copt-public.github.io/DRSOM.jl/stable
[docs-dev-url]: https://copt-public.github.io/DRSOM.jl/dev


DRSOM.jl is a Julia implementation of a few second-order optimization methods for nonlinear optimization. 

**DRSOM.jl now includes a bunch of other algorithms, beyond the original `DRSOM`**:
- Dimension-Reduced Second-Order Method (`DRSOM`)
- Homogeneous Second-order Descent Method (`HSODM`).
- Homotopy Path-Following HSODM (`PFH`)
- Universal Trust-Region Method (`UTR`)

## Reference
You are welcome to cite our papers : )
```
1.  Zhang, C., Ge, D., He, C., Jiang, B., Jiang, Y., Ye, Y.: DRSOM: A Dimension Reduced Second-Order Method, http://arxiv.org/abs/2208.00208, (2022)
2.  Zhang, C., Ge, D., He, C., Jiang, B., Jiang, Y., Xue, C., Ye, Y.: A Homogeneous Second-Order Descent Method for Nonconvex Optimization, http://arxiv.org/abs/2211.08212, (2022)
3.  He, C., Jiang, Y., Zhang, C., Ge, D., Jiang, B., Ye, Y.: Homogeneous Second-Order Descent Framework: A Fast Alternative to Newton-Type Methods, http://arxiv.org/abs/2306.17516, (2023)
4.  Jiang, Y., He, C., Zhang, C., Ge, D., Jiang, B., Ye, Y.: A Universal Trust-Region Method for Convex and Nonconvex Optimization, http://arxiv.org/abs/2311.11489, (2023)
```

## Developer

- Chuwen Zhang <chuwzhang@gmail.com>
- Yinyu Ye     <yyye@stanford.edu>



## Known issues
`DRSOM.jl` is still under active development. Please add issues on GitHub.

## License
`DRSOM.jl` is licensed under the MIT License. Check `LICENSE` for more details

## Acknowledgement

- Special thanks go to the COPT team and Tianyi Lin [(Darren)](https://tydlin.github.io/) for helpful suggestions.
