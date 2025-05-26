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
1.  Jiang, Y., He, C., Zhang, C., Ge, D., Jiang, B., Ye, Y.: Beyond nonconvexity: a universal trust-region method with new analyses, http://arxiv.org/abs/2311.11489, (2023)
2.  He, C., Jiang, Y., Zhang, C., Ge, D., Jiang, B., Ye, Y.: Homogeneous second-order descent framework: a fast alternative to Newton-type methods. Math. Program. (2025). https://doi.org/10.1007/s10107-025-02230-3
3.  Zhang, C., He, C., Jiang, Y., Xue, C., Jiang, B., Ge, D., Ye, Y.: A homogeneous second-order descent method for nonconvex optimization. Mathematics of OR. (2025). https://doi.org/10.1287/moor.2023.0132
4.  Zhang, C., Ge, D., He, C., Jiang, B., Jiang, Y., Ye, Y.: DRSOM: a dimension reduced second-order method, http://arxiv.org/abs/2208.00208, (2022)
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
