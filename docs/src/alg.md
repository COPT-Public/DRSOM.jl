# [Algorithm Reference](@id alg_reference_list)



## Original DRSOM with 2 directions $g_k$ and $d_k$


!!! note
    We do not currently support nonsmooth/composite optimization, this is a future plan.

```@docs
DRSOM.DRSOMState
```

```@docs
DRSOM.DRSOMIteration
```

!!! note 
    This TRS solver only works for low-dimensional subproblems

```@docs
DRSOM.SimpleTrustRegionSubproblem
```

```@docs
DRSOM.TrustRegionSubproblem
```