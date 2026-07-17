# Fractional hypergeometric operators

`tensordev.fractional_hypergeometric` implements the left-sided
Marichev–Saigo–Maeda (MSM) integral and its Saigo, Riemann–Liouville and
Erdelyi–Kober reductions on finite power expansions.

```python
from tensordev.fractional_hypergeometric import (
    MarichevSaigoMaedaLeftIntegral,
    SaigoLeftIntegral,
    RiemannLiouvilleLeftIntegral,
    ErdelyiKoberLeftIntegral,
)

msm = MarichevSaigoMaedaLeftIntegral(
    a=0.1, a_prime=0.1, b=0.05, b_prime=0.2, c=0.8,
)
value = msm.power(d=1.5, x=1.0)  # MSM[t**0.5](1)
```

The MSM kernel involves Appell's `F3` and depends on `x` and `t` separately;
it is not a stationary convolution kernel `k(x-t)`. It therefore intentionally
does not inherit from `ConvolutionKernel` and cannot yet be passed to
`VolterraSignature`.

The power-action formulas are analytic gamma-ratio identities from Jain,
Cattani and Agarwal (2022), Eq. (29), rather than a truncated `F3` series.
This avoids incorrectly using the defining Appell series outside its convergence
domain. The general numerical MSM integral for arbitrary sampled functions,
which requires stable analytic continuation of `F3`, is intentionally outside
this first implementation.
