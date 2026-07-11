# Branched Volterra prototype

This experimental module implements the finite-grid feature recursion

\[
 Z^{B_a^+(\tau_1\cdots\tau_r)}_{t_i}
 = \sum_{j<i} \bar K_{ij}
   \prod_{q=1}^r Z^{\tau_q}_{t_j}\,\Delta X_j^a,
\]

where trees are non-planar decorated rooted trees and \(\bar K_{ij}\) is a
cell-averaged causal kernel.  The tree basis is the Butcher/Connes--Kreimer
basis and is closed under subtrees.

The public module is `tensordev.branched_volterra`:

```python
import jax.numpy as jnp
from tensordev.branched_volterra import (
    BranchedVolterraSignature,
    fractional_causal_weights,
)

times = jnp.linspace(0.0, 1.0, 51)
X = jnp.stack([jnp.sin(times), times], axis=-1)[None]
weights = fractional_causal_weights(times, beta=0.7)

transform = BranchedVolterraSignature(dimension=2, trunc=3)
result = transform(X, weights)
print(result.values.shape)
```

`fractional_causal_weights` integrates the diagonal fractional singularity
analytically.  This module is a deterministic, left-point product-integration
prototype for smooth or discretely observed paths.  It does not yet implement
the renormalised stochastic Volterra lift needed for arbitrary singular random
drivers.  That extension requires additional Itô/Stratonovich and
renormalisation data.
