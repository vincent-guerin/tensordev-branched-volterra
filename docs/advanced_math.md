# Mathematical status of the branched Volterra layer

The implementation now separates three levels of structure:

1. `RootedTree` and commutative `Forest` objects, with canonical child order.
2. Formal Connes--Kreimer operations: admissible-cut coproduct, tree factorial
   and the connected Hopf-algebra antipode.
3. A numerical Volterra character evaluated on a causal grid.

For a tree `B_a^+(tau_1 ... tau_m)` and an external readout time `t_i`, the
left product-integration approximation is

\[
 Z_{t_i}^{B_a^+(\tau_1\cdots\tau_m)}
 = \sum_{j<i}\bar K(t_i,[t_j,t_{j+1}])
   \prod_{q=1}^m Z_{t_j}^{\tau_q}\,\Delta X_j^a.
\]

`multitime_branched_vsig` returns the full `(batch, readout_time, tree)` array;
the terminal vector is only a convenience view. This is the first explicit
multitime layer and avoids conflating Volterra readout time with integration
time.

The Hopf operations are formal and exact on finite trees. They do not by
themselves construct a stochastic rough-path lift. For Brownian drivers or
strongly singular kernels, a future layer must specify Ito versus Stratonovich
and a renormalisation model, then establish a Wong--Zakai/convergence theorem.
The current numerical character is therefore intended for smooth or discretely
observed paths and for controlled empirical comparisons.
