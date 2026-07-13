# Multi-time Volterra rough path

The implementation is a finite-grid realisation of the word-indexed Volterra
rough path of Harang--Tindel.  It stores

`Z[n][batch, tau, s, t, word]`

for every grid triple `s <= t <= tau`.  The scalar kernel is cell-averaged and
the matrix `A` acts on every path increment before iteration.

For a word obtained by appending coordinate `a`, the defining recursion is

`Z^(w,a),tau_ts = sum_{j=s}^{t-1} W[tau,j] Z^w,j_sj tensor dY_j^a`.

## Convolution product and Chen identity

The seeded future recursion implements the discrete convolution product
`star`.  Splitting at `u`, every word of degree `n` satisfies

`Z^n_ts = Z^n_su + sum_(m=0)^(n-1) Star_(n-m)(Z^m_su; [u,t])`.

`chen_report(n)` checks this identity exhaustively on the current grid.

## Analytic diagnostics

`discrete_volterra_holder_report` evaluates the grid analogue of the main
`V^(alpha,gamma)` size estimate.  `compare_terminal_nested_grids` compares
terminal coefficients under mesh refinement.

These checks do not replace a continuous proof.  A genuine rough path above a
non-smooth limiting signal is obtained only when a sequence of smooth/discrete
lifts converges and the Chen and Holder bounds remain uniform.  Renormalised,
Ito and branched lifts require additional data and are outside this word-indexed
geometric implementation.
