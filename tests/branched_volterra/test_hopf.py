from tensordev.branched_volterra import RootedTree
from tensordev.branched_volterra.hopf import UNIT, antipode, coproduct, forest, tree_factorial


def test_coproduct_of_leaf_has_unit_terms_only():
    leaf = RootedTree(0)
    assert set(coproduct(leaf)) == {(forest(leaf), UNIT), (UNIT, forest(leaf))}


def test_coproduct_contains_cut_term_for_one_edge_tree():
    leaf = RootedTree(0)
    tree = RootedTree(0, (leaf,))
    terms = set(coproduct(tree))
    assert (forest(tree), UNIT) in terms
    assert (UNIT, forest(tree)) in terms
    assert (forest(leaf), forest(RootedTree(0))) in terms


def test_tree_factorial_and_antipode_leaf():
    leaf = RootedTree(0)
    assert tree_factorial(leaf) == 1
    assert antipode(leaf) == {forest(leaf): -1}
