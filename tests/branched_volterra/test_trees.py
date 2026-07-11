from tensordev.branched_volterra import RootedTree, decorated_trees, symmetry_factor


def test_one_decoration_has_standard_rooted_tree_counts():
    trees = decorated_trees(1, 4)
    assert [sum(tree.degree == n for tree in trees) for n in range(1, 5)] == [1, 1, 2, 4]


def test_children_are_canonical_and_symmetry_is_butcher_factor():
    leaf0, leaf1 = RootedTree(0), RootedTree(1)
    tree = RootedTree(0, (leaf1, leaf0, leaf0))
    assert tree.children == (leaf0, leaf0, leaf1)
    assert symmetry_factor(tree) == 2
