from q_tomo.counterfactual import _balanced_design


def test_balanced_design_is_complementary_and_balanced():
    targets = list(range(17))
    design = _balanced_design(targets, n_models=12, seed=3)
    for target in targets:
        assert sum(target in design[model] for model in range(12)) == 6
    for pair in range(6):
        left, right = design[2 * pair], design[2 * pair + 1]
        assert left.isdisjoint(right)
        assert left | right == set(targets)
