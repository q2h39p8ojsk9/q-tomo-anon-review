from q_tomo.config import DataConfig
from q_tomo.data import NaturalMemoryCorpus, RuleMemoryCorpus


def test_corpus_is_deterministic_and_balanced():
    config = DataConfig(modulus=31, n_rule_relations=3, n_memory_relations=3, exposures_per_pair=2)
    first = RuleMemoryCorpus(config)
    second = RuleMemoryCorpus(config)
    assert first.metadata() == second.metadata()
    assert len(first.examples_for_group("seen_rule")) == len(first.examples_for_group("memorized"))
    assert len(first.examples_for_group("generalized")) == len(first.examples_for_group("nonmember"))
    assert set(first.train_keys).isdisjoint(first.holdout_keys)


def test_causal_labels_predict_digits_without_self_leakage():
    corpus = RuleMemoryCorpus(DataConfig(modulus=31, n_rule_relations=1, n_memory_relations=1))
    example = corpus.eval_examples[0]
    target_positions = [index for index, label in enumerate(example.labels) if label != -100]
    assert tuple(target_positions) == corpus.prediction_positions
    for position in target_positions:
        assert example.labels[position] == example.input_ids[position + 1]


def test_copy_and_memory_have_exact_matched_output_marginals():
    corpus = RuleMemoryCorpus(
        DataConfig(rule_family="copy", modulus=31, n_rule_relations=2, n_memory_relations=2)
    )
    seen_values = sorted(example.value for example in corpus.examples_for_group("seen_rule"))
    memory_values = sorted(example.value for example in corpus.examples_for_group("memorized"))
    generalized_values = sorted(example.value for example in corpus.examples_for_group("generalized"))
    nonmember_values = sorted(example.value for example in corpus.examples_for_group("nonmember"))
    assert seen_values == memory_values
    assert generalized_values == nonmember_values


def test_digitwise_rules_are_bijective_and_exactly_matched():
    corpus = RuleMemoryCorpus(
        DataConfig(
            rule_family="digitwise",
            modulus=100,
            number_base=10,
            n_rule_relations=16,
            n_memory_relations=16,
        )
    )
    for relation in range(16):
        assert len({corpus.value_for(relation, key) for key in range(100)}) == 100
        memory_relation = relation + 16
        assert sorted(corpus.value_for(relation, key) for key in corpus.train_keys) == sorted(
            corpus.value_for(memory_relation, key) for key in corpus.train_keys
        )
        assert sorted(corpus.value_for(relation, key) for key in corpus.holdout_keys) == sorted(
            corpus.value_for(memory_relation, key) for key in corpus.holdout_keys
        )


def test_mixed_layout_removes_relation_identity_and_matches_compared_outputs():
    corpus = RuleMemoryCorpus(
        DataConfig(
            rule_family="digitwise",
            relation_layout="mixed",
            modulus=100,
            number_base=10,
            n_rule_relations=16,
            n_memory_relations=16,
        )
    )
    generalized_relations = {example.relation for example in corpus.examples_for_group("generalized")}
    memorized_relations = {example.relation for example in corpus.examples_for_group("memorized")}
    assert generalized_relations == memorized_relations == set(range(16))
    for relation in range(16):
        generalized = sorted(
            example.value for example in corpus.examples_for_group("generalized") if example.relation == relation
        )
        memorized = sorted(
            example.value for example in corpus.examples_for_group("memorized") if example.relation == relation
        )
        assert generalized == memorized


def test_natural_memory_corpus_has_matched_groups_and_counterfacts():
    config = DataConfig(corpus_kind="natural", relation_layout="mixed", modulus=40, n_rule_relations=4, n_memory_relations=4, seed=9)
    corpus = NaturalMemoryCorpus(config)
    assert {g: len(corpus.examples_for_group(g)) for g in corpus.groups} == {g: 40 for g in corpus.groups}
    assert len(corpus.eval_examples[0].input_ids) == 13
    assert corpus.prediction_positions == (10,)
    assert all(e.group in {"seen_rule", "memorized"} for e in corpus.train_examples)
    assert all(e.group == "seen_rule" for e in corpus.rule_train_examples)
    assert all(e.group == "memorized" for e in corpus.memory_train_examples)
    assert all(e.value != corpus._rule_value(e.relation, e.key) for e in corpus.examples_for_group("memorized"))
    for relation in range(4):
        generalized = sorted(e.value for e in corpus.examples_for_group("generalized") if e.relation == relation)
        memorized = sorted(e.value for e in corpus.examples_for_group("memorized") if e.relation == relation)
        nonmember = sorted(e.value for e in corpus.examples_for_group("nonmember") if e.relation == relation)
        assert generalized == memorized == nonmember
