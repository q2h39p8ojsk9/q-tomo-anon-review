import random

from q_tomo.pretrained import (
    TRAIN_TEMPLATES,
    UNSEEN_EVAL_TEMPLATES,
    PretrainedConfig,
    _counter_targets,
    build_pretrained_examples,
    paraphrase_pretrained_examples,
)


class FakeTokenizer:
    def encode(self, text, add_special_tokens=False):
        del add_special_tokens
        return [abs(hash(text)) % 10000]


def test_pretrained_design_is_within_relation_and_counterfactual():
    config = PretrainedConfig(n_relations=5, residents_per_relation=40, n_cities=4, include_rule_cue=True, seed=11)
    examples, metadata = build_pretrained_examples(config, FakeTokenizer())
    assert metadata["group_sizes"] == {group: 50 for group in ("seen_rule", "generalized", "memorized", "nonmember")}
    for relation in range(config.n_relations):
        relation_examples = [example for example in examples if example.relation == relation]
        assert {example.group for example in relation_examples} == {"seen_rule", "generalized", "memorized", "nonmember"}
        for example in relation_examples:
            if example.group in {"memorized", "nonmember"}:
                city_index = next(index for index, city in enumerate(metadata["cities"]) if city["text"] == example.answer)
                canonical = metadata["rule_maps"][str(relation)][example.resident % config.n_cities]
                assert city_index != canonical
                assert " badge" in example.prompt
        for exception_group, reference_group in (("memorized", "generalized"), ("nonmember", "seen_rule")):
            assert sorted(example.target_id for example in relation_examples if example.group == exception_group) == sorted(
                example.target_id for example in relation_examples if example.group == reference_group
            )


def test_counter_targets_are_valid_city_tokens_and_change_the_answer():
    config = PretrainedConfig(n_relations=5, residents_per_relation=40, n_cities=4, seed=7)
    examples, _ = build_pretrained_examples(config, FakeTokenizer())
    selected = [example for example in examples if example.group == "memorized"][:12]
    city_ids = sorted({example.target_id for example in examples})
    replacements = _counter_targets(selected, city_ids, seed=19)
    assert set(replacements.values()).issubset(city_ids)
    for example in selected:
        assert replacements[(example.relation, example.resident)] != example.target_id


def test_unseen_templates_preserve_examples_without_reusing_training_prompts():
    config = PretrainedConfig(n_relations=5, residents_per_relation=40, n_cities=4, include_rule_cue=True, seed=7)
    examples, _ = build_pretrained_examples(config, FakeTokenizer())
    assert not set(UNSEEN_EVAL_TEMPLATES.values()) & set(TRAIN_TEMPLATES)
    for name in UNSEEN_EVAL_TEMPLATES:
        paraphrased = paraphrase_pretrained_examples(config, examples, name)
        assert [example.target_id for example in paraphrased] == [example.target_id for example in examples]
        assert [example.group for example in paraphrased] == [example.group for example in examples]
        assert all(new.prompt != old.prompt for new, old in zip(paraphrased, examples))
