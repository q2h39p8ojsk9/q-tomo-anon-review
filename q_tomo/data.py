from __future__ import annotations

import json
import math
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

import torch
from torch.utils.data import Dataset

from q_tomo.config import DataConfig


PAD, BOS, EOS, KEY, ARROW = 0, 1, 2, 3, 4


@dataclass(frozen=True)
class Example:
    input_ids: tuple[int, ...]
    labels: tuple[int, ...]
    group: str
    relation: int
    key: int
    value: int
    is_member: bool
    source_index: int


@dataclass(frozen=True)
class Vocabulary:
    n_relations: int
    modulus: int
    number_base: int

    @property
    def digit_offset(self) -> int:
        return 5

    @property
    def relation_offset(self) -> int:
        return self.digit_offset + self.number_base

    @property
    def width(self) -> int:
        return max(1, math.ceil(math.log(self.modulus, self.number_base)))

    @property
    def size(self) -> int:
        return self.relation_offset + self.n_relations

    def relation_token(self, relation: int) -> int:
        return self.relation_offset + relation

    def digit_token(self, digit: int) -> int:
        return self.digit_offset + digit

    def encode_number(self, value: int) -> tuple[int, ...]:
        digits = [0] * self.width
        remainder = value
        for position in range(self.width - 1, -1, -1):
            digits[position] = remainder % self.number_base
            remainder //= self.number_base
        return tuple(self.digit_token(digit) for digit in digits)


class RuleMemoryDataset(Dataset[dict[str, torch.Tensor]]):
    def __init__(self, examples: Iterable[Example]):
        self.examples = list(examples)

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        example = self.examples[index]
        return {
            "input_ids": torch.tensor(example.input_ids, dtype=torch.long),
            "labels": torch.tensor(example.labels, dtype=torch.long),
            "index": torch.tensor(example.source_index, dtype=torch.long),
        }


class RuleMemoryCorpus:
    """Matched relation corpus with known generalization and membership labels.

    Every item has identical syntax: ``BOS REL KEY ARROW VALUE EOS``. Rule
    relations use an affine permutation over a prime-sized field. Memory
    relations use a random permutation. Both therefore have uniform value
    marginals and identical sequence lengths.
    """

    groups = ("seen_rule", "generalized", "memorized", "nonmember")

    def __init__(self, config: DataConfig):
        config.validate()
        self.config = config
        self.n_relations = (
            config.n_rule_relations
            if config.relation_layout == "mixed"
            else config.n_rule_relations + config.n_memory_relations
        )
        self.vocab = Vocabulary(self.n_relations, config.modulus, config.number_base)
        self._rng = random.Random(config.seed)
        self.rule_parameters = self._make_rule_parameters()
        self.relation_group_keys = self._split_mixed_keys() if config.relation_layout == "mixed" else None
        if self.relation_group_keys is None:
            self.train_keys, self.holdout_keys = self._split_keys()
        else:
            self.train_keys, self.holdout_keys = [], []
        self.memory_maps = self._make_memory_maps()
        self.eval_examples = self._make_eval_examples()
        self.train_examples = self._make_train_examples()

    def _make_rule_parameters(self) -> list[tuple[int, ...]]:
        if self.config.rule_family == "copy":
            return [(1, 0)] * self.config.n_rule_relations
        if self.config.rule_family == "digitwise":
            digit_affines = ((1, 0), (1, 1), (1, 3), (1, 5), (3, 0), (3, 1), (7, 0), (9, 9))
            return [
                (*digit_affines[index % len(digit_affines)], index // len(digit_affines))
                for index in range(self.config.n_rule_relations)
            ]
        parameters: list[tuple[int, int]] = []
        for _ in range(self.config.n_rule_relations):
            candidates = [a for a in range(1, self.config.modulus) if math.gcd(a, self.config.modulus) == 1]
            parameters.append((self._rng.choice(candidates), self._rng.randrange(self.config.modulus)))
        return parameters

    def _make_memory_maps(self) -> list[list[int]]:
        if self.config.relation_layout == "mixed":
            mappings: list[list[int]] = []
            assert self.relation_group_keys is not None
            for relation, keys_by_group in enumerate(self.relation_group_keys):
                values = [self._rule_value(relation, key) for key in range(self.config.modulus)]
                memorized_values = [self._rule_value(relation, key) for key in keys_by_group["generalized"]]
                nonmember_values = [self._rule_value(relation, key) for key in keys_by_group["nonmember"]]
                self._rng.shuffle(memorized_values)
                self._rng.shuffle(nonmember_values)
                for key, value in zip(keys_by_group["memorized"], memorized_values):
                    values[key] = value
                for key, value in zip(keys_by_group["nonmember"], nonmember_values):
                    values[key] = value
                mappings.append(values)
            return mappings
        mappings: list[list[int]] = []
        for _ in range(self.config.n_memory_relations):
            if self.config.rule_family in {"copy", "digitwise"}:
                # Preserve the train/holdout output marginals exactly. This
                # prevents answer-frequency or range cues from separating a
                # copied rule from a random association.
                memory_index = len(mappings)
                paired_rule = memory_index % self.config.n_rule_relations
                train_values = [self._rule_value(paired_rule, key) for key in self.train_keys]
                holdout_values = [self._rule_value(paired_rule, key) for key in self.holdout_keys]
                self._rng.shuffle(train_values)
                self._rng.shuffle(holdout_values)
                values = [0] * self.config.modulus
                for key, value in zip(self.train_keys, train_values):
                    values[key] = value
                for key, value in zip(self.holdout_keys, holdout_values):
                    values[key] = value
            else:
                values = list(range(self.config.modulus))
                self._rng.shuffle(values)
            mappings.append(values)
        return mappings

    def _split_keys(self) -> tuple[list[int], list[int]]:
        keys = list(range(self.config.modulus))
        self._rng.shuffle(keys)
        split = round(len(keys) * self.config.train_fraction)
        return sorted(keys[:split]), sorted(keys[split:])

    def _split_mixed_keys(self) -> list[dict[str, list[int]]]:
        result: list[dict[str, list[int]]] = []
        quarter = self.config.modulus // 4
        for _ in range(self.config.n_rule_relations):
            keys = list(range(self.config.modulus))
            self._rng.shuffle(keys)
            result.append(
                {
                    "seen_rule": sorted(keys[:quarter]),
                    "generalized": sorted(keys[quarter : 2 * quarter]),
                    "memorized": sorted(keys[2 * quarter : 3 * quarter]),
                    "nonmember": sorted(keys[3 * quarter :]),
                }
            )
        return result

    def value_for(self, relation: int, key: int) -> int:
        if relation < self.config.n_rule_relations:
            return self._rule_value(relation, key)
        memory_index = relation - self.config.n_rule_relations
        return self.memory_maps[memory_index][key]

    def _rule_value(self, relation: int, key: int) -> int:
        parameters = self.rule_parameters[relation]
        if self.config.rule_family == "digitwise":
            a, b, swap = parameters
            high, low = divmod(key, self.config.number_base)
            if swap:
                high, low = low, high
            high = (a * high + b) % self.config.number_base
            low = (a * low + b) % self.config.number_base
            return high * self.config.number_base + low
        a, b = parameters
        return (a * key + b) % self.config.modulus

    def _example(self, relation: int, key: int, group: str, index: int) -> Example:
        if self.config.relation_layout == "mixed" and group in {"memorized", "nonmember"}:
            value = self.memory_maps[relation][key]
        else:
            value = self.value_for(relation, key)
        key_tokens = self.vocab.encode_number(key)
        value_tokens = self.vocab.encode_number(value)
        tokens = (
            BOS,
            self.vocab.relation_token(relation),
            KEY,
            *key_tokens,
            ARROW,
            *value_tokens,
            EOS,
        )
        arrow_position = 3 + self.vocab.width
        labels_list = [-100] * len(tokens)
        # The arrow predicts the first digit; each teacher-forced digit predicts
        # the next. This is standard causal-LM supervision without target leak.
        for offset, token in enumerate(value_tokens):
            labels_list[arrow_position + offset] = token
        labels = tuple(labels_list)
        return Example(
            input_ids=tokens,
            labels=labels,
            group=group,
            relation=relation,
            key=key,
            value=value,
            is_member=group in {"seen_rule", "memorized"},
            source_index=index,
        )

    def _make_eval_examples(self) -> list[Example]:
        examples: list[Example] = []
        index = 0
        if self.config.relation_layout == "mixed":
            assert self.relation_group_keys is not None
            for relation, keys_by_group in enumerate(self.relation_group_keys):
                for group in self.groups:
                    for key in keys_by_group[group]:
                        examples.append(self._example(relation, key, group, index))
                        index += 1
            return examples
        for relation in range(self.n_relations):
            is_rule = relation < self.config.n_rule_relations
            for key in self.train_keys:
                group = "seen_rule" if is_rule else "memorized"
                examples.append(self._example(relation, key, group, index))
                index += 1
            for key in self.holdout_keys:
                group = "generalized" if is_rule else "nonmember"
                examples.append(self._example(relation, key, group, index))
                index += 1
        return examples

    def _make_train_examples(self) -> list[Example]:
        members = [example for example in self.eval_examples if example.is_member]
        repeated = members * self.config.exposures_per_pair
        self._rng.shuffle(repeated)
        return repeated

    def examples_for_group(self, group: str) -> list[Example]:
        if group not in self.groups:
            raise KeyError(f"unknown group: {group}")
        return [example for example in self.eval_examples if example.group == group]

    def metadata(self) -> dict[str, object]:
        return {
            "config": asdict(self.config),
            "vocab_size": self.vocab.size,
            "sequence_length": 5 + 2 * self.vocab.width,
            "number_width": self.vocab.width,
            "prediction_positions": list(self.prediction_positions),
            "rule_parameters": self.rule_parameters,
            "memory_maps": self.memory_maps,
            "train_keys": self.train_keys,
            "holdout_keys": self.holdout_keys,
            "relation_group_keys": self.relation_group_keys,
            "group_sizes": {group: len(self.examples_for_group(group)) for group in self.groups},
        }

    @property
    def prediction_positions(self) -> tuple[int, ...]:
        arrow_position = 3 + self.vocab.width
        return tuple(arrow_position + offset for offset in range(self.vocab.width))

    def save_metadata(self, path: str | Path) -> None:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open("w", encoding="utf-8") as handle:
            json.dump(self.metadata(), handle, indent=2, sort_keys=True)


@dataclass(frozen=True)
class NaturalVocabulary:
    n_relations: int
    n_people: int
    n_cities: int

    @property
    def digit_offset(self) -> int: return 15
    @property
    def relation_offset(self) -> int: return self.digit_offset + 10
    @property
    def city_offset(self) -> int: return self.relation_offset + self.n_relations
    @property
    def size(self) -> int: return self.city_offset + self.n_cities
    @property
    def width(self) -> int: return 1
    def relation_token(self, relation: int) -> int: return self.relation_offset + relation
    def person_tokens(self, person: int) -> tuple[int, int]: return (self.digit_offset + person // 10, self.digit_offset + person % 10)
    def encode_number(self, value: int) -> tuple[int, ...]: return (self.city_offset + value,)


class NaturalMemoryCorpus:
    """Small lexical fact/counterfact benchmark with matched within-topic groups.

    Prompts have the surface forms ``Where does resident number N from DISTRICT
    live?`` and ``Home city for resident N from DISTRICT is?``.
    Rule facts follow a learnable district/person mapping; memorized facts are
    counterfactual exceptions with relation-matched answer marginals.
    """
    groups = ("seen_rule", "generalized", "memorized", "nonmember")

    def __init__(self, config: DataConfig):
        config.validate(); self.config = config; self._rng = random.Random(config.seed)
        self.n_relations = config.n_rule_relations; self.n_cities = max(8, config.number_base)
        self.vocab = NaturalVocabulary(self.n_relations, config.modulus, self.n_cities)
        self.relation_group_keys = self._partition_keys()
        self.memory_maps = self._make_counterfacts()
        self.eval_examples = self._make_eval_examples()
        self.rule_train_examples, self.memory_train_examples, self.train_examples = self._make_train_examples()

    def _partition_keys(self):
        result = []
        for _ in range(self.n_relations):
            keys = list(range(self.config.modulus)); self._rng.shuffle(keys); q = len(keys) // 4
            result.append({"seen_rule": sorted(keys[:q]), "generalized": sorted(keys[q:2*q]), "memorized": sorted(keys[2*q:3*q]), "nonmember": sorted(keys[3*q:4*q])})
        return result

    def _rule_value(self, relation: int, key: int) -> int:
        # A learnable semantic rule: the last digit of the resident number
        # determines the canonical city. Counterfacts violate this rule.
        return key % self.n_cities

    def _make_counterfacts(self):
        maps = []
        for relation, groups in enumerate(self.relation_group_keys):
            targets = [self._rule_value(relation, k) for k in groups["generalized"]]
            mapping = {}
            for group in ("memorized", "nonmember"):
                vals = targets[:]
                for _ in range(1000):
                    self._rng.shuffle(vals)
                    if all(value != self._rule_value(relation, key) for key, value in zip(groups[group], vals)):
                        break
                else:
                    raise RuntimeError("could not construct matched natural-language counterfacts")
                mapping.update(zip(groups[group], vals))
            maps.append(mapping)
        return maps

    @property
    def prediction_positions(self) -> tuple[int, ...]: return (10,)

    def _example(self, relation: int, key: int, group: str, index: int, paraphrase: bool = False):
        value = self.memory_maps[relation][key] if group in {"memorized", "nonmember"} else self._rule_value(relation, key)
        tens, ones = self.vocab.person_tokens(key)
        if paraphrase:
            # BOS home city for resident DD from DISTRICT is ARROW CITY EOS
            tokens = (BOS, 11, 12, 13, 7, tens, ones, 9, self.vocab.relation_token(relation), 14, ARROW, *self.vocab.encode_number(value), EOS)
        else:
            # BOS where does resident number DD from DISTRICT live ARROW CITY EOS
            tokens = (BOS, 5, 6, 7, 8, tens, ones, 9, self.vocab.relation_token(relation), 10, ARROW, *self.vocab.encode_number(value), EOS)
        labels = [-100] * len(tokens); labels[10] = self.vocab.encode_number(value)[0]
        return Example(tokens, tuple(labels), group, relation, key, value, group in {"seen_rule", "memorized"}, index)

    def _make_eval_examples(self):
        examples = []; index = 0
        for relation, groups in enumerate(self.relation_group_keys):
            for group in self.groups:
                for key in groups[group]: examples.append(self._example(relation, key, group, index, group in {"generalized", "nonmember"})); index += 1
        return examples

    def _make_train_examples(self):
        seen = [e for e in self.eval_examples if e.group == "seen_rule"]
        memories = [e for e in self.eval_examples if e.group == "memorized"]
        seen_paraphrases = [self._example(e.relation, e.key, e.group, e.source_index, True) for e in seen]
        memory_paraphrases = [self._example(e.relation, e.key, e.group, e.source_index, True) for e in memories]
        rule_examples = (seen + seen_paraphrases) * self.config.exposures_per_pair
        memory_examples = (memories + memory_paraphrases) * self.config.exposures_per_pair
        epoch = (seen + seen_paraphrases) * self.config.natural_rule_multiplier + memories + memory_paraphrases
        repeated = epoch * self.config.exposures_per_pair
        self._rng.shuffle(rule_examples); self._rng.shuffle(memory_examples); self._rng.shuffle(repeated)
        return rule_examples, memory_examples, repeated

    def examples_for_group(self, group: str): return [e for e in self.eval_examples if e.group == group]
    def metadata(self):
        return {"config": asdict(self.config), "vocab_size": self.vocab.size, "sequence_length": 13, "prediction_positions": [10], "group_sizes": {g: len(self.examples_for_group(g)) for g in self.groups}, "surface_templates": ["Where does resident number N from DISTRICT live?", "Home city for resident N from DISTRICT is?"], "counterfactual_maps": self.memory_maps, "relation_group_keys": self.relation_group_keys}
    def save_metadata(self, path: str | Path):
        target = Path(path); target.parent.mkdir(parents=True, exist_ok=True); target.write_text(json.dumps(self.metadata(), indent=2, sort_keys=True), encoding="utf-8")


def build_corpus(config: DataConfig):
    return NaturalMemoryCorpus(config) if config.corpus_kind == "natural" else RuleMemoryCorpus(config)
