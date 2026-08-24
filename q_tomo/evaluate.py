from __future__ import annotations

import torch
import torch.nn.functional as F

from q_tomo.data import Example, RuleMemoryCorpus


@torch.inference_mode()
def batch_statistics(
    model: torch.nn.Module,
    examples: list[Example],
    corpus: RuleMemoryCorpus,
    device: torch.device,
) -> dict[str, torch.Tensor]:
    input_ids = torch.tensor([example.input_ids for example in examples], dtype=torch.long, device=device)
    target_digits = torch.tensor(
        [corpus.vocab.encode_number(example.value) for example in examples], dtype=torch.long, device=device
    )
    positions = list(corpus.prediction_positions)
    logits = model(input_ids)["logits"][:, positions, :].float()
    log_probabilities = logits.log_softmax(dim=-1)
    losses = -log_probabilities.gather(2, target_digits[:, :, None]).squeeze(2).mean(dim=1)
    target_logits = logits.gather(2, target_digits[:, :, None]).squeeze(2)
    masked = logits.clone()
    masked.scatter_(2, target_digits[:, :, None], float("-inf"))
    margins = (target_logits - masked.max(dim=-1).values).mean(dim=1)

    generated = input_ids.clone()
    predictions: list[torch.Tensor] = []
    for position in positions:
        step_logits = model(generated)["logits"][:, position, :].float()
        predicted = step_logits.argmax(dim=-1)
        predictions.append(predicted)
        generated[:, position + 1] = predicted
    predicted_digits = torch.stack(predictions, dim=1)
    correct = predicted_digits.eq(target_digits).all(dim=1)
    return {"nll": losses, "margin": margins, "correct": correct, "predicted_digits": predicted_digits}
