# Pre-registered hypothesis

Committed before any baseline result was inspected. The baseline runs were launched minutes
earlier and no output had been read when this file was committed.

## Hypothesis

Models recall EEG terminology correctly but fail when constraints must propagate across multiple
parameters simultaneously.

## Operational predictions

1. **Family ordering.** Accuracy will be highest on split_hygiene, the only family answerable from
   conceptual knowledge with no arithmetic, and lowest on pipeline_order and epoch_rejection, which
   require tracking many interacting constraints or table cells.
2. **Chain length.** Within pipeline_order, accuracy will fall with the number of steps. The
   held-out lengths, 6 to 7 steps, will score lower than the training lengths, 4 to 5 steps.
3. **Held-out axes.** Accuracy on the held-out split will be lower than on the train split for
   filter_design and power, whose held-out values fall outside the ranges most common in EEG
   practice.
4. **Scale.** Larger models will score higher overall, but the gap between split_hygiene and the
   multi-constraint families will persist at every size tested.
5. **Contamination.** Accuracy on the anchor split, which uses exact published or textbook values,
   will not be higher than on the train split by more than 10 percentage points for any family.
   A larger gap would suggest answers recalled from training data rather than computed.

## Analysis plan

- Metric: exact-match accuracy per family, split, and model. Errored rollouts count as wrong.
- Sample: 5 items per template per split (175 per split per model), seed 0, temperature 0.
- Uncertainty: 95% Wilson intervals on each accuracy. With 35 items per family-split cell, only
  differences larger than roughly 20 points are interpretable.
- Models: five open instruct models from 0.5B to 7B parameters, run locally at 4-bit quantization.
