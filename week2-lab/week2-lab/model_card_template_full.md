# Model Card: <model name>

*Template after Mitchell et al., "Model Cards for Model Reporting" (2019).
Booklet Chapter 15. One page. If it runs to three, you are describing the code
rather than the decisions.*

## Model details
- Owner and contact:
- Version, date, and model type:
- Training date and data cut-off:

## Intended use
- Intended use:
- Intended users:
- **Out-of-scope uses** (be specific; this is the section auditors read first):

## The target variable
- What we actually care about, in plain language, with no reference to data:
- The column we predict:
- **How these differ, and who the difference falls on:**

## Training data
- Source, size, collection period:
- Datasheet reference:
- Groups present, and groups absent:

## Evaluation
| group | n | base rate | selection rate | accuracy | TPR | FPR | PPV |
|---|---|---|---|---|---|---|---|
| | | | | | | | |
| **ALL** | | | | | | | |

- Worst-group accuracy: (report this next to the average, always)
- Calibration error by group:

## Fairness
- **Criterion chosen:**
- **Why this criterion, given the harm:**
- **What we gave up to satisfy it, measured:**
- Who agreed this, and when:

## Threshold
- Value, and how it was chosen:
- Who bears a false positive:
- Who bears a false negative:

## Limitations and known failure modes

## Human oversight and contestability
- Who reviews, with how much time and information:
- How an affected person learns the system was used:
- Appeal route and timescale:

## Monitoring
- Subgroup metrics tracked, and review triggers:
- Named owner:
