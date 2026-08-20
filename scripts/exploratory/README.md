# Exploratory scripts

Early data-exploration scripts from before `step1_build_dataset.py` /
`step2_train.py` etc. existed. They answer "what does this data actually
look like" (per-class beat shapes, RR tachograms, Poincaré plots, lead
comparisons) and directly motivated decisions documented in the root
[`README.md`](../../README.md) (e.g. polarity normalisation, MLII lead
selection).

Not part of the training/deployment pipeline and not required to reproduce
any result — kept for transparency into how those decisions were reached.
Run from the repo root, e.g. `python scripts/exploratory/inspect_leads.py`.
