# Exploratory scripts

Early data-exploration scripts, from before `step1_build_dataset.py`,
`step2_train.py` and the rest existed. They answer "what does this data
actually look like": per-class beat shapes, RR tachograms, Poincare plots and
lead comparisons. Some of the decisions in the root
[`README.md`](../../README.md) came out of these, like polarity normalisation
and picking the MLII lead by name.

They are not part of the training or deployment pipeline and you do not need
them to reproduce any result. Run from the repo root, e.g.
`python scripts/exploratory/inspect_leads.py`.
