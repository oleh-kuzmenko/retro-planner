# Experiments

Committed evaluation results (JSON) behind the tables in [`../RESULTS.md`](../RESULTS.md).
Each file is a `run_reactiont5_topk.py` / `evaluate_conditions_model_topk.py` /
`ensemble_topk.py` output: a `summary` block plus, for the topk runs, per-record
`records`.

| Folder | What |
|---|---|
| `v2_baselines/` | Published `ReactionT5v2` checkpoints (with / without USPTO fine-tuning) on ORD and USPTO — the "existing solutions" table |
| `v2_model1_eval/` | Model 1 (reactant) variant-2 top-k evals on ORD and USPTO |
| `v2_model1_topk/` | Model 1 top-k: every training variant (57k/150k/300k/root-aligned), the variant-2+4 ensemble, and the round-trip / diverse-beam checks that did not beat it |
| `v2_model2_topk/` | Model 2 (conditions) t5-small — summaries for the 41,139- and 138,869-reaction training sets on the shared 7,714-reaction test |

See `RESULTS.md` for the numbers and their interpretation.
