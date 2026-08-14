"""Vocabulary repair shared by `train_conditions_model.py` and `train_reactant_model_ord.py`.

Both models can be trained from a base checkpoint whose tokenizer predates the
data it is now being fine-tuned on, so both need the same `<unk>`-corruption fix.

Lives in the installed package rather than next to the scripts because
`train_reactant_model_ord.py` runs under `torchrun`, where a flat sibling import
would depend on how the launcher sets `sys.path[0]`.
"""

from __future__ import annotations


def ensure_full_char_coverage(tokenizer, model, texts, logger) -> int:
    """Add any character used in `texts` that the tokenizer would otherwise map to
    `<unk>` as a new single-character token, then resize the model's embedding matrix
    to match.

    Found by direct debugging of the ReactionT5-base Model 2 attempt (RESULTS.md,
    "Спроба: хімічно-передтренований базовий чекпоінт"): its 268-token vocabulary was
    built purely from SMILES chemistry notation, so most JSON punctuation (`{`, `}`,
    `"`, `:`, `,`) and most English letters -- both needed for this task's "not
    specified" / JSON-structured targets -- silently collapse to `<unk>`, a many-to-one
    mapping that destroys the training targets themselves. Teacher-forced `eval_loss`
    stayed healthy throughout (0.14) because the model was correctly learning to
    reproduce the *corrupted* (but self-consistent) `<unk>`-riddled targets; actual
    generation was 0% valid on all 2285 held-out records.

    Model 1 hits the mirror image of this with `sagawa/CompoundT5` as its base. That
    checkpoint is ReactionT5 one step before ORD reaction-pretraining, so it carries no
    ORD contamination -- but its 221-token vocabulary was trained on ZINC20, a library
    of single drug-like molecules. It therefore has no token for `.` (the fragment
    separator, 158,963 occurrences in this project's ORD/USPTO data), nor for the
    letters that spell out metals and counterions (`K`, `i`, `a`, `P`, `L`, `A`, `M`,
    `g`): 61,067 of 115,200 SMILES strings across `data/v2_ord_eval_targets.json`,
    `data/v2_uspto_eval_targets.json` and `data/v2_ord_train/reactants_train.jsonl`
    tokenize with at least one `<unk>`. Without this repair a CompoundT5-based Model 1
    cannot even emit a multi-fragment reactant set, so `exact_match` is 0 by
    construction. For reference on the same corpus: `sagawa/ReactionT5v2-retrosynthesis`
    hits 40 strings, `t5-small` 1,341.

    Only characters that genuinely tokenize to `<unk>` in isolation are added. An
    earlier version added *every* distinct character via `tokenizer.add_tokens(chars)`
    and relied on it skipping ones "already present" -- but for a SentencePiece
    tokenizer the vocab keys are metaspace-prefixed pieces (`"▁a"`), not bare
    characters, so `add_tokens("a")` treated every bare char as new. On t5-small (whose
    C4 vocab represents all these characters fine) that wrongly added 50+ redundant
    bare-char tokens, shadowing the model's learned sub-word tokenization. The correct
    test is whether the character actually encodes to the unk id: on t5-small that
    leaves just `{` and `}` (the only two genuinely missing), on ReactionT5 the ~30
    JSON/English characters that matter, on CompoundT5 the 30 listed above.

    Applying this to a base that barely needs it is safe, which is what lets Model 1 call
    it unconditionally: on `sagawa/ReactionT5v2-retrosynthesis` it adds 22 rows (268 ->
    290), and re-tokenizing all 114,000 strings of `data/v2_ord_train/reactants_*.jsonl`
    before and after shows exactly 40 differing -- the same 40 that previously carried an
    `<unk>`, each strictly repaired (`[La]`: `['[', '<unk>', ']']` -> `['[', 'L', 'a',
    ']']`). Multi-char pieces are not shadowed: `[Si]`, `[Na]`, `[Mg+2]` still tokenize
    to `Si`/`Na`/`Mg`. The other rows stay unused, so a ReactionT5 run's recipe is
    unchanged apart from a wider (partly dead) embedding matrix.

    `mean_resizing=False` is required. The transformers default (`mean_resizing=True`)
    initializes the new rows by sampling a multivariate normal fitted to the *existing*
    embeddings' mean and covariance. On Kaggle's transformers version (newer lazy
    "Materializing param" weight loader) this produced pathological large-norm rows:
    the first held-out eval_loss started at 10.44 -- worse than uniform-random
    (ln(vocab)=5.74) -- and never recovered below ~8.5 over 4 full epochs. The exact
    same code/optimizer/lr on this machine's transformers learned normally (eval_loss
    4.93 -> 3.48 in 120 Trainer steps), so the fault is that version-specific covariance
    sampling, not the resize or the training recipe. `mean_resizing=False` skips the
    covariance fit entirely (simple small-normal init), which is version-robust.
    """
    chars = sorted({ch for text in texts for ch in text})
    unk_id = tokenizer.unk_token_id
    missing = [c for c in chars if unk_id in tokenizer(c, add_special_tokens=False)["input_ids"]]
    added = tokenizer.add_tokens(missing)
    if added:
        model.resize_token_embeddings(len(tokenizer), mean_resizing=False)
        logger.info(
            "Added %d new character token(s) to vocab (size now %d) to fix <unk> "
            "corruption of the training data: %s",
            added,
            len(tokenizer),
            missing,
        )
    return added
