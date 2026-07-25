# retro-eval

*[Українською](README.uk.md)*

Compares 4 retrosynthesis-prediction approaches on the same fixed 100-reaction
test set: ReactionT5v2, ChemLLM-20B, this project's own trained Qwen2.5-7B +
LoRA, and GPT-OSS-120B + Qdrant RAG + Chain-of-Thought. Every stage writes a
self-describing run folder under `experiments/`, so results from a GPU
notebook (downloaded and unzipped) and a local run combine into one CSV.

## Pipeline

1. **Build the test set** (local, no GPU) -- pick 100 held-out targets:
   ```bash
   pip install -e ".[indexing]"
   python scripts/build_eval_targets_uspto.py            # -> data/uspto_eval_targets.json
   python scripts/build_eval_targets_ord.py               # -> data/ord_eval_targets.json
   ```
2. **ReactionT5v2** -- `colab/02_reactiont5v2.ipynb` (GPU, upload the JSON from step 1, download `experiments.zip`)
3. **ChemLLM-20B-Chat-SFT** (GGUF) -- `colab/03_chemllm.ipynb`, same pattern
4. **Qwen2.5-7B + LoRA** (this project's trained adapter) -- `colab/04_qwen_lora.ipynb`, same pattern
5. **GPT-OSS-120B + Qdrant RAG + CoT** -- run locally (needs the RAG index and an API key for
   any OpenAI-compatible LLM host):
   ```bash
   docker compose up -d qdrant
   python scripts/index_uspto_to_qdrant.py     # excludes step 1's targets automatically
   pip install -e ".[eval-runner]"
   python scripts/models/run_rag_cot_llm.py --input data/uspto_eval_targets.json \
       --base-url https://api.groq.com/openai/v1 --api-key $GROQ_API_KEY \
       --model openai/gpt-oss-120b
   ```
   `run_rag_cot_llm.py`/`run_cot_llm.py` talk to any OpenAI-compatible chat-completions
   endpoint via `--base-url`/`--api-key`/`--model` -- there's no baked-in default provider,
   so rate-limited on one host's free tier? Just point at another:
   ```bash
   python scripts/models/run_rag_cot_llm.py --input data/uspto_eval_targets.json \
       --base-url https://openrouter.ai/api/v1 --api-key $OPENROUTER_API_KEY \
       --model openai/gpt-oss-120b
   ```
   Skip RAG entirely with `run_cot_llm.py` (same flags, no Qdrant/index step needed) -- e.g.
   against the ORD target set on Cerebras:
   ```bash
   python scripts/models/run_cot_llm.py --input data/ord_eval_targets.json \
       --base-url https://api.cerebras.ai/v1 --api-key $CEREBRAS_API_KEY \
       --model gpt-oss-120b
   ```
6. **Aggregate**: unzip every downloaded `experiments.zip` into your local `experiments/<experiment-id>/`, then:
   ```bash
   python scripts/aggregate_results.py --input data/uspto_eval_targets.json
   # -> experiments/<experiment-id>/final_aggregated_results.csv
   ```

Every step 2-5 script writes `experiments/<experiment-id>/<model-slug>/{results.json,
inference_logs.json, run_meta.json}` -- predictions, the exact prompts/responses sent
to each model, and the run's CLI args/status. Pass the same `--experiment-id` to every
stage so they group into one comparison; it defaults to today's date.

## Local setup

```bash
python3 -m venv venv && source venv/bin/activate
pip install -e .
pip install -e ".[eval-runner]"     # every scripts/models/run_*.py script
pip install -e ".[local-models]"    # only run_reactiont5.py / run_qwen_lora_peft.py, if run locally instead of Colab
pip install -e ".[indexing]"        # build_eval_targets_*.py, index_*.py
pip install -e ".[test]"            # pytest
```

Linux: `sudo apt-get install libxrender1 libxext6 libgl1` if RDKit can't find GUI libs.

## RAG index (step 5)

`index_uspto_to_qdrant.py` / `index_ord_to_qdrant.py` populate two Qdrant collections
(`reactions_morgan` product fingerprints, `reaction_transforms` reaction fingerprints)
and, by default, skip every reaction whose `product_smiles` is in the matching
`data/*_eval_targets.json` -- so the step-1 test set never leaks into the RAG index.
Run `build_eval_targets_*.py` first. Each indexer always drops and recreates both
collections; running the USPTO and ORD indexer back to back replaces rather than merges
data unless you point `--collection`/`--transform-collection` at different names.

```bash
python scripts/index_uspto_to_qdrant.py --limit 1000    # small dev run
python scripts/index_uspto_to_qdrant.py                 # full USPTO-50K (500k cap, --limit 0 for no cap)
python scripts/index_ord_to_qdrant.py --ord-data-dir /path/to/ord-data
```

ORD (2.4M+ reactions across 550 source files in the current snapshot) is too large to
index in full for routine runs. `--max-per-source N` caps how many reactions are read
from any single source file before moving to the next one, keeping every contributed
dataset represented instead of letting whichever large file(s) sort first dominate --
the same imbalance `build_eval_targets_ord.py` already guards against for eval targets.
`--max-per-source 100` yields ~54k reactions (parity with USPTO-50K) in well under a
minute, instead of the ~40+ minutes a full unbounded read takes:

```bash
python scripts/index_ord_to_qdrant.py --max-per-source 100
```

## Development

```bash
python -m py_compile scripts/*.py scripts/models/*.py src/retro_eval/*.py src/retro_eval/harness/*.py src/retro_eval/providers/*.py
pip install -e ".[test]"
python -m pytest tests/
```

Tests have no network/Qdrant/GPU dependency.

## Research materials

`research/fine-tune/v2/` -- the notebooks that trained this project's Qwen2.5-7B LoRA
adapters (reactant/class + conditions), used by step 4 and by `run_qwen_lora_peft.py`.
