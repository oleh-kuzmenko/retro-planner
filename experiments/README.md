# Experiments report

Порівняння 4 підходів до one-step ретросинтезу на двох фіксованих
тест-наборах по 100 реакцій (`data/uspto_eval_targets.json`,
`data/ord_eval_targets.json`): **ReactionT5v2** (seq2seq baseline, без RAG),
**CoT** (GPT-OSS-120B, chain-of-thought, без RAG), **RAG+CoT** (те саме +
Qdrant retrieval прикладів), **Hybrid** (ReactionT5v2 кандидати +
Qdrant retrieval + LLM-реранкінг). Дизайн і повний пайплайн — у [README.MD](../README.MD).

## Структура

```
experiments/
  ord/                          -- Open Reaction Database, 100 held-out targets
    reactiont5v2/
    cot_no_rag/
    rag_cot/
    hybrid_rag_rerank/
    final_aggregated_results.csv
    _archive/
      hybrid_rag_rerank_weak_index/   -- застарілий прогон, див. "Застереження" нижче
  uspto/                        -- USPTO-50k, 100 held-out targets
    reactiont5v2/
    cot_no_rag/
    rag_cot/
    hybrid_rag_rerank/
    final_aggregated_results.csv
```

Кожна модель-тека містить `run_meta.json` (CLI-параметри, час запуску, хеш
вхідного файлу), `results.json` (передбачення) і `inference_logs.json`
(повні промпти/відповіді LLM, де застосовно). `final_aggregated_results.csv`
в кожному датасеті об'єднує всі 4 моделі по одному тест-сету і містить
`exact_match` / `core_exact_match` / `valid` на кожен рядок.

Раніше теки називались за датою запуску (`2026-07-26`, `2026-07-26-ord`,
`RAG_USPTO`, `2026-07-26_500K_RAG` тощо) — це був дефолтний `--experiment-id`
скриптів, а не змістовна назва. Перейменовано за датасетом + моделлю, дані
(результати, логи, метрики) не змінювались.

## Результати (exact_match / core_exact_match / valid, зі 100)

| Модель | ORD | USPTO |
|---|---|---|
| ReactionT5v2 | 31% / 52% / 100% | 90% / 90% / 100% |
| CoT (no RAG) | 3% / 4% / 94% | 11% / 11% / 94% |
| RAG + CoT | 3% / 3% / 100% | 20% / 25% / 95% |
| Hybrid (T5 + RAG + rerank) | 27% / 48% / 100% | 71% / 76% / 100% |

ReactionT5v2 — найсильніший baseline на обох датасетах. Hybrid суттєво
покращує його на USPTO (не б'є, але наближається) і трохи погіршує на ORD.
Чистий LLM CoT (з RAG чи без) значно слабший за T5-підходи на точний збіг,
хоч і майже завжди валідний SMILES.

ChemLLM-20B і Qwen2.5-7B+LoRA (є в пайплайні, `colab/03_chemllm.ipynb` та
`colab/04_qwen_lora.ipynb`) ще не прогнані/не заагреговані локально —
результатів для них в `experiments/` немає.

## Застереження: якість Qdrant-індексу для ORD

Для ORD Qdrant спочатку індексували з `--max-per-source 100` — замало
унікальних реакцій на джерело, через що RAG-retrieval повертав майже
однакові дублікати одної реакції. Побачивши погані результати, індекс
перебудували з `--max-per-source 1000`.

- `ord/hybrid_rag_rerank/` — прогнано **після** переіндексації (хороший
  індекс). Це фінальний, "чесний" результат для hybrid на ORD.
- `ord/_archive/hybrid_rag_rerank_weak_index/` — той самий hybrid, але
  прогнаний **до** переіндексації (слабкий індекс, дублікати в retrieval,
  exact_match 24% / core 38% замість фінальних 27% / 48%). Залишено в
  архіві для історії, у `final_aggregated_results.csv` не враховується.
- `ord/rag_cot/` (чистий RAG+CoT, без T5) прогнаний **до** переіндексації і
  **не перезапускався** після виправлення індексу. Тобто його
  exact_match=3%/core=3% для ORD відображає слабкий індекс — це не чесне
  порівняння з `hybrid_rag_rerank`, який тестувався вже на виправленому.
  Якщо потрібне справедливе порівняння RAG+CoT vs Hybrid на ORD,
  `rag_cot` варто перезапустити на поточному індексі.
- Тестові дані (сам 100-таргетний eval-set) в обох прогонах ідентичні —
  перевірено по `input_sha256` в `run_meta.json`, збігається з поточним
  `data/ord_eval_targets.json`. Розбіжність лише в Qdrant-індексі, не в
  тестовому наборі.
- Для USPTO такої проблеми немає: `uspto/hybrid_rag_rerank` і
  `uspto/rag_cot` обидва прогнані вже після фінальної переіндексації.
