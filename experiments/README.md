# Experiments report

Порівняння підходів до one-step ретросинтезу на фіксованих тест-наборах по
100 реакцій. Дизайн і пайплайн — у [README.MD](../README.MD).

## Результати (exact_match / core_exact_match / valid, зі 100)

| Модель | ORD | USPTO |
|---|---|---|
| ReactionT5v2 | 31% / 52% / 100% | 90% / 90% / 100% |
| ChemLLM-20B (GGUF, Q4_K_M) | 1% / 2% / 45% | 2% / 2% / 41% |
| CoT (GPT-OSS-120B, без RAG) | 3% / 4% / 94% | 11% / 11% / 94% |
| RAG + CoT | 3% / 3% / 100% | 20% / 25% / 95% |
| Hybrid (T5 + RAG + rerank) | 27% / 48% / 100% | 71% / 76% / 100% |
| Hybrid, застарілий слабкий індекс (`_archive`) | 24% / 38% / 100% | — |

## LoRA fine-tune (Qwen2.5-7B) і чесні held-out спліти

Первинні `uspto_eval_targets.json` / `ord_eval_targets.json` семплюються
незалежно від train/test поділу, на якому реально тренувались
ReactionT5v2 і LoRA-адаптер — тобто містять memorization, а не
generalization (детальніше в докстрінгах `scripts/build_eval_targets_*_holdout.py`).
Тому для цих двох моделей є окремі, гарантовано непересічні з тренуванням
тест-набори:

| Модель | Тест-набір | exact_match / core / valid |
|---|---|---|
| ReactionT5v2 | канонічний USPTO-50k test-спліт (`uspto_test_holdout`) | 91% / 92% / 100% |
| Qwen2.5-7B + LoRA | власний held-out спліт (`uspto_lora`) | 35% / 35% / 97% |
| Qwen2.5-7B + LoRA | ORD, крос-датасет (`lora_ord_eval`) | 7% / 10% / 95% |

ReactionT5v2 на чесному test-спліті лишається на рівні заявленого в
статті (~90%) — це не memorization, модель просто тренувалась саме на
цей розподіл. LoRA-адаптер на власному held-out — 35%, і різко просідає
(7%) поза межами домену тренування (ORD) — очікуваний ефект
доменно-специфічного fine-tune.
