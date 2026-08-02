# retro-planner

*[English](README.MD)*

Інтелектуальна система для **ретросинтезу органічних сполук**. Дві незалежно навчені
моделі, обидві на Open Reaction Database (ORD); USPTO-50K використовується як незалежний
тест на узагальнення:

- **Модель 1 — передбачення реагентів.** Дотренований `ReactionT5v2` (T5 з хімічним
  претренуванням), що передбачає SMILES реагентів за SMILES продукту.
- **Модель 2 — передбачення умов реакції.** `t5-small`, що передбачає розчинник,
  каталізатор, температуру й вихід для пари продукт/реагенти.

Повна методологія й усі виміряні числа — у **[RESULTS.md](RESULTS.md)**. Головне:
Модель 1 досягає **82,3% top-5 `core_exact_match` на ORD** за рахунок об'єднання
beam-кандидатів двох незалежно дотренованих чекпоінтів; Модель 2 покращується на кожному
полі при навчанні на 138 869 (проти 41 139) дедупльованих реакціях без витоку.

## Структура репозиторію

| Шлях | Вміст |
|---|---|
| `scripts/` | Побудова даних, навчання та оцінювання |
| `scripts/models/` | Інференс/оцінювання Моделі 1: top-k beam search, об'єднання, round-trip переранжування |
| `src/retro_eval/` | Бібліотека: канонізація SMILES, метрики exact/core-match, схожість умов |
| `kaggle/`, `colab/` | GPU-ноутбуки, що запускають скрипти навчання/оцінювання |
| `experiments/` | Збережені результати оцінювання (JSON) — див. `experiments/README.md` |
| `tests/` | Юніт-тести бібліотеки метрик |
| `RESULTS.md` | Звіт: усі таблиці й висновки |

## Пайплайн

Великі похідні спліти в `data/` — у gitignore й регенеруються детерміновано
(фіксований seed); у git зберігаються лише малі eval-target JSON.

**1. Побудова даних** (потрібен ORD через Hugging Face; без GPU):

```bash
pip install -e ".[indexing]"
python scripts/build_eval_targets_ord.py                  # тест Моделі 1 (data/v2_ord_eval_targets.json)
python scripts/build_train_data_ord.py --pool-count 60000 # спліти train/val(/test) для реагентів і умов
python scripts/build_root_aligned_data.py \               # опційно: root-aligned ціль реагентів (варіант 6 Моделі 1)
    --input data/v2_ord_train/reactants_train.jsonl \
    --output data/v2_ord_train_rootaligned/reactants_train.jsonl
```

**2. Навчання** (GPU — через ноутбуки `kaggle/`/`colab/`, що викликають ці скрипти):

```bash
# Модель 1 (реагенти): дотренування ReactionT5v2 на ORD
python scripts/train_reactant_model_ord.py --no-augment --learning-rate 5e-5 --num-train-epochs 3 ...
# Модель 2 (умови): навчання t5-small на ORD
python scripts/train_conditions_model.py --base-model t5-small --learning-rate 5e-4 ...
```

**3. Оцінювання**:

```bash
# Модель 1: top-1/3/5 exact і core-exact
python scripts/models/run_reactiont5_topk.py --input data/v2_ord_eval_targets.json --t5-model <ckpt> --num-beams 10 --output out.json
# об'єднання кандидатів двох чекпоінтів
python scripts/models/ensemble_topk.py --primary a.json --secondary b.json --output ensemble.json
# Модель 2: строгі й послаблені метрики по кожному полю
python scripts/evaluate_conditions_model_topk.py --model-dir <ckpt> --test-file data/v2_ord_train_300k/conditions_test.jsonl --num-beams 10 --output out.json
```

## Локальне налаштування

```bash
python3 -m venv venv && source venv/bin/activate
pip install -e ".[local-models,indexing]"   # torch/transformers + інструменти даних ORD
pip install -e ".[test]"                     # pytest
```

`build_root_aligned_data.py` додатково потребує `pip install rxnmapper "setuptools<81"`.
Linux: `sudo apt-get install libxrender1 libxext6 libgl1`, якщо RDKit не знаходить GUI-бібліотеки.

## Тести

```bash
python -m pytest tests/
```

Без залежностей від мережі/GPU.
