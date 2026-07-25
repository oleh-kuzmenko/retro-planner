# retro-eval

*[English](README.MD)*

Порівнює 4 підходи до передбачення ретросинтезу на одному фіксованому наборі
з 100 реакцій: ReactionT5v2, ChemLLM-20B, власну навчену модель Qwen2.5-7B +
LoRA та GPT-OSS-120B + Qdrant RAG + Chain-of-Thought. Кожен етап записує
самоописову теку прогону в `experiments/`, тож результати з GPU-ноутбука
(завантажені й розпаковані) та локального прогону об'єднуються в один CSV.

## Пайплайн

1. **Побудова тестового набору** (локально, без GPU) -- вибір 100 відкладених цілей:
   ```bash
   pip install -e ".[indexing]"
   python scripts/build_eval_targets_uspto.py            # -> data/uspto_eval_targets.json
   python scripts/build_eval_targets_ord.py               # -> data/ord_eval_targets.json
   ```
2. **ReactionT5v2** -- `colab/02_reactiont5v2.ipynb` (GPU, завантажте JSON з кроку 1, скачайте `experiments.zip`)
3. **ChemLLM-20B-Chat-SFT** (GGUF) -- `colab/03_chemllm.ipynb`, той самий підхід
4. **Qwen2.5-7B + LoRA** (власний навчений адаптер проєкту) -- `colab/04_qwen_lora.ipynb`, той самий підхід
5. **GPT-OSS-120B + Qdrant RAG + CoT** -- запускається локально (потрібні RAG-індекс і ключ
   до будь-якого OpenAI-сумісного хоста LLM):
   ```bash
   docker compose up -d qdrant
   python scripts/index_uspto_to_qdrant.py     # автоматично виключає цілі з кроку 1
   pip install -e ".[eval-runner]"
   python scripts/models/run_rag_cot_llm.py --input data/uspto_eval_targets.json \
       --base-url https://api.groq.com/openai/v1 --api-key $GROQ_API_KEY \
       --model openai/gpt-oss-120b
   ```
   `run_rag_cot_llm.py`/`run_cot_llm.py` працюють з будь-яким OpenAI-сумісним ендпоінтом
   через `--base-url`/`--api-key`/`--model` -- вбудованого дефолтного провайдера немає,
   тож впираєтесь у ліміт одного хоста? Просто вкажіть інший:
   ```bash
   python scripts/models/run_rag_cot_llm.py --input data/uspto_eval_targets.json \
       --base-url https://openrouter.ai/api/v1 --api-key $OPENROUTER_API_KEY \
       --model openai/gpt-oss-120b
   ```
   `run_cot_llm.py` пропускає RAG повністю (ті самі прапорці, Qdrant/індекс не потрібні) --
   напр. на наборі цілей ORD через Cerebras:
   ```bash
   python scripts/models/run_cot_llm.py --input data/ord_eval_targets.json \
       --base-url https://api.cerebras.ai/v1 --api-key $CEREBRAS_API_KEY \
       --model gpt-oss-120b
   ```
6. **Агрегація**: розпакуйте кожен завантажений `experiments.zip` у локальну `experiments/<experiment-id>/`, потім:
   ```bash
   python scripts/aggregate_results.py --input data/uspto_eval_targets.json
   # -> experiments/<experiment-id>/final_aggregated_results.csv
   ```

Кожен скрипт кроків 2-5 записує `experiments/<experiment-id>/<model-slug>/{results.json,
inference_logs.json, run_meta.json}` -- прогнози, точні запити/відповіді, надіслані
кожній моделі, та аргументи CLI/статус прогону. Передавайте однаковий `--experiment-id`
на кожному етапі, щоб вони групувалися в одне порівняння; за замовчуванням -- сьогоднішня дата.

## Локальне налаштування

```bash
python3 -m venv venv && source venv/bin/activate
pip install -e .
pip install -e ".[eval-runner]"     # потрібен для кожного скрипту scripts/models/run_*.py
pip install -e ".[local-models]"    # лише для run_reactiont5.py / run_qwen_lora_peft.py, якщо запускати локально, а не в Colab
pip install -e ".[indexing]"        # build_eval_targets_*.py, index_*.py
pip install -e ".[test]"            # pytest
```

Linux: `sudo apt-get install libxrender1 libxext6 libgl1`, якщо RDKit не знаходить GUI-бібліотеки.

## RAG-індекс (крок 5)

`index_uspto_to_qdrant.py` / `index_ord_to_qdrant.py` наповнюють дві колекції Qdrant
(`reactions_morgan` -- відбитки продуктів, `reaction_transforms` -- відбитки реакцій)
і за замовчуванням пропускають кожну реакцію, чий `product_smiles` є у відповідному
`data/*_eval_targets.json` -- тож тестовий набір з кроку 1 ніколи не потрапляє в RAG-індекс.
Спершу запустіть `build_eval_targets_*.py`. Кожен індексатор завжди видаляє й перестворює
обидві колекції; послідовний запуск індексаторів USPTO та ORD замінює, а не об'єднує дані,
якщо не вказати різні назви через `--collection`/`--transform-collection`.

```bash
python scripts/index_uspto_to_qdrant.py --limit 1000    # невеликий тестовий прогін
python scripts/index_uspto_to_qdrant.py                 # повний USPTO-50K (ліміт 500k, --limit 0 -- без ліміту)
python scripts/index_ord_to_qdrant.py --ord-data-dir /path/to/ord-data
```

ORD (2.4М+ реакцій у 550 файлах-джерелах в поточному знімку) завеликий, щоб індексувати
повністю для звичайних прогонів. `--max-per-source N` обмежує, скільки реакцій читається
з одного файлу-джерела перед переходом до наступного -- так кожен з внесених датасетів
залишається представленим замість того, щоб перші за сортуванням великі файли домінували
в індексі. Це той самий дисбаланс, від якого вже захищає `build_eval_targets_ord.py` для
eval-таргетів. `--max-per-source 100` дає ~54k реакцій (паритет з USPTO-50K) менш ніж за
хвилину, замість ~40+ хвилин на повне необмежене читання:

```bash
python scripts/index_ord_to_qdrant.py --max-per-source 100
```

## Розробка

```bash
python -m py_compile scripts/*.py scripts/models/*.py src/retro_eval/*.py src/retro_eval/harness/*.py src/retro_eval/providers/*.py
pip install -e ".[test]"
python -m pytest tests/
```

Тести не залежать від мережі/Qdrant/GPU.

## Дослідницькі матеріали

`research/fine-tune/v2/` -- ноутбуки, якими навчено LoRA-адаптери Qwen2.5-7B цього
проєкту (реагенти/клас + умови реакції), використовуються на кроці 4 та в `run_qwen_lora_peft.py`.
