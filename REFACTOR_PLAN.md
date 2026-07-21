# План рефакторингу retro-planner під пояснювальну записку

Джерело істини: `ПЗ_Кузьменко.pdf` — "Інтелектуальна система для ретросинтезу органічних сполук"
(дипломна робота бакалавра, спеціальність "Системи і методи штучного інтелекту").

Цей документ фіксує (1) конкретні розбіжності між тим, що описано в ПЗ, і тим, що реалізовано
в репозиторії зараз, (2) цільову архітектуру, яка закриває ці розбіжності і водночас залишає
місце для підключення інших моделей (як прямо просив автор), і (3) покроковий план виконання
рефакторингу файл за файлом.

Рефакторинг великий і зачіпає майже кожен модуль `src/retro_planner/`. План розбитий на фази,
кожна з яких дає робочий застосунок — можна зупинитися після будь-якої фази й мати систему,
що частково або повністю відповідає ПЗ.

---

## 1. Що саме описано в ПЗ (стисло, з посиланням на розділи)

| Розділ ПЗ | Вимога |
|---|---|
| 1.4, 2.4 (стор. 10-22) | Гібридна **конвеєрна** (не graph-search) архітектура: RAG + Prompt Engineering + CoT навколо однієї авторегресійної LLM. Явна відмова від AND-OR дерев і багатокрокового пошуку — уся обчислювальна вага йде на якість **одного** ретросинтетичного кроку. |
| 3.1, рис. 3.1 (стор. 23-24) | Конвеєр: `Вхідний інтерфейс → [Модуль RAG || Prompt Engineering] → LLM → Парсинг результатів (think/answer теги) → Вихід (обґрунтування + прекурсори)`. |
| 3.2, рис. 3.2 (стор. 26-30) | RAG: Morgan fingerprint цілі → паралельний пошук у двох колекціях Qdrant (`reactions_morgan`, `reaction_transforms`, остання = `FP_product XOR FP_reactant`) → **Tanimoto similarity** → гібридна оцінка = `product score + transform score` → top-k з метаданими ORD/USPTO. |
| 3.3 (стор. 30-32) | Суворий 4-блоковий шаблон промпту: `[System] / [Context: RAG_Examples] / [Instruction: CoT-вимога] / [Input: Target_SMILES]`. Відповідь LLM: увесь аналіз — усередині `<think>...</think>`, **тільки** SMILES реагентів через крапку — усередині `<answer>...</answer>`, без жодного іншого тексту в тегах. |
| 3.4, рис. 3.4 (стор. 33-35) | Послідовність: (1) валідація/канонізація SMILES (RDKit) → (2) RAG retrieval (Qdrant, k-NN) → (3) побудова промпту → (4) виклик LLM API (LLaMA/Qwen через OpenAI-сумісний інтерфейс) → (5) парсинг `<think>`/`<reason>` і `<answer>` тегів, хімічна валідація прекурсорів (валентності, збереження маси), візуалізація реакції + окремий блок тексту міркувань. |
| 4.1-4.2 (стор. 36-40) | Оцінювання на USPTO-50K: Top-1/3/5 exact match, Structure Success Rate (частка SMILES, що парсяться RDKit), порівняння Zero-shot проти RAG+CoT. Таблиця 4.1 і рис. 4.1 у ПЗ мають плейсхолдери `[ВСТАВИТИ %]` — цифр ще немає, їх має дати ваш власний eval-прогін. |
| 4.3 (стор. 40-43) | Окремий інтерфейсний блок **"Chemist's Reasoning"**, що показує розгорнутий текст із `<think>` користувачу для довіри й верифікації. |
| Перелік посилань, п.12 | Практична реалізація названа `retro-planner` — тобто цей репозиторій сам є артефактом захисту, а не абстрактний опис. |

## 2. Конкретні розбіжності з поточним кодом

| # | Що каже ПЗ | Що є в коді зараз | Файл(и) |
|---|---|---|---|
| G1 | Вивід LLM = `<think>…</think>` + `<answer>…</answer>`, без JSON | `json_mode=True`, парситься `json.loads`, схема `ROUTES_JSON_SCHEMA` з 15+ полями на крок | [planning.py:216-247](src/retro_planner/planning.py#L216-L247), [prompts.py:24-55](src/retro_planner/prompts.py#L24-L55), [llm_providers.py:82-110](src/retro_planner/llm_providers.py#L82-L110) |
| G2 | Один ретросинтетичний крок за генерацію (Розділ 2.4: "зосередити зусилля на однокроковому аналізі") | `route_count` (1-5) — LLM просять одразу кілька паралельних `routes[]` в одному виклику | [planning.py:24-25](src/retro_planner/planning.py#L24-L25), [prompts.py:58-63](src/retro_planner/prompts.py#L58-L63) |
| G3 | 4-блоковий шаблон промпту `[System]/[Context]/[Instruction]/[Input]`, суворо мінімальний | Промпт із 15 "CRITICAL RULES", вимогами до stoichiometry/atmosphere/workup/objective_fit тощо — інша структура і мета | [prompts.py:73-195](src/retro_planner/prompts.py#L73-L195) |
| G4 | Метрика подібності — коефіцієнт **Танімото** | Qdrant-колекції створені з `Distance.COSINE`; реальний Танімото ніде не рахується | [scripts/index_uspto50k_to_qdrant.py:193](scripts/index_uspto50k_to_qdrant.py#L193) |
| G5 | Гібридна оцінка = **два** компоненти (product score + transform score) | Додано третій вигаданий компонент `reaction_class_similarity` (SMARTS-евристики), якого в ПЗ немає | [retrieval.py:134-178](src/retro_planner/retrieval.py#L134-L178), [reaction_classes.py](src/retro_planner/reaction_classes.py), [config.py:14-21](src/retro_planner/config.py#L14-L21) |
| G6 | Парсинг відповіді додатково хімічно валідує прекурсори (валентність, збереження маси) | Валідація перевіряє тільки збіг `product_smiles` із ціллю; окремої mass-balance перевірки нема | [planning.py:44-55](src/retro_planner/planning.py#L44-L55) |
| G7 | Окремий UI-блок **"Chemist's Reasoning"** з текстом міркувань | Такого блоку немає; є `rationale`/`objective_fit` як текстові поля всередині route-картки | [streamlit_views.py:190-313](src/retro_planner/streamlit_views.py#L190-L313) |
| G8 | LLM = "LLaMA або Qwen через OpenAI-сумісний інтерфейс" (локальний деплой) | Є `custom_openai` провайдер (це і є той інтерфейс), але дефолтний/головний у UI — Groq | [llm_providers.py:311-341](src/retro_planner/llm_providers.py#L311-L341) |
| G9 (вимога користувача, не з ПЗ) | Потрібне "місце для гнучкої інтеграції з іншими моделями" — включно з моделями, які вже фактично тестуються в `eval.md` (ReactionT5v2 seq2seq, ChemLLM GGUF, Qwen LoRA two-stage) | Ці моделі існують лише як одноразовий код у ноутбуках `fine-tune/v2/res/*.ipynb`, недоступні через `LLMProvider`-протокол і UI | `fine-tune/v2/res/06_agent_inference_demo.ipynb`, `08_reactiont5_conditions_agent_demo.ipynb`, [eval.md:1-40](eval.md#L1-L40) |
| G10 | Розділ 4: кількісне порівняння Zero-shot vs RAG+CoT (Top-k, Structure Success Rate) на USPTO-50K | Є лише ручний якісний eval.md на ~десятку молекул, без автоматизованого підрахунку метрик | [eval.md](eval.md) |

Пункти, які код уже **робить правильно** й чіпати не треба:
- Дві Qdrant-колекції з точними назвами з ПЗ (`reactions_morgan`, `reaction_transforms`) — [config.py:5-6](src/retro_planner/config.py#L5-L6).
- Формула трансформ-вектора `FP_transform = FP_product XOR FP_reactant` — точний збіг з формулою на стор. 27 — [chemistry.py:141-154](src/retro_planner/chemistry.py#L141-L154).
- RDKit-канонізація на вході (Крок 1 у рис. 3.4) — [chemistry.py:60-79](src/retro_planner/chemistry.py#L60-L79).
- Провайдерний реєстр (`LLMProviderConfig`, `LLM_PROVIDER_REGISTRY`) — хороша основа для вимоги "гнучкої інтеграції", просто потребує розширення на локальні/некат-API моделі — [llm_providers.py:39-51,311-341](src/retro_planner/llm_providers.py#L39-L51).

---

## 3. Цільова архітектура

```
Користувач ──> Streamlit UI ──> canonicalize_smiles (RDKit)
                                        │
                     ┌──────────────────┼──────────────────┐
                     ▼                                      ▼
              RAG retrieval                         Prompt Engineering
       (Qdrant: reactions_morgan +                  (4-блоковий CoT-шаблон,
        reaction_transforms, Tanimoto,               Context = RAG hits,
        product+transform score)                     Instruction = think/answer)
                     │                                      │
                     └──────────────────┬───────────────────┘
                                         ▼
                            LLMProvider.generate(...)
                    (registry: chat API / local HF / GGUF)
                                         ▼
                         reasoning.parse_response()
                 (<think>|<reason> ... <answer> SMILES.SMILES </answer>)
                                         ▼
                    Хімічна валідація (RDKit: валідність + target match)
                                         ▼
                    UI: "Chemist's Reasoning" + схема реакції + прекурсори
```

Ключова архітектурна ідея: **одна генерація = один ретросинтетичний крок**, точно як у ПЗ.
"Кілька варіантів маршруту" у UI реалізується не проханням до LLM повернути JSON-масив, а
**повторним викликом того самого одно-крокового CoT-конвеєра** (наприклад, з різним
seed/temperature або різними top-k precedents). Це чесніше відповідає діаграмі 3.1/3.4 ПЗ
(там немає гілки "route_count"), і кожен окремий виклик і далі парситься тим самим
`<think>/<answer>` контрактом — код валідації один, просто викликається N разів.

### 3.1. Контракт відповіді LLM (нове ядро)

**Мовне правило: увесь текст, що йде в тіло промпту (system/context/instruction/input),
повинен бути англійською** — так само, як зараз в існуючому `prompts.py`. Українською
залишаються лише цей план, докстрінги/коментарі в коді (за потреби) та UI-лейбли Streamlit;
сам текст, який бачить LLM, — завжди англійська. Це важливо і тому, що навчальні дані
USPTO-50K/ORD, назви реакцій і хімічна номенклатура в промптах природно англомовні, і тому,
що змішування мов у промпті шкодить якості генерації.

Переклад шаблону зі стор. 32 ПЗ (`[System]/[Context]/[Instruction]/[Input]`) англійською —
саме це піде в код:

```
[System] You are an expert organic chemist. Perform a single-step retrosynthetic analysis
for the given target molecule.
[Context] The following similar reaction precedents were retrieved from the database:
{RAG_Examples}
[Instruction] Analyze the target molecule step by step inside <think>...</think> tags:
identify functional groups, likely reaction centers, and the thermodynamic/kinetic
feasibility of the proposed bond disconnection. After that, output only the reactant
SMILES strings separated by a dot inside <answer>...</answer> tags, with no other text.
[Input] Target molecule (SMILES): {Target_SMILES}
```

Це нова функція `build_cot_prompt(...)`, яка замінить `build_rag_prompt`/
`build_no_rag_system_prompt` у `prompting.py` (перейменований з `prompts.py`). Increase у
Фазі 1 треба явно перевірити тестом (`tests/test_prompting.py`), що жодна з англомовних
рядків-констант шаблону не містить кириличних символів — щоб випадкова майбутня правка
(наприклад, копіпаст із цього ж українського плану) не просочила українську мову в промпт.

### 3.2. Розширюваність під інші моделі (вимога користувача, не суперечить ПЗ)

ПЗ називає ядро "LLaMA або Qwen через OpenAI-сумісний інтерфейс" — це вже `custom_openai`
провайдер. Але `eval.md` показує, що реально тестуються ще й моделі поза чат-API
(ReactionT5v2 — seq2seq з transformers, ChemLLM GGUF — llama-cpp-python, Qwen LoRA —
peft-адаптери). Зараз ця логіка живе тільки в ноутбуках. Ціль рефакторингу — звести все під
один `Protocol`, щоб UI, планувальник і eval-скрипт працювали з будь-якою моделлю однаково:

```python
class LLMProvider(Protocol):
    def generate(self, messages: list[dict], model: str,
                 temperature: float, json_mode: bool = False) -> str: ...
```

`json_mode` для нового контракту здебільшого ігнорується (не потрібен — тегований текстовий
вивід і так детермінований у форматі), але лишається в сигнатурі заради сумісності зі старими
чат-провайдерами, які, можливо, ще використовуються для інших задач.

Нові адаптери під `src/retro_planner/providers/`:
- `chat_api.py` — Groq / OpenAI / будь-який OpenAI-сумісний ендпоінт (перенесення з `llm_providers.py`, без змін логіки).
- `local_seq2seq.py` — обгортка над `sagawa/ReactionT5v2-retrosynthesis*` (HF `transformers`, `generate()`), повертає прекурсори без `<think>` — парсер повинен приймати "тільки-answer" відповіді як legacy-режим.
- `local_causal.py` — двоетапний Qwen2.5-7B LoRA (reactants/class-адаптер + умови-адаптер) з `06_agent_inference_demo.ipynb`, обгорнутий у той самий `generate()`.
- `local_gguf.py` — ChemLLM-20B GGUF через `llama-cpp-python`, з `base_models_res/ChemLLM-20B-Chat.ipynb`.

Додавання нової моделі в майбутньому = один клас з методом `generate()` + один запис у
`LLM_PROVIDER_REGISTRY`. Жодних змін у `planning.py`, `retrieval.py`, `streamlit_views.py`.
Важкі залежності (`transformers`, `peft`, `llama-cpp-python`, `torch`) підключаються як
окремий extras-набір `[local-models]` у `pyproject.toml` (за зразком уже наявного `[indexing]`),
щоб не обтяжувати базовий Docker-образ хмарного UI.

---

## 4. Мапа файлів: що видаляється / переписується / додається

| Поточний файл | Дія | Новий файл / роль |
|---|---|---|
| `src/retro_planner/prompts.py` | переписати | `src/retro_planner/prompting.py` — 4-блоковий CoT-шаблон (§3.1), repair-шаблон під той самий тег-контракт |
| `src/retro_planner/planning.py` | переписати | той самий шлях — оркестрація: retrieval → prompting → provider.generate → reasoning.parse → валідація → опційний repair |
| `src/retro_planner/llm_providers.py` | розділити | `src/retro_planner/providers/__init__.py` (Protocol + реєстр) + `providers/chat_api.py` (перенесений існуючий код) |
| — (нове) | створити | `src/retro_planner/reasoning.py` — парсинг `<think>/<reason>` + `<answer>`, хімічна валідація прекурсорів (валентність + груба mass-balance перевірка) |
| — (нове) | створити | `src/retro_planner/providers/local_seq2seq.py`, `local_causal.py`, `local_gguf.py` |
| `src/retro_planner/retrieval.py` | переписати скоринг | реальний Tanimoto (RDKit `BulkTanimotoSimilarity`/побітовий intersection/union) замість Qdrant-Cosine як фінальної цифри; дефолтна гібридна формула — точно 2 компоненти (product + transform), вага `reaction_class` за замовчуванням = 0 |
| `src/retro_planner/reaction_classes.py` | звузити роль | лишити файл, але підключати лише як опційний `rerankers.py`-крок, вимкнений за замовчуванням (задокументувати як "розширення понад ПЗ") |
| `src/retro_planner/config.py` | доповнити | прибрати з дефолту вагу reaction_class (або позначити як `EXPERIMENTAL_RETRIEVAL_WEIGHTS`), додати константи тегів (`THINK_TAGS = ("think", "reason")`, `ANSWER_TAG = "answer"`) |
| `src/retro_planner/streamlit_views.py` | переписати | додати блок **"Chemist's Reasoning"** (рендер сирого `<think>`-тексту), прибрати route/JSON-специфічний рендер, додати "Generate another candidate" для повторної одно-крокової генерації |
| `src/retro_planner/app.py` | оновити виклики | під нову сигнатуру `planning.py`/`streamlit_views.py`; провайдер-селектор у сайдбарі групує чат-API та локальні провайдери |
| `scripts/index_uspto50k_to_qdrant.py` | без змін структури | коментар/докстрінг про те, що Cosine у Qdrant — лише для ANN-кандидатів, фінальний Tanimoto рахується в `retrieval.py` |
| — (нове) | створити | `scripts/evaluate_retrosynthesis.py` — автоматичний прогін Zero-shot vs RAG+CoT на USPTO-50K test split, рахує Top-1/3/5 + Structure Success Rate, друкує markdown-таблицю у форматі Табл. 4.1 ПЗ |
| — (нове) | створити | `src/retro_planner/evaluation.py` — перевикористовувані функції метрик (`top_k_exact_match`, `structure_success_rate`), імпортуються і в `scripts/evaluate_retrosynthesis.py`, і потенційно в тестах |
| — (нове) | створити | `tests/test_reasoning.py`, `tests/test_prompting.py`, `tests/test_retrieval_scoring.py`, `tests/test_chemistry.py` |
| `docs/system_diagrams.md` | оновити | перемалювати схеми 3-7 під новий tag-based конвеєр (зараз описують JSON/routes-архітектуру) |
| `AGENTS.md` | оновити | розділ "Project Overview" і "Development Guidance" — замінити опис JSON-контракту на think/answer-контракт, додати правила для `providers/` |
| `eval.md` | зберегти як історичний журнал, додати посилання на автоматизований `scripts/evaluate_retrosynthesis.py` | — |
| `pyproject.toml` | доповнити | новий extras `[local-models]` (`transformers`, `peft`, `llama-cpp-python`, `torch`), опційно `[test]` (`pytest`) |

---

## 5. Фазований план виконання

### Фаза 0 — Підготовка (без змін логіки)
- [x] ~~Створити гілку `refactor/pz-alignment`~~ — рефакторинг виконано напряму на `main` (одноосібний навчальний прототип без CI/PR-процесу); зафіксовано послідовними комітами.
- [x] ~~Зафіксувати поточну поведінку скріншотами~~ — регрес-порівняння зроблено ручною перевіркою в headless Chrome під час Фази 4 (див. нотатку в Фазі 4) замість окремих скріншотів "до".
- [x] Додати `pytest` у dev-залежності (`[test]` extras у `pyproject.toml`), каркас `tests/` із 44 тестами (`test_reasoning.py`, `test_prompting.py`, `test_retrieval_scoring.py`, `test_providers.py`, `test_evaluation.py`), усі проходять (`python -m pytest tests/`).

### Фаза 1 — Ядро мислення/відповіді (найбільша зміна, G1-G3, G6)
- [x] `reasoning.py`: `parse_reasoning_response(text) -> ReasoningResult(think, answer_smiles, raw)` з підтримкою обох тегів `<think>`/`<reason>`, толерантний fallback (немає тегів → трактувати весь текст як `<answer>`, лог warning).
- [x] `reasoning.py`: `validate_precursors(answer_smiles, target_smiles)` — RDKit-парсинг кожного SMILES (валідність), груба mass-balance перевірка (сума важких атомів прекурсорів проти продукту з допуском `LEAVING_GROUP_HEAVY_ATOM_TOLERANCE` на leaving groups).
- [x] `prompting.py`: `build_cot_prompt(target_smiles, rag_examples) -> str` — точний 4-блоковий шаблон з §3.1, текст промпту англійською; `build_cot_repair_prompt(...)` — той самий контракт для невалідного результату.
- [x] `planning.py`: нова `generate_single_step(request) -> StepResult(think, precursors, product_smiles, raw_response, warnings, errors)`; стара `GenerationRequest`/`PlanResult`/`route_count`/JSON-гілка видалені.
- [x] Юніт-тести `tests/test_reasoning.py`, `tests/test_prompting.py` (включно з перевіркою відсутності кирилиці в англомовних константах шаблону).

### Фаза 2 — RAG-шар (G4, G5)
- [x] `retrieval.py`: додати точний Tanimoto (побітове `popcount(a & b) / popcount(a | b)` над numpy-масивами, `chemistry.tanimoto_similarity`) — рахується на top-N кандидатах, повернутих Qdrant-ANN через `with_vectors=True` (Cosine лишається лише для швидкого відбору кандидатів, не як фінальна цифра).
- [x] `merge_retrieval_hits`: дефолтна формула = `weights.molecule * tanimoto_product + weights.reaction * tanimoto_transform` (2 доданки, як на рис. 3.2). Параметр `reaction_class` лишено, default = `0.0`, задокументовано як "розширення понад ПЗ, вимкнене за замовчуванням".
- [x] `config.py`: `DEFAULT_RETRIEVAL_WEIGHTS` → двокомпонентний за замовчуванням (`molecule=0.5, reaction=0.5, reaction_class=0.0`); окремий `EXPERIMENTAL_RETRIEVAL_WEIGHTS` (`0.5/0.3/0.2`) для тих, хто хоче ввімкнути розширення.
- [x] `tests/test_retrieval_scoring.py`: формула перевірена на синтетичних векторах (tanimoto edge cases, дефолтна vs experimental вага, сортування, merge за reaction_id).

### Фаза 3 — Провайдери моделей (G8, G9)
- [x] Розбити `llm_providers.py` → `providers/__init__.py` (Protocol, `LLMProviderConfig`, реєстр) + `providers/chat_api.py` (перенесені `GroqLLMProvider`, `OpenAILLMProvider`, `OpenAICompatibleLLMProvider` без змін поведінки; прибрано мертвий `ROUTES_JSON_SCHEMA`/`REACTION_STEP_SCHEMA`, оскільки `json_mode` тепер завжди `False`).
- [x] `providers/local_seq2seq.py`: обгортка ReactionT5v2 на основі `fine-tune/v2/res/08_reactiont5_conditions_agent_demo.ipynb` — `generate()` повертає прекурсори (без `<think>`; `reasoning.py` приймає legacy no-think відповіді).
- [x] `providers/local_causal.py`: двоетапний Qwen2.5-7B LoRA з `06_agent_inference_demo.ipynb` (reactants/class-адаптер + умови-адаптер), результат форматується як `<think>/<answer>` для спільного парсера.
- [x] `providers/local_gguf.py`: ChemLLM GGUF з `base_models_res/ChemLLM-20B-Chat.ipynb` через `llama-cpp-python`, messages/CoT-промпт передаються без змін (як у чат-провайдерів).
- [x] Реєстрація нових провайдерів у `LLM_PROVIDER_REGISTRY` за прапорцем `api_key_required=False`, позначка "(local)" в UI-лейблі.
- [x] `pyproject.toml`: extras `[local-models]` (`torch`, `transformers`, `peft`, `accelerate`, `bitsandbytes`, `sentencepiece`, `llama-cpp-python`).
- [x] `tests/test_providers.py`: реєстр провайдерів, `extract_target_smiles`, JSON-парсинг LoRA-виводу — усе без важких ML-залежностей.

### Фаза 4 — UI (G7)
- [x] `streamlit_views.py`: `display_step_result(step_result)` з окремим `st.expander("🧠 Chemist's Reasoning", expanded=True)` для `think`-тексту, окремо — картка реакції (реагенти/продукт/зображення).
- [x] Кнопка "Generate another candidate" — повторний виклик `generate_single_step` (замінює `route_count`-слайдер); RAG-контекст (`reactions`) переретрієвиться лише один раз і повторно використовується для кожного наступного кандидата, кандидати накопичуються в `step_results` і рендеряться під заголовками "Candidate N", коли їх більше одного.
- [x] Провайдер-селектор у сайдбарі: групування "Cloud API" / "Local / research" (`LLMProviderConfig.category`, `st.radio` + відфільтрований `st.selectbox`) з попередженням про вагу залежностей (`torch`/`transformers`/`peft`/`llama-cpp-python`, `pip install -e ".[local-models]"`) для локальних.
- [x] Ручна регресійна перевірка (headless Chrome через CDP, оскільки в оточенні немає GROQ_API_KEY): групування провайдерів і попередження відображаються коректно; відсутній API-ключ показує помилку `_missing_credentials_message`; RAG-fallback при недоступному Qdrant показує попередження один раз; локальний провайдер без встановлених `[local-models]` падає безпечно (`LLM API failure: No module named 'torch'`) замість краху застосунку; "Generate another candidate" додає "Candidate 2" без повторного RAG-запиту.

### Фаза 5 — Автоматизоване оцінювання (G10)
- [x] `src/retro_planner/evaluation.py`: `top_k_exact_match(predictions, references, k)` (ranked, order-independent, exact-match SMILES-set порівняння через `canonical_precursor_set`/`is_exact_match`), `structure_success_rate(smiles_list)`, плюс `format_results_table(...)` для Табл. 4.1-стилю виводу.
- [x] `scripts/evaluate_retrosynthesis.py`: CLI, що бере test-спліт USPTO-50K (через перевикористаний `normalize_row` з `index_uspto50k_to_qdrant.py`), прогонить Zero-shot і/або RAG+CoT конфігурації через будь-якого зареєстрованого провайдера (`--provider`, підтримує cloud-API й local-моделі), генерує `k`=max(--k) кандидатів на ціль повторними викликами `generate_single_step` (RAG-контекст ретрієвиться раз на ціль, як і в UI), друкує markdown-таблицю Top-k + Structure Success Rate у форматі Табл. 4.1 ПЗ. Офлайн перевірено fake-провайдером для обох режимів (zero_shot і rag_cot) — метрики, промпт-контекст і RAG-гілка логічно коректні.
- [ ] **Ще не виконано** — прогнати на реальному провайдері (мінімум Groq/custom_openai) і зафіксувати отримані числа для Розділу 4.2 фінального ПЗ. Досі неможливо без `GROQ_API_KEY`/`OPENAI_API_KEY`, пакета `datasets` (`pip install -e ".[indexing]"`) і запущеного Qdrant з проіндексованими даними (`docker compose up -d qdrant && python scripts/index_uspto50k_to_qdrant.py --recreate`) — жодного з цих трьох немає в поточному середовищі виконання агента. Потребує ручного запуску людиною з доступом до credentials/інфраструктури, наприклад `GROQ_API_KEY=... python scripts/evaluate_retrosynthesis.py --limit 25`.

### Фаза 6 — Документація
- [x] Перемальовано `docs/system_diagrams.md` (вступ + розділи 1, 3, 4, 7, 8) під think/answer-конвеєр без `routes`/JSON: реєстр провайдерів, `build_cot_prompt`/`parse_reasoning_response`/`validate_precursors`, "Generate another candidate", двокомпонентна Tanimoto-формула.
- [x] Оновлено `AGENTS.md` (Project Overview, Repository Layout, Setup, Development Guidance, Verification, Style) під нову структуру `providers/`/`prompting.py`/`reasoning.py`/`evaluation.py`/`tests/`.
- [x] Додано в `README.MD` опис think/answer-контракту, provider-категорій (Cloud API / Local / research), оновлено "Interpreting Results" і "Hybrid Retrieval Scores" під фактичний `StepResult`/двокомпонентну формулу, додано `pytest`- і `scripts/evaluate_retrosynthesis.py`-команди.
- [x] Оновлено `eval.md` приміткою на початку файлу, що ручні прогони доповнюються автоматичним `scripts/evaluate_retrosynthesis.py`.

---

## 6. Definition of Done (звірка з ПЗ)

- [x] Будь-яка відповідь LLM у продакшн-конвеєрі парситься через `<think>|<reason>` + `<answer>` — жодного `json_mode`/`ROUTES_JSON_SCHEMA` у новому коді (`json_mode` завжди `False` у `planning._call_provider`; старі схеми видалені разом з `llm_providers.py`/`prompts.py`).
- [x] Один виклик планувальника = один ретросинтетичний крок; "кілька варіантів" — це кілька викликів `generate_single_step`/"Generate another candidate", не один JSON-масив.
- [x] Промпт до LLM буквально складається з 4 блоків `[System]/[Context]/[Instruction]/[Input]`, як на стор. 32 ПЗ (`prompting.build_cot_prompt`).
- [x] Увесь текст, що надсилається до LLM (system/context/instruction/input, repair-промпт), — англійською; жодних кириличних рядків у `prompting.py` (перевірено `test_no_prompt_template_strings_contain_cyrillic_characters`).
- [x] Фінальна гібридна оцінка RAG — реальний коефіцієнт Танімото (`chemistry.tanimoto_similarity`, побітовий popcount), формула = product score + transform score за замовчуванням (`DEFAULT_RETRIEVAL_WEIGHTS`, `reaction_class` вимкнено).
- [x] UI має окремий блок "Chemist's Reasoning" (`streamlit_views.display_step_result`, `st.expander("🧠 Chemist's Reasoning")`).
- [x] Прекурсори з `<answer>` проходять хімічну валідацію (RDKit-парсинг + груба mass-balance перевірка проти цілі) до показу користувачу (`reasoning.validate_precursors`).
- [x] Додано три провайдери поза чат-API (`local_seq2seq`, `local_causal`, `local_gguf`), які працюють через той самий `LLMProvider`-контракт — закриває вимогу користувача про "гнучку інтеграцію з іншими моделями".
- [ ] `scripts/evaluate_retrosynthesis.py` видає реальні Top-1/3/5 і Structure Success Rate цифри, якими можна закрити плейсхолдери `[ВСТАВИТИ %]` у Розділі 4.2 ПЗ — **скрипт готовий і офлайн-перевірений, але ще не прогнаний на реальному провайдері/датасеті** (див. Фазу 5, останній пункт); потребує ручного запуску з `GROQ_API_KEY`/`OPENAI_API_KEY` і Qdrant.
- [x] `docs/system_diagrams.md` і `AGENTS.md` більше не описують JSON/routes-архітектуру.

## 7. Ризики та сумісність

- **Ламаюча зміна контракту.** Будь-хто, хто зараз парсить JSON-вивід (`routes`/`steps`), зламається. Оскільки це навчальний прототип без зовнішніх споживачів API, це прийнятно — але варто зробити зміну одним чітким PR/комітом, а не поступово.
- **Локальні провайдери важкі.** `transformers`/`torch`/`llama-cpp-python` не повинні потрапляти в базовий `Dockerfile` хмарного UI-деплою — тримати їх за extras і імпортувати лінькво (`from transformers import ...` всередині методу `generate()`, за зразком уже наявного патерну лінивого імпорту `groq`/`openai` в `llm_providers.py`).
- **Точний Tanimoto дорожчий за Qdrant ANN-Cosine.** Рахувати його тільки на top-N (N = query_limit, вже обчислюється як `top_k * 3`), а не на всій колекції — продуктивність не постраждає.
- **Старі seq2seq-моделі (ReactionT5) не вміють `<think>`.** `reasoning.py` повинен явно підтримувати "no-think" відповіді (весь текст = precursors) як легітимний режим, а не помилку — це вже частково є в `_reactant_plan_from_text` (`planning.py`), логіку варто перенести, а не викинути.
