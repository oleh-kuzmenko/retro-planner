# Схеми інтелектуальної системи ретросинтезу органічних сполук

Користувач задає цільову молекулу через вебінтерфейс. Система валідовує SMILES,
отримує релевантні приклади реакцій з векторної бази, формує 4-блоковий CoT-промпт,
викликає один із зареєстрованих LLM-провайдерів, парсить відповідь за контрактом
`<think>/<reason>` + `<answer>` і відображає один ретросинтетичний крок (з опцією
згенерувати ще один кандидатний крок для того самого продукту).

У реалізації основними модулями є:

- `src/retro_planner/app.py` - Streamlit UI та оркестрація сценарію.
- `src/retro_planner/chemistry.py` - валідація, канонізація SMILES і fingerprint-и RDKit (включно з побітовим Tanimoto).
- `src/retro_planner/retrieval.py` - hybrid RAG пошук у Qdrant (Tanimoto product + transform score).
- `src/retro_planner/planning.py` - `generate_single_step()`: оркестрація виклику провайдера, парсингу й repair для одного кроку.
- `src/retro_planner/prompting.py` - 4-блоковий `[System]/[Context]/[Instruction]/[Input]` CoT-промпт і repair-промпт (англійською).
- `src/retro_planner/reasoning.py` - парсинг `<think>/<reason>` + `<answer>` тегів і хімічна валідація прекурсорів (RDKit, mass-balance).
- `src/retro_planner/providers/` - реєстр `LLMProvider`: `chat_api.py` (Groq/OpenAI/custom OpenAI-compatible) плюс `local_seq2seq.py`, `local_causal.py`, `local_gguf.py` для дослідницьких моделей поза чат-API.
- `src/retro_planner/rendering.py` і `streamlit_views.py` - візуалізація молекул, реакцій і "Chemist's Reasoning" блоку.
- `scripts/index_uspto50k_to_qdrant.py` - побудова векторної бази з USPTO-50K та ORD.
- `scripts/evaluate_retrosynthesis.py` та `src/retro_planner/evaluation.py` - автоматизоване Top-k/Structure Success Rate оцінювання на USPTO-50K.

## 1. Контекстна схема системи

```mermaid
flowchart LR
    User["Користувач / дослідник"] -->|"Малює структуру або вводить SMILES"| Web["Вебінтерфейс Streamlit<br/>AI Retro-Synthesis Planner"]

    Web -->|"SMILES"| System["Інтелектуальна система<br/>ретросинтезу"]

    System --> Chem["RDKit<br/>валідація, канонізація,<br/>fingerprint-и"]
    System --> Qdrant["Qdrant<br/>векторна база реакцій"]
    System --> LLM["LLM-провайдер<br/>Groq / OpenAI-compatible / local model"]

    Qdrant -->|"Схожі реакції,<br/>умови, yield, source"| System
    LLM -->|"&lt;think&gt;міркування&lt;/think&gt;<br/>&lt;answer&gt;SMILES.SMILES&lt;/answer&gt;"| System
    Chem -->|"Валідні SMILES<br/>та зображення реакцій"| System

    System -->|"Один ретросинтетичний крок:<br/>reasoning, прекурсори,<br/>продукт, попередження"| Web
    Web -->|"Візуальний результат"| User

    Datasets["USPTO-50K / ORD"] --> Indexer["Індексатор реакцій"]
    Indexer -->|"Morgan product vectors<br/>reaction transform vectors<br/>payload metadata"| Qdrant
```

## 2. Цільова схема системи для хімічного підприємства

```mermaid
flowchart LR
    Chemist["Хімік-синтетик<br/>отримує замовлення на молекулу"] --> Web["Вебінтерфейс агента<br/>цільова структура, обмеження,<br/>кількість варіантів"]

    Web --> Core["AI-ядро ретросинтезу<br/>поточний прототип з цього репозиторію"]

    Core --> Inventory["Модуль складу реагентів<br/>наявність, кількість,<br/>вартість, небезпечність"]
    Core --> InternalDB["Внутрішня база реакцій<br/>підприємства<br/>ELN/LIMS/журнали синтезів"]
    Core --> Literature["Літературний пошук<br/>Reaxys / статті / патенти"]
    Core --> VectorDB["Векторна база реакцій<br/>USPTO, ORD, внутрішні реакції"]
    Core --> ModelBlock["AI-блок ретросинтезу<br/>LLM, RAG, fine-tuned models"]
    Core --> Validation["Хімічна валідація<br/>RDKit, target matching,<br/>фільтри коректності"]

    Inventory -->|"Доступні реагенти<br/>і практичні обмеження"| Core
    InternalDB -->|"Реакції, які вже робили<br/>колеги на підприємстві"| Core
    Literature -->|"Аналогічні реакції<br/>з літератури"| Core
    VectorDB -->|"Схожі продукти,<br/>трансформації, умови"| Core
    ModelBlock -->|"Кандидатні маршрути"| Core
    Validation -->|"Валідні/відхилені варіанти"| Core

    Core --> Ranking["Ранжування маршрутів<br/>доступність реагентів,<br/>схожість з внутрішніми реакціями,<br/>очікувана практичність"]
    Ranking --> Web
    Web --> Chemist

    Chemist -->|"Проба 1, 2, 3...<br/>зміна реагентів/умов"| ExperimentLog["Журнал експериментів"]
    ExperimentLog --> InternalDB
    ExperimentLog --> VectorDB
```

## 3. Компонентна схема застосунку

```mermaid
flowchart TB
    subgraph Client["Клієнтський рівень"]
        Browser["Браузер користувача"]
        Ketcher["Ketcher editor<br/>малювання молекули"]
        SmilesInput["SMILES input"]
    end

    subgraph App["Application layer: Streamlit app.py"]
        Sidebar["Налаштування<br/>provider category, provider, model,<br/>API key/base URL, RAG, Top-K"]
        TargetPanel["Панель цільової молекули"]
        Orchestrator["generate_plan() /<br/>generate_another_candidate()"]
        ResultView["display_step_result()<br/>Chemist's Reasoning +<br/>precursors + reaction image"]
    end

    subgraph Chemistry["Chemistry services: chemistry.py / rendering.py"]
        Canonicalizer["canonicalize_smiles()<br/>RDKit parsing + canonical SMILES"]
        Fingerprints["Morgan fingerprints<br/>2048-bit vectors"]
        ReactionFP["Reaction transform fingerprint<br/>product_fp XOR reactants_fp"]
        Tanimoto["tanimoto_similarity()<br/>bitwise popcount(a&b)/popcount(a|b)"]
        Renderer["RDKit rendering<br/>molecule/reaction images"]
    end

    subgraph Retrieval["RAG retrieval: retrieval.py"]
        QClient["create_qdrant_client()"]
        ProductSearch["Search reactions_morgan<br/>ANN shortlist (Cosine)"]
        TransformSearch["Search reaction_transforms<br/>ANN shortlist (Cosine)"]
        Reranker["merge_retrieval_hits()<br/>0.5*Tanimoto_product +<br/>0.5*Tanimoto_transform (default)"]
    end

    subgraph Planning["Planning layer: planning.py / prompting.py / reasoning.py"]
        PromptBuilder["build_cot_prompt() /<br/>build_cot_repair_prompt()<br/>[System]/[Context]/[Instruction]/[Input]"]
        LLMProviderNode["LLMProvider registry<br/>providers/chat_api.py +<br/>local_seq2seq/local_causal/local_gguf"]
        Parser["parse_reasoning_response()<br/>&lt;think|reason&gt; + &lt;answer&gt; tags"]
        Validator["validate_precursors()<br/>RDKit parse + mass-balance vs target"]
        Repair["build_cot_repair_prompt()<br/>same tag contract, temperature=0"]
    end

    subgraph Storage["Data layer"]
        QdrantProduct[("Qdrant collection<br/>reactions_morgan")]
        QdrantTransform[("Qdrant collection<br/>reaction_transforms")]
    end

    Browser --> Ketcher
    Browser --> SmilesInput
    Ketcher --> TargetPanel
    SmilesInput --> TargetPanel
    Sidebar --> Orchestrator
    TargetPanel --> Canonicalizer
    Canonicalizer --> Orchestrator

    Orchestrator --> QClient
    QClient --> ProductSearch
    QClient --> TransformSearch
    ProductSearch --> QdrantProduct
    TransformSearch --> QdrantTransform
    ProductSearch --> Reranker
    TransformSearch --> Reranker
    Fingerprints --> ProductSearch
    ReactionFP --> TransformSearch
    Tanimoto --> Reranker
    Reranker --> PromptBuilder

    Orchestrator --> PromptBuilder
    PromptBuilder --> LLMProviderNode
    LLMProviderNode --> Parser
    Parser --> Validator
    Validator -->|"missing/invalid answer"| Repair
    Repair --> LLMProviderNode
    Validator -->|"valid precursors"| ResultView
    Renderer --> ResultView
```

## 4. Sequence-діаграма генерації одного ретросинтетичного кроку

```mermaid
sequenceDiagram
    actor User as Користувач
    participant UI as Streamlit UI (app.py)
    participant Chem as RDKit chemistry.py
    participant Retrieval as retrieval.py
    participant Qdrant as Qdrant vector DB
    participant Planner as planning.py<br/>generate_single_step()
    participant Prompt as prompting.py
    participant LLM as LLMProvider<br/>(chat_api / local_*)
    participant Reason as reasoning.py
    participant Render as RDKit rendering

    User->>UI: Вводить або малює цільову молекулу
    UI->>Chem: canonicalize_smiles(smiles)

    alt SMILES невалідний
        Chem-->>UI: None
        UI-->>User: Помилка "Invalid SMILES"
    else SMILES валідний
        Chem-->>UI: canonical_target_smiles
        UI->>Render: generate_molecule_image(target)
        Render-->>UI: Зображення цільової молекули
        User->>UI: Натискає Generate retrosynthesis

        alt RAG enabled
            UI->>Retrieval: retrieve_reactions_for_smiles(target, top_k)
            Retrieval->>Chem: generate_morgan_fingerprint(target)
            Chem-->>Retrieval: product vector
            Retrieval->>Chem: generate_reaction_fingerprint(target)
            Chem-->>Retrieval: transform vector
            Retrieval->>Qdrant: search reactions_morgan (ANN + vectors)
            Qdrant-->>Retrieval: product-similar hits
            Retrieval->>Qdrant: search reaction_transforms (ANN + vectors)
            Qdrant-->>Retrieval: transform-similar hits
            Retrieval->>Chem: tanimoto_similarity() rescoring on shortlist
            Retrieval->>Retrieval: merge_retrieval_hits() by hybrid score
            Retrieval-->>UI: top-K reactions (RAG_Examples) + warnings
        else RAG disabled
            UI->>UI: reactions = []
        end

        UI->>Planner: generate_single_step(GenerationRequest)
        Planner->>Prompt: build_cot_prompt(target, reactions)
        Prompt-->>Planner: [System]/[Context]/[Instruction]/[Input] prompt
        Planner->>LLM: generate(messages, json_mode=False)
        LLM-->>Planner: raw text with &lt;think&gt;/&lt;answer&gt;
        Planner->>Reason: parse_reasoning_response(raw)
        Reason-->>Planner: ReasoningResult(think, answer_smiles)
        Planner->>Reason: validate_precursors(answer_smiles, target)
        Reason-->>Planner: precursors | None, warnings, errors

        alt Прекурсори валідні
            Planner-->>UI: StepResult(think, precursors, product, warnings)
        else Answer відсутній/невалідний
            Planner->>Prompt: build_cot_repair_prompt(target, reactions, raw, errors)
            Prompt-->>Planner: repair prompt (same tag contract)
            Planner->>LLM: generate(repair prompt, temperature=0)
            LLM-->>Planner: corrected &lt;think&gt;/&lt;answer&gt; text
            Planner->>Reason: parse + validate_precursors again
            Planner-->>UI: StepResult(valid precursors або errors)
        end

        UI->>Render: generate_reaction_image(precursors, product)
        Render-->>UI: Reaction scheme image або warning
        UI-->>User: "Chemist's Reasoning" (think) + precursors + image + warnings

        opt User натискає "Generate another candidate"
            UI->>Planner: generate_single_step(GenerationRequest) again<br/>(same cached RAG reactions, no re-query)
            Planner-->>UI: додатковий StepResult -> "Candidate N"
        end
    end
```

## 5. Sequence-діаграма цільового виробничого сценарію

Файли: [PNG](diagrams/production_sequence.png), [PDF](diagrams/production_sequence.pdf), [Mermaid](diagrams/production_sequence.mmd)

```mermaid
sequenceDiagram
    actor Chemist as Хімік-синтетик
    participant UI as Вебагент
    participant Inventory as Склад реагентів
    participant Internal as Внутрішня база реакцій
    participant Literature as Літературний пошук
    participant RAG as Векторна база/RAG
    participant AI as AI-блок ретросинтезу
    participant Validation as Валідація і фільтри
    participant Ranker as Ранжування
    participant LabLog as Журнал експериментів

    Chemist->>UI: Задає цільову молекулу і обмеження
    UI->>Inventory: Перевірити доступні реагенти та запаси
    Inventory-->>UI: Список доступних/відсутніх реагентів

    UI->>Internal: Знайти подібні внутрішні синтези
    Internal-->>UI: Реакції колег, умови, виходи, невдалі спроби

    UI->>Literature: Знайти подібні літературні реакції
    Literature-->>UI: Аналоги з Reaxys/статей/патентів

    UI->>RAG: Пошук схожих реакцій і трансформацій
    RAG-->>UI: Retrieved evidence для prompt

    UI->>AI: Згенерувати маршрути з урахуванням складу та evidence
    AI-->>UI: Кандидатні маршрути ретросинтезу

    UI->>Validation: Перевірити SMILES, product match, формат, практичність
    Validation-->>UI: Валідні маршрути і попередження

    UI->>Ranker: Оцінити маршрути
    Ranker-->>UI: Рейтинг за доступністю, evidence, умовами, ризиками

    UI-->>Chemist: Пропонує кілька схем синтезу
    Chemist->>LabLog: Фіксує результат експерименту
    LabLog->>Internal: Оновити історію підприємства
    LabLog->>RAG: Додати новий приклад до індексу
```

## 6. Схема формування векторної бази реакцій

```mermaid
flowchart LR
    USPTO["USPTO-50K<br/>reaction_smiles, class,<br/>reactants/product"] --> Loader["Data loading"]
    ORD["Open Reaction Database<br/>protobuf reactions,<br/>conditions, yields"] --> Loader

    Loader --> Normalizer["Normalization<br/>canonical product/reactants SMILES<br/>shared payload schema"]
    Normalizer --> Filter["RDKit validation<br/>skip invalid molecules"]

    Filter --> ProductFP["Product fingerprint<br/>2048-bit Morgan"]
    Filter --> TransformFP["Reaction transform fingerprint<br/>product_fp XOR combined_reactants_fp"]

    ProductFP --> ProductCollection[("Qdrant<br/>reactions_morgan")]
    TransformFP --> TransformCollection[("Qdrant<br/>reaction_transforms")]

    Filter --> Payload["Payload metadata<br/>reaction_id, reaction_smiles,<br/>reaction_class, solvent,<br/>temperature, yield, source"]
    Payload --> ProductCollection
    Payload --> TransformCollection
```

## 7. Схема валідації та постобробки результату LLM

```mermaid
flowchart TD
    Raw["Raw LLM response<br/>(plain tagged text, не JSON)"] --> FindThink{"Є &lt;think&gt; або<br/>&lt;reason&gt; тег?"}

    FindThink -->|"так"| Think["think = вміст тегу"]
    FindThink -->|"ні"| NoThink["think = None<br/>(legacy no-think провайдер, напр. ReactionT5)"]

    Think --> FindAnswer{"Є &lt;answer&gt; тег?"}
    NoThink --> FindAnswer

    FindAnswer -->|"так"| SplitAnswer["Розбити вміст &lt;answer&gt;<br/>по крапці на SMILES-фрагменти"]
    FindAnswer -->|"ні"| Fallback["Warning: тег відсутній.<br/>Трактувати всю відповідь<br/>як &lt;answer&gt; (fallback-режим)"]
    Fallback --> SplitAnswer

    SplitAnswer --> EmptyCheck{"Є хоча б один<br/>SMILES-фрагмент?"}
    EmptyCheck -->|"ні"| Error1["Error: немає прекурсорів<br/>у відповіді"]
    EmptyCheck -->|"так"| RDKitParse["RDKit-парсинг кожного<br/>фрагмента (валідність, валентність)"]

    RDKitParse --> ParseOk{"Усі фрагменти<br/>валідні?"}
    ParseOk -->|"ні"| Error2["Error: невалідний SMILES<br/>у прекурсорі"]
    ParseOk -->|"так"| MassBalance["Груба mass-balance перевірка:<br/>сума важких атомів прекурсорів<br/>vs target (з допуском на leaving groups)"]

    MassBalance --> BalanceOk{"Баланс мас<br/>прийнятний?"}
    BalanceOk -->|"ні"| Warn1["Warning: можлива<br/>втрата атомів"]
    BalanceOk -->|"так"| Canonical["Канонічні прекурсори"]
    Warn1 --> Canonical

    Error1 --> RepairDecision
    Error2 --> RepairDecision
    RepairDecision["generate_single_step():<br/>errors непорожні?"] -->|"так"| RepairLLM["build_cot_repair_prompt()<br/>той самий tag-контракт,<br/>temperature = 0"]
    RepairLLM --> Raw

    Canonical --> Render{"RDKit може<br/>намалювати реакцію?"}
    Render -->|"так"| Final["StepResult: think, precursors,<br/>product, image, warnings"]
    Render -->|"ні"| Warning["StepResult + warning<br/>про некоректне зображення"]
```

## 8. Порівняння моделей

```mermaid
flowchart TB
    Benchmark["Тестовий набір молекул<br/>eval.md"] --> Modes["Режими порівняння"]

    Modes --> A["Groq Llama<br/>без RAG"]
    Modes --> B["Groq Llama<br/>гібридний RAG"]
    Modes --> C["ReactionT5v2<br/>retrosynthesis"]
    Modes --> D["ChemLLM-20B-Chat-SFT<br/>GGUF локальний запуск"]
    Modes --> F["Two-stage Qwen2.5-7B LoRA<br/>reactants/class + conditions"]
    Modes --> G["ReactionT5v2 + Qwen-умови<br/>reactants + conditions LoRA"]

    A --> Rubric["Оцінювання 0-10"]
    B --> Rubric
    C --> Rubric
    D --> Rubric
    F --> Rubric
    G --> Rubric

    Rubric --> Criteria["Критерії:<br/>збіг прекурсорів (0-8),<br/>якість метаданих (0-2)"]
    Criteria --> Conclusion["Порівняння результатів<br/>між режимами"]
```

Автоматизоване доповнення до цього ручного порівняння: `scripts/evaluate_retrosynthesis.py`
рахує Top-1/3/5 exact match і Structure Success Rate на USPTO-50K для Zero-shot і RAG+CoT
режимів через будь-якого зареєстрованого провайдера (див. `LLM_PROVIDER_REGISTRY`).
