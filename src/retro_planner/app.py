import logging
import os
from dataclasses import dataclass

import streamlit as st
from streamlit_ketcher import st_ketcher

from retro_planner.chemistry import canonicalize_smiles
from retro_planner.config import DEFAULT_TARGET_SMILES
from retro_planner.providers import (
    CATEGORY_CLOUD_API,
    CATEGORY_LABELS,
    CATEGORY_LOCAL_RESEARCH,
    LLM_PROVIDER_REGISTRY,
    LLMProviderConfig,
)
from retro_planner.planning import GenerationRequest, StepResult, generate_single_step
from retro_planner.rendering import generate_molecule_image
from retro_planner.retrieval import (
    RetrievalConfig,
    create_qdrant_client,
    retrieve_reactions_for_smiles,
)
from retro_planner.streamlit_views import display_hybrid_retrieval, display_step_result


def configure_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    logging.getLogger("retro_planner").setLevel(logging.INFO)


@dataclass(frozen=True)
class SidebarSettings:
    provider_key: str
    provider_config: LLMProviderConfig
    api_key: str
    base_url: str | None
    rag_enabled: bool
    top_k: int
    model: str

    @property
    def provider_label(self) -> str:
        return self.provider_config.label


@st.cache_resource(show_spinner=False)
def get_cached_qdrant_client():
    return create_qdrant_client(RetrievalConfig())


def configure_page() -> None:
    st.set_page_config(
        page_title="AI Retrosynthesis Planner",
        layout="wide",
        page_icon="🧪",
    )
    st.markdown("""
        <style>
        .stButton>button {
            height: 3em;
            background-color: #FF4B4B;
            color: white;
        }
        .reportview-container .main .block-container{
            padding-top: 2rem;
        }
        </style>
    """, unsafe_allow_html=True)
    st.title("🧪 AI Retro-Synthesis Planner")
    st.markdown("""
**Instruction:** Draw or paste a target molecule below. The AI will propose alternative ways to make it with practical reaction conditions.
""")


def render_sidebar() -> SidebarSettings:
    with st.sidebar:
        st.header("⚙️ Configuration")
        category = st.radio(
            "Provider category",
            options=[CATEGORY_CLOUD_API, CATEGORY_LOCAL_RESEARCH],
            format_func=lambda key: CATEGORY_LABELS[key],
            horizontal=True,
        )
        category_provider_keys = [
            key
            for key, config in LLM_PROVIDER_REGISTRY.items()
            if config.category == category
        ]
        if category == CATEGORY_LOCAL_RESEARCH:
            st.caption(
                "⚠️ Local / research providers run heavy ML dependencies "
                "(torch, transformers, peft, or llama-cpp-python) in this process. "
                'Install them first with `pip install -e ".[local-models]"`; the '
                "first call also loads model weights, which can take a while."
            )
        provider_key = st.selectbox(
            "LLM provider",
            options=category_provider_keys,
            format_func=lambda key: LLM_PROVIDER_REGISTRY[key].label,
        )
        provider_config = LLM_PROVIDER_REGISTRY[provider_key]
        provider_label = provider_config.label
        api_key_help = (
            "Required for AI generation"
            if provider_config.api_key_required
            else "Optional for local endpoints such as Ollama"
        )
        api_key = st.text_input(
            (
                f"{provider_label} API Key"
                if provider_config.api_key_required
                else f"{provider_label} API Key (optional)"
            ),
            value=os.getenv(provider_config.api_key_env_var, ""),
            type="password",
            help=api_key_help,
        )
        api_key = api_key.strip()
        if provider_config.key_url:
            st.markdown(f"[Get {provider_label} API Key]({provider_config.key_url})")

        base_url = None
        if provider_config.base_url_env_var:
            base_url = st.text_input(
                f"{provider_label} base URL",
                value=os.getenv(
                    provider_config.base_url_env_var,
                    provider_config.default_base_url or "",
                ),
                help="OpenAI-compatible base URL, for example an HF Space root URL or Ollama /v1.",
            )
            base_url = base_url.strip()

        st.divider()
        rag_enabled = st.checkbox("RAG enabled", value=True)
        top_k = st.slider("Top-K", min_value=1, max_value=20, value=5)
        model = st.text_input(
            f"{provider_label} model",
            value=os.getenv(provider_config.model_env_var, provider_config.default_model),
        )

    return SidebarSettings(
        provider_key=provider_key,
        provider_config=provider_config,
        api_key=api_key,
        base_url=base_url,
        rag_enabled=rag_enabled,
        top_k=top_k,
        model=model,
    )


def select_target_smiles() -> tuple[str | None, str | None]:
    drawn_smiles = st_ketcher(DEFAULT_TARGET_SMILES, height=450)
    input_source = st.radio(
        "Input source",
        ["Ketcher", "SMILES"],
        horizontal=True,
    )
    manual_smiles = st.text_input(
        "SMILES input",
        value=drawn_smiles or DEFAULT_TARGET_SMILES,
    )
    smiles_input = drawn_smiles if input_source == "Ketcher" else manual_smiles
    return smiles_input, canonicalize_smiles(smiles_input)


def render_target_panel(canonical_input: str | None) -> bool:
    st.markdown("### Selected Target")
    if canonical_input:
        target_image = generate_molecule_image(canonical_input)
        if target_image:
            st.image(target_image, caption="Target structure", width="stretch")
        st.code(canonical_input, language="text")
        st.success("Structure Validated ✅")
    else:
        st.error("Invalid SMILES")

    return st.button("Generate retrosynthesis", type="primary", width="stretch")


def _show_messages(messages: list[str], level: str) -> None:
    for message in messages:
        if level == "warning":
            st.warning(message)
        elif level == "error":
            st.error(message)


def _missing_credentials_message(settings: SidebarSettings) -> str | None:
    if settings.provider_config.api_key_required and not settings.api_key:
        return (
            f"Missing {settings.provider_config.api_key_env_var}. Add it in the sidebar "
            f"or set the {settings.provider_config.api_key_env_var} environment variable."
        )
    if settings.provider_config.base_url_env_var and not settings.base_url:
        return (
            f"Missing {settings.provider_config.base_url_env_var}. Add it in the sidebar "
            f"or set the {settings.provider_config.base_url_env_var} environment variable."
        )
    return None


def _generate_step(
    canonical_input: str,
    settings: SidebarSettings,
    reactions: list[dict],
) -> StepResult:
    provider = settings.provider_config.create_provider(
        settings.api_key,
        settings.base_url,
    )
    request = GenerationRequest(
        target_smiles=canonical_input,
        llm_provider=provider,
        model=settings.model,
        reactions=reactions,
    )
    with st.spinner(f"Generating a retrosynthesis step with {settings.provider_label}..."):
        return generate_single_step(request)


def generate_plan(
    canonical_input: str,
    settings: SidebarSettings,
) -> dict:
    """Run RAG retrieval (once) plus a single retrosynthetic-step generation.

    Additional candidates for the same target are produced by calling
    `_generate_step` again (see `generate_another_candidate`), reusing the
    same retrieved reactions rather than re-querying Qdrant.
    """
    reactions: list[dict] = []
    rag_warnings: list[str] = []

    if settings.rag_enabled:
        with st.spinner("Retrieving similar reactions from Qdrant for model context..."):
            try:
                retrieval_result = retrieve_reactions_for_smiles(
                    canonical_input,
                    settings.top_k,
                    client=get_cached_qdrant_client(),
                )
                reactions = retrieval_result.reactions
                rag_warnings.extend(retrieval_result.warnings)
            except Exception as exc:
                rag_warnings.append(
                    f"Qdrant unavailable; generating without retrieved context. Details: {exc}"
                )

    step_result = _generate_step(canonical_input, settings, reactions)
    return {
        "canonical_input": canonical_input,
        "provider_key": settings.provider_key,
        "model": settings.model,
        "base_url": settings.base_url,
        "rag_enabled": settings.rag_enabled,
        "reactions": reactions,
        "step_results": [step_result],
        "rag_warnings": rag_warnings,
    }


def generate_another_candidate(
    canonical_input: str,
    settings: SidebarSettings,
    run: dict,
) -> None:
    """Append one more single-step candidate to an existing run in place.

    This is the "several routes" affordance from the PZ target architecture:
    not one LLM call returning several routes, but several one-step calls
    through the same CoT pipeline, reusing the RAG context already retrieved
    for this target.
    """
    step_result = _generate_step(canonical_input, settings, run.get("reactions", []))
    run["step_results"].append(step_result)


def render_latest_run(canonical_input: str | None, settings: SidebarSettings) -> None:
    latest_run = st.session_state.get("latest_retrosynthesis_run")
    if not (
        latest_run
        and latest_run.get("canonical_input") == canonical_input
        and latest_run.get("provider_key") == settings.provider_key
        and latest_run.get("model") == settings.model
        and latest_run.get("base_url") == settings.base_url
    ):
        return

    st.markdown("### Canonical SMILES")
    st.code(canonical_input, language="text")

    _show_messages(latest_run.get("rag_warnings", []), "warning")

    if latest_run.get("rag_enabled"):
        display_hybrid_retrieval(latest_run.get("reactions", []))

    step_results: list[StepResult] = latest_run.get("step_results", [])
    show_candidate_labels = len(step_results) > 1
    for index, step_result in enumerate(step_results, start=1):
        display_step_result(
            step_result,
            canonical_input,
            candidate_label=f"Candidate {index}" if show_candidate_labels else None,
        )
        st.divider()

    if step_results and st.button(
        "🔁 Generate another candidate", width="stretch", key="generate_another_candidate"
    ):
        credentials_error = _missing_credentials_message(settings)
        if credentials_error:
            st.error(credentials_error)
        else:
            generate_another_candidate(canonical_input, settings, latest_run)
            st.rerun()


def main() -> None:
    configure_logging()
    configure_page()
    settings = render_sidebar()
    _, canonical_input = select_target_smiles()

    col1, col2 = st.columns([1, 2])
    with col1:
        analyze_clicked = render_target_panel(canonical_input)

    with col2:
        if analyze_clicked:
            credentials_error = _missing_credentials_message(settings)
            if credentials_error:
                st.session_state.pop("latest_retrosynthesis_run", None)
                st.error(credentials_error)
            elif not canonical_input:
                st.session_state.pop("latest_retrosynthesis_run", None)
                st.error("Invalid SMILES")
            else:
                st.session_state["latest_retrosynthesis_run"] = generate_plan(
                    canonical_input,
                    settings,
                )

        render_latest_run(canonical_input, settings)


if __name__ == "__main__":
    main()
