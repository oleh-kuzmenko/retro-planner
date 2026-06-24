import logging
import os
from dataclasses import dataclass

import streamlit as st
from streamlit_ketcher import st_ketcher

from retro_planner.chemistry import canonicalize_smiles
from retro_planner.config import DEFAULT_TARGET_SMILES, OPTIMIZATION_OBJECTIVES
from retro_planner.llm_providers import LLM_PROVIDER_REGISTRY, LLMProviderConfig
from retro_planner.planning import (
    GenerationRequest,
    call_llm_with_rag,
    get_retrosynthesis_plan,
)
from retro_planner.rendering import generate_molecule_image
from retro_planner.retrieval import (
    RetrievalConfig,
    create_qdrant_client,
    retrieve_reactions_for_smiles,
)
from retro_planner.streamlit_views import display_hybrid_retrieval, display_llm_answer


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
    target_smiles_only: bool
    rag_enabled: bool
    top_k: int
    route_count: int
    optimization_objective: str
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
        provider_key = st.selectbox(
            "LLM provider",
            options=list(LLM_PROVIDER_REGISTRY.keys()),
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
                help="OpenAI-compatible chat completions endpoint, for example Ollama /v1.",
            )
            base_url = base_url.strip()

        st.divider()
        target_smiles_only = False
        if provider_key == "custom_openai":
            target_smiles_only = st.toggle(
                "Send target SMILES only",
                value=False,
                help=(
                    "Send one user message containing only the canonical target SMILES. "
                    "No system prompt, custom prompt, RAG context, or JSON response format "
                    "is added."
                ),
            )
            if target_smiles_only:
                st.caption(
                    "The model response is interpreted as dot-separated reactant SMILES."
                )

        rag_enabled = st.checkbox(
            "RAG enabled",
            value=not target_smiles_only,
            disabled=target_smiles_only,
        )
        if target_smiles_only:
            rag_enabled = False
        top_k = st.slider(
            "Top-K",
            min_value=1,
            max_value=20,
            value=5,
            disabled=target_smiles_only,
        )
        route_count = st.slider(
            "Route options",
            min_value=1,
            max_value=5,
            value=3,
            disabled=target_smiles_only,
        )
        optimization_objective = st.selectbox(
            "Optimization objective",
            list(OPTIMIZATION_OBJECTIVES),
            disabled=target_smiles_only,
        )
        model = st.text_input(
            f"{provider_label} model",
            value=os.getenv(provider_config.model_env_var, provider_config.default_model),
        )

    return SidebarSettings(
        provider_key=provider_key,
        provider_config=provider_config,
        api_key=api_key,
        base_url=base_url,
        target_smiles_only=target_smiles_only,
        rag_enabled=rag_enabled,
        top_k=top_k,
        route_count=route_count,
        optimization_objective=optimization_objective,
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


def generate_plan(
    canonical_input: str,
    settings: SidebarSettings,
) -> dict:
    provider = settings.provider_config.create_provider(
        settings.api_key,
        settings.base_url,
    )
    reactions: list[dict] = []
    warnings: list[str] = []

    if settings.rag_enabled:
        with st.spinner("Retrieving similar reactions from Qdrant for model context..."):
            try:
                retrieval_result = retrieve_reactions_for_smiles(
                    canonical_input,
                    settings.top_k,
                    client=get_cached_qdrant_client(),
                )
                reactions = retrieval_result.reactions
                warnings.extend(retrieval_result.warnings)
            except Exception as exc:
                warnings.append(
                    f"Qdrant unavailable; generating without retrieved context. Details: {exc}"
                )

    request = GenerationRequest(
        target_smiles=canonical_input,
        llm_provider=provider,
        model=settings.model,
        optimization_objective=settings.optimization_objective,
        route_count=settings.route_count,
        reactions=reactions,
        target_smiles_only=settings.target_smiles_only,
    )
    with st.spinner(f"Generating route options with {settings.provider_label}..."):
        plan_result = (
            call_llm_with_rag(request)
            if settings.rag_enabled
            else get_retrosynthesis_plan(request)
        )

    warnings.extend(plan_result.warnings)
    return {
        "canonical_input": canonical_input,
        "provider_key": settings.provider_key,
        "model": settings.model,
        "base_url": settings.base_url,
        "target_smiles_only": settings.target_smiles_only,
        "route_count": settings.route_count,
        "rag_enabled": settings.rag_enabled,
        "reactions": reactions,
        "result": plan_result.result,
        "warnings": warnings,
        "errors": plan_result.errors,
    }


def render_latest_run(canonical_input: str | None, settings: SidebarSettings) -> None:
    latest_run = st.session_state.get("latest_retrosynthesis_run")
    if not (
        latest_run
        and latest_run.get("canonical_input") == canonical_input
        and latest_run.get("provider_key") == settings.provider_key
        and latest_run.get("model") == settings.model
        and latest_run.get("base_url") == settings.base_url
        and latest_run.get("target_smiles_only") == settings.target_smiles_only
        and latest_run.get("route_count") == settings.route_count
    ):
        return

    st.markdown("### Canonical SMILES")
    st.code(canonical_input, language="text")

    _show_messages(latest_run.get("warnings", []), "warning")
    _show_messages(latest_run.get("errors", []), "error")

    if latest_run.get("rag_enabled"):
        display_hybrid_retrieval(latest_run.get("reactions", []))

    if latest_run.get("result"):
        display_llm_answer(latest_run.get("result"), canonical_input)
    elif not latest_run.get("errors"):
        st.error("LLM API failure")


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
            if settings.provider_config.api_key_required and not settings.api_key:
                st.session_state.pop("latest_retrosynthesis_run", None)
                st.error(
                    f"Missing {settings.provider_config.api_key_env_var}. Add it in the sidebar "
                    f"or set the {settings.provider_config.api_key_env_var} environment variable."
                )
            elif settings.provider_config.base_url_env_var and not settings.base_url:
                st.session_state.pop("latest_retrosynthesis_run", None)
                st.error(
                    f"Missing {settings.provider_config.base_url_env_var}. Add it in the sidebar "
                    f"or set the {settings.provider_config.base_url_env_var} environment variable."
                )
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
