import streamlit as st

from retro_planner.chemistry import (
    canonicalize_smiles,
    is_known_formula_smiles,
    parse_reaction_smiles,
)
from retro_planner.planning import StepResult
from retro_planner.rendering import generate_reaction_image


PLACEHOLDER_VALUES = {"", "none", "n/a", "na", "null", "unknown", "not specified"}


def is_placeholder_value(value) -> bool:
    if value is None:
        return True
    if isinstance(value, str):
        return value.strip().lower() in PLACEHOLDER_VALUES
    return False


def display_smiles(smiles: str) -> str:
    if is_known_formula_smiles(smiles):
        return str(smiles).strip()
    return canonicalize_smiles(smiles) or str(smiles)


def display_value(value, fallback: str = "N/A") -> str:
    if is_placeholder_value(value):
        return fallback
    if isinstance(value, list):
        displayed = [display_value(item, fallback="") for item in value]
        displayed = [item for item in displayed if item]
        return ", ".join(displayed) if displayed else fallback
    return str(value)


def display_hybrid_retrieval(reactions: list[dict]):
    if not reactions:
        return

    with st.expander("Hybrid retrieval score breakdown", expanded=True):
        rows = [
            {
                "reaction_id": reaction.get("reaction_id"),
                "reaction_class": reaction.get("reaction_class") or "N/A",
                "molecule_similarity": reaction.get("molecule_similarity", 0.0),
                "reaction_similarity": reaction.get("reaction_similarity", 0.0),
                "reaction_class_similarity": reaction.get(
                    "reaction_class_similarity",
                    0.0,
                ),
                "final_hybrid_score": reaction.get("final_hybrid_score", 0.0),
                "reaction_smiles": reaction.get("reaction_smiles"),
            }
            for reaction in reactions
        ]

        column_config = {
            "molecule_similarity": st.column_config.NumberColumn(
                "molecule_similarity",
                format="%.4f",
            ),
            "reaction_similarity": st.column_config.NumberColumn(
                "reaction_similarity",
                format="%.4f",
            ),
            "reaction_class_similarity": st.column_config.NumberColumn(
                "reaction_class_similarity",
                format="%.4f",
            ),
            "final_hybrid_score": st.column_config.NumberColumn(
                "final_hybrid_score",
                format="%.4f",
            ),
            "reaction_smiles": st.column_config.TextColumn(
                "reaction_smiles",
                width="large",
            ),
        }

        selected_row = None
        try:
            table_event = st.dataframe(
                rows,
                width="stretch",
                column_config=column_config,
                hide_index=True,
                key="hybrid_retrieval_table",
                on_select="rerun",
                selection_mode="single-row",
            )
            selected_indices = table_event.selection.rows
            if selected_indices:
                selected_row = rows[selected_indices[0]]
        except TypeError:
            st.dataframe(
                rows,
                width="stretch",
                column_config=column_config,
                hide_index=True,
            )
            selected_label = st.selectbox(
                "Preview retrieved reaction",
                options=list(range(len(rows))),
                format_func=lambda idx: (
                    f"{rows[idx].get('reaction_id') or 'reaction'} "
                    f"(score {rows[idx].get('final_hybrid_score', 0.0):.4f})"
                ),
                key="hybrid_retrieval_preview_select",
            )
            selected_row = rows[selected_label]

        if not selected_row:
            st.caption("Select a reaction row to preview its structural equation.")
            return

        st.markdown("**Selected retrieved reaction**")
        col_a, col_b = st.columns([1, 1])
        with col_a:
            st.caption("Reaction ID")
            st.code(display_value(selected_row.get("reaction_id")), language="text")
        with col_b:
            st.caption("Reaction class")
            st.code(display_value(selected_row.get("reaction_class")), language="text")

        reaction_smiles = selected_row.get("reaction_smiles")
        st.caption("reaction_smiles")
        st.code(display_value(reaction_smiles), language="text")

        parsed_reaction = parse_reaction_smiles(reaction_smiles)
        if not parsed_reaction:
            st.warning("Could not parse this reaction SMILES for preview.")
            return

        reactants, products = parsed_reaction
        image = generate_reaction_image(reactants, products)
        if image:
            st.image(image, caption="Retrieved reaction", width="content")
        else:
            st.warning("Could not render this retrieved reaction image.")


def display_step_result(
    step_result: StepResult,
    canonical_input: str,
    candidate_label: str | None = None,
) -> None:
    st.subheader(candidate_label or "Generated retrosynthesis step")

    for message in step_result.warnings:
        st.warning(message)
    for message in step_result.errors:
        st.error(message)

    if step_result.think:
        with st.expander("🧠 Chemist's Reasoning", expanded=True):
            st.markdown(step_result.think)

    if not step_result.precursors:
        st.error("The model did not return a chemically valid set of precursors.")
        return

    st.markdown("**Proposed single-step disconnection**")
    col_a, col_b = st.columns([1, 1])
    with col_a:
        st.caption("Precursors")
        st.code(" + ".join(step_result.precursors), language="text")
    with col_b:
        st.caption("Product (target)")
        st.code(step_result.product_smiles or canonical_input, language="text")

    image = generate_reaction_image(step_result.precursors, step_result.product_smiles)
    if image:
        st.image(image, caption="Proposed retrosynthetic step", width="content")
    else:
        st.warning("Could not render reaction image. Check the returned SMILES.")
