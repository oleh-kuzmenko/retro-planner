import re

import streamlit as st

from retro_planner.chemistry import (
    canonicalize_smiles,
    is_known_formula_smiles,
    parse_reaction_smiles,
)
from retro_planner.rendering import generate_reaction_image


PLACEHOLDER_VALUES = {"", "none", "n/a", "na", "null", "unknown", "not specified"}
NUMERIC_TEXT_RE = re.compile(r"^\s*-?\d+(?:\.\d+)?\s*$")


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


def display_condition_value(field: str, value, fallback: str = "N/A") -> str:
    displayed = display_value(value, fallback=fallback)
    if displayed == fallback:
        return displayed

    field_lower = field.lower()
    if NUMERIC_TEXT_RE.match(displayed):
        if field_lower == "temperature":
            return f"{displayed} deg C"
        if field_lower == "time":
            return f"{displayed} h"
        if field_lower == "yield":
            return f"{displayed}%"
    return displayed


def clean_smiles_list(smiles_values) -> list[str]:
    if not isinstance(smiles_values, list):
        return []
    return [
        clean
        for clean in (canonicalize_smiles(str(smiles)) for smiles in smiles_values)
        if clean
    ]


def display_smiles_list(smiles_values) -> list[str]:
    if not isinstance(smiles_values, list):
        return []
    return [
        display_smiles(str(smiles))
        for smiles in smiles_values
        if not is_placeholder_value(smiles)
    ]


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


def route_steps(route: dict) -> list[dict]:
    steps = route.get("steps", [])
    if not isinstance(steps, list):
        return []
    return [step for step in steps if isinstance(step, dict)][:5]


def display_reaction_step(
    result: dict,
    canonical_input: str,
    caption: str = "Retrosynthetic reaction",
    is_final_step: bool = True,
) -> None:
    reaction_name = result.get("reaction_name", "Proposed reaction")
    raw_product_smiles = result.get("product_smiles") or (
        canonical_input if is_final_step else ""
    )
    product_smiles = canonicalize_smiles(str(raw_product_smiles)) or (
        canonical_input if is_final_step else None
    )
    reactants = result.get("reactants", [])
    clean_reactants = clean_smiles_list(reactants)
    displayed_reactants = display_smiles_list(reactants)
    displayed_product = display_smiles(str(raw_product_smiles))

    st.markdown(f"**Reaction:** {reaction_name}")

    col_a, col_b = st.columns([1, 1])
    with col_a:
        st.caption("Reactants")
        st.code(" + ".join(displayed_reactants) if displayed_reactants else "N/A", language="text")
    with col_b:
        st.caption("Product")
        st.code(displayed_product or "N/A", language="text")

    condition_rows = {
        "Components shown in scheme": displayed_reactants,
        "Stoichiometry": result.get("stoichiometry"),
        "Reagents": result.get("reagents"),
        "Solvent": result.get("solvent"),
        "Temperature": result.get("temperature_celsius"),
        "Time": result.get("reaction_time"),
        "Atmosphere": result.get("atmosphere"),
        "Yield": result.get("expected_yield_percent"),
        "Workup / purification": result.get("workup_purification"),
        "Important conditions": result.get("important_conditions"),
    }
    st.table([
        {"Field": key, "Value": display_condition_value(key, value)}
        for key, value in condition_rows.items()
    ])

    if result.get("rationale"):
        st.markdown(f"**Rationale:** {result.get('rationale')}")

    if result.get("objective_fit"):
        st.markdown(f"**Objective Fit:** {result.get('objective_fit')}")

    evidence_reaction_ids = result.get("evidence_reaction_ids")
    if evidence_reaction_ids:
        st.markdown(f"**Evidence reactions:** {display_value(evidence_reaction_ids)}")

    image = generate_reaction_image(clean_reactants, product_smiles)

    if image:
        st.image(image, caption=caption, width="content")
    elif clean_reactants or product_smiles:
        st.warning("Could not render reaction image. Check the returned SMILES.")
        st.write("Raw Reactant SMILES:", reactants)
        if displayed_reactants and not clean_reactants:
            st.caption("All returned reactant SMILES failed RDKit parsing.")


def display_route(route: dict, route_index: int, canonical_input: str) -> None:
    route_name = route.get("route_name") or f"Route {route_index}"
    with st.expander(f"Option {route_index}: {route_name}", expanded=route_index == 1):
        summary = route.get("summary")
        if summary:
            st.markdown(f"**Summary:** {display_value(summary)}")

        if route.get("objective_fit"):
            st.markdown(f"**Objective Fit:** {route.get('objective_fit')}")

        evidence_reaction_ids = route.get("evidence_reaction_ids")
        if evidence_reaction_ids:
            st.markdown(f"**Evidence reactions:** {display_value(evidence_reaction_ids)}")

        steps = route_steps(route)
        if not steps and route.get("product_smiles"):
            steps = [route]
        if not steps:
            st.warning("This route did not include usable steps.")
            return

        for step_index, step in enumerate(steps, start=1):
            if len(steps) > 1:
                st.markdown(f"**Step {step_index}**")
            display_reaction_step(
                step,
                canonical_input,
                caption=f"Option {route_index}, step {step_index}",
                is_final_step=step_index == len(steps),
            )
            if step_index < len(steps):
                st.divider()


def display_llm_answer(result, canonical_input: str):
    st.subheader("Generated retrosynthesis options")

    if isinstance(result, str):
        st.markdown(result)
        return

    if not isinstance(result, dict):
        st.error("The LLM returned an empty or unsupported response.")
        return

    routes = result.get("routes")
    if isinstance(routes, list) and routes:
        for route_index, route in enumerate(routes, start=1):
            if isinstance(route, dict):
                display_route(route, route_index, canonical_input)

        if result.get("overall_recommendation"):
            st.markdown(
                f"**Overall recommendation:** {display_value(result.get('overall_recommendation'))}"
            )
        return

    display_reaction_step(result, canonical_input)
