import streamlit as st

from retro_planner.chemistry import canonicalize_smiles, parse_reaction_smiles
from retro_planner.rendering import generate_reaction_image


def display_value(value, fallback: str = "N/A") -> str:
    if value is None or value == "":
        return fallback
    if isinstance(value, list):
        return ", ".join(str(item) for item in value) if value else fallback
    return str(value)


def clean_smiles_list(smiles_values) -> list[str]:
    if not isinstance(smiles_values, list):
        return []
    return [
        clean
        for clean in (canonicalize_smiles(str(smiles)) for smiles in smiles_values)
        if clean
    ]


def route_steps(route: dict) -> list[dict]:
    steps = route.get("steps", [])
    if not isinstance(steps, list):
        return []
    return [step for step in steps if isinstance(step, dict)][:5]


def display_route_step(step: dict, canonical_input: str, is_last_step: bool):
    step_number = step.get("step_number", "?")
    reaction_name = step.get("reaction_name", "Reaction step")
    st.markdown(f"**Step {step_number}: {reaction_name}**")

    raw_product_smiles = step.get("product_smiles") or ""
    product_smiles = canonicalize_smiles(str(raw_product_smiles))
    if not product_smiles and is_last_step:
        product_smiles = canonical_input

    reactants = clean_smiles_list(step.get("reactants", []))

    col_a, col_b = st.columns([1, 1])
    with col_a:
        st.caption("Reactants")
        st.code(" + ".join(reactants) if reactants else "N/A", language="text")
    with col_b:
        st.caption("Product")
        st.code(product_smiles or "N/A", language="text")

    condition_rows = {
        "Stoichiometry": step.get("stoichiometry"),
        "Reagents": step.get("reagents"),
        "Solvent": step.get("solvent"),
        "Temperature": step.get("temperature_celsius"),
        "Time": step.get("reaction_time"),
        "Atmosphere": step.get("atmosphere"),
        "Yield": step.get("expected_yield_percent"),
        "Workup / purification": step.get("workup_purification"),
        "Important conditions": step.get("important_conditions"),
    }
    st.table([
        {"Field": key, "Value": display_value(value)}
        for key, value in condition_rows.items()
    ])

    if step.get("rationale"):
        st.markdown(f"**Rationale:** {step.get('rationale')}")

    image = generate_reaction_image(reactants, product_smiles)
    if image:
        st.image(image, caption=f"Step {step_number}", width="content")
    elif reactants or product_smiles:
        st.warning("Could not render this step image. Check the returned SMILES.")


def display_route(route: dict, route_index: int, canonical_input: str):
    route_name = route.get("route_name") or f"Route {route_index}"
    with st.expander(f"Route {route_index}: {route_name}", expanded=route_index == 1):
        if route.get("strategy"):
            st.markdown(f"**Strategy:** {route.get('strategy')}")

        summary = route.get("summary", {})
        if isinstance(summary, dict):
            st.table([
                {"Aspect": "Difficulty", "Assessment": display_value(summary.get("difficulty"))},
                {"Aspect": "Cleanliness", "Assessment": display_value(summary.get("cleanliness"))},
                {"Aspect": "Cost", "Assessment": display_value(summary.get("cost"))},
                {"Aspect": "Overall yield", "Assessment": display_value(summary.get("expected_overall_yield"))},
                {"Aspect": "Major risks", "Assessment": display_value(summary.get("major_risks"))},
                {"Aspect": "Best for", "Assessment": display_value(summary.get("best_for"))},
            ])

        if route.get("objective_fit"):
            st.markdown(f"**Objective Fit:** {route.get('objective_fit')}")

        steps = route_steps(route)
        if not steps:
            st.warning("This route did not include usable steps.")
            return

        for idx, step in enumerate(steps):
            display_route_step(step, canonical_input, is_last_step=idx == len(steps) - 1)
            if idx < len(steps) - 1:
                st.divider()


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


def display_llm_answer(result, canonical_input: str):
    st.subheader("Generated retrosynthesis routes")

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
            st.success(f"Overall recommendation: {result.get('overall_recommendation')}")
        return

    st.markdown(f"**Proposed Path:** {result.get('reaction_name', 'Unknown Reaction')}")
    st.info(f"**Reagents/Conditions:** {result.get('reagents', 'N/A')}")
    st.markdown(f"**Chemist's Reasoning:** {result.get('reasoning', 'N/A')}")

    if result.get("objective_fit"):
        st.markdown(f"**Objective Fit:** {result.get('objective_fit')}")

    reactants = result.get("reactants", [])
    clean_reactants = clean_smiles_list(reactants)
    image = generate_reaction_image(clean_reactants, canonical_input)

    if image:
        st.image(image, caption="Retrosynthetic Step", width="content")
    elif reactants:
        st.warning("Could not render reaction image (SMILES might be chemically invalid).")
        st.write("Raw Reactant SMILES:", reactants)
