from retro_planner.planning import (
    filter_target_matching_routes,
    route_final_product_matches_target,
)


def _route(product_smiles: str) -> dict:
    return {
        "route_name": "Route",
        "steps": [
            {
                "step_number": 1,
                "reactants": ["CCO"],
                "product_smiles": product_smiles,
            }
        ],
    }


def test_route_final_product_matches_target_after_canonicalization():
    assert route_final_product_matches_target(_route("OC(C)=O"), "CC(=O)O")


def test_filter_target_matching_routes_removes_off_target_routes():
    result = {
        "routes": [
            _route("OC(C)=O"),
            _route("CCO"),
        ],
        "overall_recommendation": "Use route 1",
    }

    filtered, removed_count = filter_target_matching_routes(result, "CC(=O)O")

    assert removed_count == 1
    assert filtered is not None
    assert len(filtered["routes"]) == 1
    assert filtered["overall_recommendation"] == "Use route 1"


def test_filter_target_matching_routes_returns_none_when_all_routes_off_target():
    filtered, removed_count = filter_target_matching_routes(
        {"routes": [_route("CCO")]},
        "CC(=O)O",
    )

    assert filtered is None
    assert removed_count == 1
