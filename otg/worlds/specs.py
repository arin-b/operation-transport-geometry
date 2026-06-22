from __future__ import annotations

from otg.core.data import WorldSpec


def make_spec(
    name: str,
    family: str,
    paper_core: bool,
    mathematical_definition: str,
    operational_coordinate_rule: str,
    nuisance_rule: str,
    expected_behavior: str,
    failure_modes: list[str],
    recommended_diagnostics: list[str],
) -> WorldSpec:
    return WorldSpec(
        name=name,
        family=family,
        paper_core=paper_core,
        mathematical_definition=mathematical_definition,
        operational_coordinate_rule=operational_coordinate_rule,
        nuisance_rule=nuisance_rule,
        expected_behavior=expected_behavior,
        failure_modes=failure_modes,
        recommended_diagnostics=recommended_diagnostics,
    )
