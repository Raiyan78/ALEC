from __future__ import annotations

from math import pi
from typing import Sequence


def concatenate_qsp_phase_lists(
    left_phases: Sequence[float],
    right_phases: Sequence[float],
) -> tuple[float, ...]:
    """Compose two QSP phase lists."""

    if not left_phases or not right_phases:
        raise ValueError("both phase lists must contain at least one angle")

    return (
        *(float(x) for x in left_phases[:-1]),
        float(left_phases[-1]) + float(right_phases[0]),
        *(float(x) for x in right_phases[1:]),
    )


def apply_conjugation_to_phase_list(
    phases: Sequence[float],
    m: int,
    n: int,
    eta: float,
) -> tuple[float, ...]:
    """Apply one conjugation step from the recovery construction."""

    if not phases:
        raise ValueError("phases must contain at least one angle")

    return (
        -(eta + pi * (2 * m + n + 0.5)),
        pi * (n + 0.5) + float(phases[0]),
        *(float(x) for x in phases[1:]),
        float(eta),
    )


def _normalize_half_angle(angle: float) -> float:
    """Normalize into (-pi, 0]."""

    while angle > 0.0:
        angle -= pi
    while angle <= -pi:
        angle += pi
    return float(angle)


def _conjugation_chain_phase_list(
    etas: Sequence[float],
    *,
    n: int,
) -> tuple[float, ...]:
    """Build a recovery chain from the list of half-angles."""

    phases = (0.0,)
    for eta in etas:
        phases = apply_conjugation_to_phase_list(phases, 0, n, float(eta))
    return phases


__all__ = [
    "apply_conjugation_to_phase_list",
    "concatenate_qsp_phase_lists",
    "_conjugation_chain_phase_list",
    "_normalize_half_angle",
]
