from __future__ import annotations

from dataclasses import dataclass
from math import ceil, pi
from typing import Sequence

import numpy as np

from .phase_list import (
    _conjugation_chain_phase_list,
    _normalize_half_angle,
    concatenate_qsp_phase_lists,
)
from .qsp_builder import Matrix, build_qsp_unitary


_PAULI_X = np.array([[0.0, 1.0], [1.0, 0.0]], dtype=complex)
_PAULI_Y = np.array([[0.0, -1j], [1j, 0.0]], dtype=complex)
_PAULI_Z = np.array([[1.0, 0.0], [0.0, -1.0]], dtype=complex)
_IDENTITY = np.eye(2, dtype=complex)


@dataclass(frozen=True)
class FirstOrderErrorProfileFit:
    theta_grid: tuple[float, ...]
    p_x: tuple[float, ...]
    p_y: tuple[float, ...]
    j_max: int
    max_fit_residual_x: float
    max_fit_residual_y: float
    eps_probe: float


@dataclass(frozen=True)
class DegreewiseRecoveryIteration:
    iteration: int
    j_max: int
    p_x: tuple[float, ...]
    p_y: tuple[float, ...]
    n: int
    etas: tuple[float, ...]
    block_phase_list: tuple[float, ...]
    block_length: int


@dataclass(frozen=True)
class DegreewiseRecoveryResult:
    recovery_phase_list: tuple[float, ...]
    iterations: tuple[DegreewiseRecoveryIteration, ...]
    final_profile: FirstOrderErrorProfileFit
    eps_probe: float
    tol: float


def pauli_decomposition(matrix: Matrix) -> tuple[complex, complex, complex, complex]:
    """Return the I, X, Y, Z coefficients of a 2x2 matrix."""

    return (
        np.trace(matrix) / 2.0,
        np.trace(matrix @ _PAULI_X) / 2.0,
        np.trace(matrix @ _PAULI_Y) / 2.0,
        np.trace(matrix @ _PAULI_Z) / 2.0,
    )


def _default_profile_theta_grid(num_points: int) -> tuple[float, ...]:
    return tuple(np.linspace(0.12, np.pi / 2.0 - 0.12, num_points))


def corrected_error_operator_from_phase_list(
    phases: Sequence[float],
    recovery_phase_list: Sequence[float],
    theta: float,
    epsilon: float,
) -> Matrix:
    """Return U_ideal(theta)^dagger U_corrected(theta, epsilon)."""

    ideal = build_qsp_unitary(phases, theta, 0.0)
    corrected = build_qsp_unitary(phases, theta, epsilon) @ build_qsp_unitary(
        recovery_phase_list,
        theta,
        epsilon,
    )
    return ideal.conj().T @ corrected


def fit_first_order_error_profile(
    phases: Sequence[float],
    *,
    recovery_phase_list: Sequence[float] = (0.0,),
    eps_probe: float = 1e-7,
    theta_grid: Sequence[float] | None = None,
    tol: float = 1e-5,
    j_max_threshold: float | None = None,
) -> FirstOrderErrorProfileFit:
    """Fit the first-order X/Y error profile in the even-power cosine basis."""

    if not phases:
        raise ValueError("phases must contain at least one angle")
    if len(phases) < 2:
        raise ValueError("phases must describe a non-trivial QSP sequence")

    if theta_grid is None:
        theta_grid = _default_profile_theta_grid(max(41, 16 * (len(phases) - 1) + 1))
    theta_values = tuple(float(theta) for theta in theta_grid)

    max_degree = len(phases) - 1
    design_matrix = np.empty((len(theta_values), max_degree), dtype=float)
    target_x = np.empty(len(theta_values), dtype=float)
    target_y = np.empty(len(theta_values), dtype=float)

    for i, theta in enumerate(theta_values):
        sin_2theta = np.sin(2.0 * theta)
        if abs(sin_2theta) < 1e-8:
            raise ValueError("theta_grid must stay away from sin(2 theta)=0")

        error_operator = corrected_error_operator_from_phase_list(
            phases,
            recovery_phase_list,
            theta,
            eps_probe,
        )
        generator = (error_operator - _IDENTITY) / eps_probe
        _, c_x, c_y, _ = pauli_decomposition(generator)

        cosine = np.cos(theta)
        design_matrix[i, :] = [cosine ** (2 * degree) for degree in range(max_degree)]
        target_x[i] = np.imag(c_x) / sin_2theta
        target_y[i] = np.imag(c_y) / sin_2theta

    p_x, *_ = np.linalg.lstsq(design_matrix, target_x, rcond=None)
    p_y, *_ = np.linalg.lstsq(design_matrix, target_y, rcond=None)

    residual_x = float(np.max(np.abs(design_matrix @ p_x - target_x)))
    residual_y = float(np.max(np.abs(design_matrix @ p_y - target_y)))

    threshold = tol if j_max_threshold is None else j_max_threshold
    j_max = -1
    for degree in range(max_degree - 1, -1, -1):
        if max(abs(p_x[degree]), abs(p_y[degree])) > threshold:
            j_max = degree
            break

    return FirstOrderErrorProfileFit(
        theta_grid=theta_values,
        p_x=tuple(float(x) for x in p_x),
        p_y=tuple(float(y) for y in p_y),
        j_max=j_max,
        max_fit_residual_x=residual_x,
        max_fit_residual_y=residual_y,
        eps_probe=float(eps_probe),
    )


def _profile_reference_scale(profile: FirstOrderErrorProfileFit) -> float:
    return max(
        max((abs(x) for x in profile.p_x), default=0.0),
        max((abs(y) for y in profile.p_y), default=0.0),
    )


def recovery_phase_list_k1_degreewise(
    phases: Sequence[float],
    *,
    eps_probe: float = 1e-7,
    tol: float = 1e-5,
    tol_relative: float = 1e-4,
    theta_grid: Sequence[float] | None = None,
) -> DegreewiseRecoveryResult:
    """Build the k=1 recovery list by cancelling the top surviving degree."""

    phases = tuple(float(phase) for phase in phases)
    theta_grid = None if theta_grid is None else tuple(float(theta) for theta in theta_grid)

    recovery_phase_list = (0.0,)
    iterations: list[DegreewiseRecoveryIteration] = []

    initial_profile = fit_first_order_error_profile(
        phases,
        recovery_phase_list=recovery_phase_list,
        eps_probe=eps_probe,
        theta_grid=theta_grid,
        tol=tol,
    )
    j_max_threshold = tol_relative * _profile_reference_scale(initial_profile)

    for iteration in range(len(phases)):
        profile = fit_first_order_error_profile(
            phases,
            recovery_phase_list=recovery_phase_list,
            eps_probe=eps_probe,
            theta_grid=theta_grid,
            tol=tol,
            j_max_threshold=j_max_threshold,
        )
        if profile.j_max < 0:
            return DegreewiseRecoveryResult(
                recovery_phase_list=recovery_phase_list,
                iterations=tuple(iterations),
                final_profile=profile,
                eps_probe=eps_probe,
                tol=tol,
            )

        p_x = np.asarray(profile.p_x)
        p_y = np.asarray(profile.p_y)
        norm = float(np.hypot(p_x[profile.j_max], p_y[profile.j_max]))

        if profile.j_max == 0:
            n = max(0, int(ceil(norm / pi - 0.5)))
            eta_base = _normalize_half_angle(-0.5 * np.arctan2(p_x[0], p_y[0]))
            delta_eta = 0.5 * np.arccos(np.clip(norm / (2.0 * pi * (n + 0.5)), 0.0, 1.0))
            etas = (float(eta_base + delta_eta), float(eta_base - delta_eta))
            first = _conjugation_chain_phase_list((etas[0],), n=n)
            second = _conjugation_chain_phase_list((etas[1],), n=n)
            block_phase_list = concatenate_qsp_phase_lists(first, second)
        else:
            scale = pi * (2.0 ** (2 * profile.j_max))
            n = max(0, int(ceil(norm / scale - 0.5)))
            denominator = scale * (n + 0.5)
            eta_top = np.arccos(np.sqrt(np.clip(norm / denominator, 0.0, 1.0)))
            eta_last = _normalize_half_angle(-0.5 * np.arctan2(p_x[profile.j_max], p_y[profile.j_max]))
            etas = (*([0.0] * (profile.j_max - 1)), float(eta_top), float(eta_last))
            block_phase_list = _conjugation_chain_phase_list(etas, n=n)

        recovery_phase_list = concatenate_qsp_phase_lists(recovery_phase_list, block_phase_list)
        iterations.append(
            DegreewiseRecoveryIteration(
                iteration=iteration,
                j_max=profile.j_max,
                p_x=profile.p_x,
                p_y=profile.p_y,
                n=n,
                etas=tuple(etas),
                block_phase_list=tuple(float(x) for x in block_phase_list),
                block_length=len(block_phase_list) - 1,
            )
        )

    final_profile = fit_first_order_error_profile(
        phases,
        recovery_phase_list=recovery_phase_list,
        eps_probe=eps_probe,
        theta_grid=theta_grid,
        tol=tol,
        j_max_threshold=j_max_threshold,
    )
    return DegreewiseRecoveryResult(
        recovery_phase_list=recovery_phase_list,
        iterations=tuple(iterations),
        final_profile=final_profile,
        eps_probe=eps_probe,
        tol=tol,
    )


def build_corrected_qsp_phase_list(
    phases: Sequence[float],
    *,
    eps_probe: float = 1e-7,
    tol: float = 1e-5,
    tol_relative: float = 1e-4,
    theta_grid: Sequence[float] | None = None,
) -> tuple[float, ...]:
    """Merge the base QSP phases with the degree-wise k=1 recovery."""

    base_phases = tuple(float(phase) for phase in phases)
    recovery_result = recovery_phase_list_k1_degreewise(
        base_phases,
        eps_probe=eps_probe,
        tol=tol,
        tol_relative=tol_relative,
        theta_grid=theta_grid,
    )
    return concatenate_qsp_phase_lists(base_phases, recovery_result.recovery_phase_list)


__all__ = [
    "DegreewiseRecoveryIteration",
    "DegreewiseRecoveryResult",
    "FirstOrderErrorProfileFit",
    "build_corrected_qsp_phase_list",
    "corrected_error_operator_from_phase_list",
    "fit_first_order_error_profile",
    "pauli_decomposition",
    "recovery_phase_list_k1_degreewise",
]
