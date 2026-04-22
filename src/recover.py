from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from math import ceil, pi
from typing import Sequence

import numpy as np

try:
    from .phase_list import (
        _conjugation_chain_phase_list,
        _normalize_half_angle,
        concatenate_qsp_phase_lists,
    )
    from .qsp_builder import Matrix, build_qsp_unitary
except ImportError:  # pragma: no cover - lets notebooks import directly from src/
    from phase_list import (
        _conjugation_chain_phase_list,
        _normalize_half_angle,
        concatenate_qsp_phase_lists,
    )
    from qsp_builder import Matrix, build_qsp_unitary


_PAULI_X = np.array([[0.0, 1.0], [1.0, 0.0]], dtype=complex)
_PAULI_Y = np.array([[0.0, -1j], [1j, 0.0]], dtype=complex)
_PAULI_Z = np.array([[1.0, 0.0], [0.0, -1.0]], dtype=complex)
_IDENTITY = np.eye(2, dtype=complex)


@dataclass(frozen=True)
class FirstOrderErrorProfileFit:
    """Fitted first-order X/Y error profile for one recovery attempt.

    `p_x` and `p_y` are the polynomial coefficients from the paper's canonical
    first-order form. `j_max` is the highest degree that is still non-negligible.
    That is the degree the next recovery block will target.
    """

    theta_grid: tuple[float, ...]
    p_x: tuple[float, ...]
    p_y: tuple[float, ...]
    j_max: int
    max_fit_residual_x: float
    max_fit_residual_y: float
    eps_probe: float


@dataclass(frozen=True)
class DegreewiseRecoveryIteration:
    """One step of the top-degree cancellation loop."""

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
    """Final result of the degree-wise construction."""

    recovery_phase_list: tuple[float, ...]
    iterations: tuple[DegreewiseRecoveryIteration, ...]
    final_profile: FirstOrderErrorProfileFit
    eps_probe: float
    tol: float


def pauli_decomposition(matrix: Matrix) -> tuple[complex, complex, complex, complex]:
    """Decompose a 2x2 matrix into I, X, Y, Z components.

    The recovery proof talks about the first-order error generator in Pauli form.
    This helper turns a raw matrix into those four coefficients.
    """

    return (
        np.trace(matrix) / 2.0,
        np.trace(matrix @ _PAULI_X) / 2.0,
        np.trace(matrix @ _PAULI_Y) / 2.0,
        np.trace(matrix @ _PAULI_Z) / 2.0,
    )


def _default_profile_theta_grid(num_points: int) -> tuple[float, ...]:
    """Pick theta samples away from the singular endpoints.

    We divide by `sin(2 theta)` later, so values too close to `0` or `pi/2`
    would make the fit unstable.
    """

    return tuple(float(theta) for theta in np.linspace(0.12, np.pi / 2.0 - 0.12, num_points))


def _effective_corrected_unitary(
    phases: Sequence[float],
    recovery_phase_list: Sequence[float],
    theta: float,
    epsilon: float,
    *,
    append_recovery_on_right: bool = True,
) -> Matrix:
    """Build the noisy circuit with the recovery sequence attached."""

    bare = build_qsp_unitary(phases, theta, epsilon)
    recovery = build_qsp_unitary(recovery_phase_list, theta, epsilon)
    if append_recovery_on_right:
        return bare @ recovery
    return recovery @ bare


def corrected_error_operator_from_phase_list(
    phases: Sequence[float],
    recovery_phase_list: Sequence[float],
    theta: float,
    epsilon: float,
    *,
    append_recovery_on_right: bool = True,
) -> Matrix:
    """Return the residual error after factoring out the ideal QSP action.

    The corrected circuit itself is not the cleanest thing to fit. The object we
    care about is:

    `U_ideal(theta)^dagger * U_corrected(theta, epsilon)`

    If recovery is working, this should be close to identity, and its first-order
    part is exactly what the degree-wise procedure cancels.
    """

    ideal = build_qsp_unitary(phases, theta, 0.0)
    corrected = _effective_corrected_unitary(
        phases,
        recovery_phase_list,
        theta,
        epsilon,
        append_recovery_on_right=append_recovery_on_right,
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
    append_recovery_on_right: bool = True,
) -> FirstOrderErrorProfileFit:
    """Fit the leading first-order error profile for the current recovery.

    Intuition:

    1. Probe the corrected circuit at a tiny epsilon.
    2. Approximate the first-order generator by `(E - I) / eps_probe`.
    3. Read off the X and Y Pauli pieces.
    4. Fit those pieces to the expected polynomial basis in `cos(theta)^2`.

    The output tells us which degree is still left to cancel.
    """

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

    for index, theta in enumerate(theta_values):
        sin_2theta = np.sin(2.0 * theta)
        if abs(sin_2theta) < 1e-8:
            raise ValueError("theta_grid must stay away from singular endpoints where sin(2 theta)=0")

        error_operator = corrected_error_operator_from_phase_list(
            phases,
            recovery_phase_list,
            theta,
            eps_probe,
            append_recovery_on_right=append_recovery_on_right,
        )

        # For tiny epsilon, this is the first derivative of the error operator.
        generator = (error_operator - _IDENTITY) / eps_probe
        _, c_x, c_y, _ = pauli_decomposition(generator)

        cosine = np.cos(theta)
        design_matrix[index, :] = [cosine ** (2 * degree) for degree in range(max_degree)]

        # The paper factors out sin(2 theta), leaving the polynomial coefficients
        # we call p_x and p_y.
        target_x[index] = float(np.imag(c_x) / sin_2theta)
        target_y[index] = float(np.imag(c_y) / sin_2theta)

    p_x, *_ = np.linalg.lstsq(design_matrix, target_x, rcond=None)
    p_y, *_ = np.linalg.lstsq(design_matrix, target_y, rcond=None)

    residual_x = float(np.max(np.abs(design_matrix @ p_x - target_x)))
    residual_y = float(np.max(np.abs(design_matrix @ p_y - target_y)))

    threshold = float(tol if j_max_threshold is None else j_max_threshold)
    j_max = -1
    for degree in range(max_degree - 1, -1, -1):
        if max(abs(p_x[degree]), abs(p_y[degree])) > threshold:
            j_max = degree
            break

    return FirstOrderErrorProfileFit(
        theta_grid=theta_values,
        p_x=tuple(float(value) for value in p_x),
        p_y=tuple(float(value) for value in p_y),
        j_max=j_max,
        max_fit_residual_x=residual_x,
        max_fit_residual_y=residual_y,
        eps_probe=float(eps_probe),
    )


def _profile_reference_scale(profile: FirstOrderErrorProfileFit) -> float:
    """Use the initial fit size as the natural cancellation scale."""

    return max(
        max((abs(value) for value in profile.p_x), default=0.0),
        max((abs(value) for value in profile.p_y), default=0.0),
    )


def _degreewise_recovery_result_uncached(
    phases: tuple[float, ...],
    eps_probe: float,
    tol: float,
    tol_relative: float,
    theta_grid: tuple[float, ...] | None,
) -> DegreewiseRecoveryResult:
    """Run the iterative top-degree cancellation loop.

    At each step we:

    1. fit the current first-order profile
    2. find the highest surviving degree `j_max`
    3. build a recovery block that cancels exactly that degree
    4. append that block to the running recovery phase list
    """

    recovery_phase_list = (0.0,)
    iterations: list[DegreewiseRecoveryIteration] = []
    max_iterations = len(phases)

    initial_profile = fit_first_order_error_profile(
        phases,
        recovery_phase_list=recovery_phase_list,
        eps_probe=eps_probe,
        theta_grid=theta_grid,
        tol=tol,
    )
    j_max_threshold = tol_relative * _profile_reference_scale(initial_profile)

    for iteration_index in range(max_iterations):
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

        p_x = np.asarray(profile.p_x, dtype=float)
        p_y = np.asarray(profile.p_y, dtype=float)
        norm = float(np.hypot(p_x[profile.j_max], p_y[profile.j_max]))

        if profile.j_max >= 1:
            # For higher degrees, one conjugation chain is enough.
            # `eta_top` fixes the size of the highest-degree term.
            # `eta_last` rotates that term into the right X/Y direction.
            scale = pi * (2.0 ** (2 * profile.j_max))
            n = max(0, int(ceil(norm / scale - 0.5)))
            denominator = scale * (n + 0.5)
            eta_top = float(np.arccos(np.sqrt(np.clip(norm / denominator, 0.0, 1.0))))
            eta_last = _normalize_half_angle(
                -0.5 * np.arctan2(p_x[profile.j_max], p_y[profile.j_max])
            )
            etas = (*([0.0] * (profile.j_max - 1)), eta_top, eta_last)
            block_phase_list = _conjugation_chain_phase_list(etas, n=n)
        else:
            # Degree 0 is the special case in the paper: it needs two short chains
            # rather than one longer chain.
            n = max(0, int(ceil(norm / pi - 0.5)))
            eta_base = _normalize_half_angle(-0.5 * np.arctan2(p_x[0], p_y[0]))
            delta_eta = float(
                0.5 * np.arccos(np.clip(norm / (2.0 * pi * (n + 0.5)), 0.0, 1.0))
            )
            first_copy = _conjugation_chain_phase_list((eta_base + delta_eta,), n=n)
            second_copy = _conjugation_chain_phase_list((eta_base - delta_eta,), n=n)
            etas = (eta_base + delta_eta, eta_base - delta_eta)
            block_phase_list = concatenate_qsp_phase_lists(first_copy, second_copy)

        recovery_phase_list = concatenate_qsp_phase_lists(recovery_phase_list, block_phase_list)
        iterations.append(
            DegreewiseRecoveryIteration(
                iteration=iteration_index,
                j_max=profile.j_max,
                p_x=profile.p_x,
                p_y=profile.p_y,
                n=n,
                etas=tuple(float(value) for value in etas),
                block_phase_list=tuple(float(value) for value in block_phase_list),
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


@lru_cache(maxsize=None)
def _degreewise_recovery_result_cached(
    phases: tuple[float, ...],
    eps_probe: float,
    tol: float,
    tol_relative: float,
    theta_grid: tuple[float, ...] | None,
) -> DegreewiseRecoveryResult:
    return _degreewise_recovery_result_uncached(phases, eps_probe, tol, tol_relative, theta_grid)


def recovery_phase_list_k1_degreewise(
    phases: Sequence[float],
    *,
    eps_probe: float = 1e-7,
    tol: float = 1e-5,
    tol_relative: float = 1e-4,
    theta_grid: Sequence[float] | None = None,
) -> DegreewiseRecoveryResult:
    """Build the recovery phase list using the degree-wise method."""

    phases_tuple = tuple(float(phase) for phase in phases)
    theta_grid_tuple = None if theta_grid is None else tuple(float(theta) for theta in theta_grid)
    return _degreewise_recovery_result_cached(
        phases_tuple,
        float(eps_probe),
        float(tol),
        float(tol_relative),
        theta_grid_tuple,
    )


def build_corrected_qsp_phase_list(
    phases: Sequence[float],
    *,
    append_recovery_on_right: bool = True,
    eps_probe: float = 1e-7,
    tol: float = 1e-5,
    tol_relative: float = 1e-4,
    theta_grid: Sequence[float] | None = None,
) -> tuple[float, ...]:
    """Return one final phase list for the full `QSP + recovery` circuit.

    This is probably the most useful public helper in the file:

    - first build the degree-wise recovery phase list
    - then merge it with the original QSP phase list

    The result is a single phase sequence you can pass straight into
    `build_qsp_unitary(...)`.
    """

    base_phases = tuple(float(phase) for phase in phases)
    recovery_result = recovery_phase_list_k1_degreewise(
        base_phases,
        eps_probe=eps_probe,
        tol=tol,
        tol_relative=tol_relative,
        theta_grid=theta_grid,
    )

    if append_recovery_on_right:
        return concatenate_qsp_phase_lists(base_phases, recovery_result.recovery_phase_list)
    return concatenate_qsp_phase_lists(recovery_result.recovery_phase_list, base_phases)


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
