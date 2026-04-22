import numpy as np
from typing import Sequence
Matrix = np.ndarray
def phase_rotation_z(phi: float, epsilon: float = 0.0) -> Matrix:
    """Return e^{i phi (1 + epsilon) Z}."""

    noisy_phi = phi * (1.0 + epsilon)
    return np.array(
        [
            [np.exp(1j * noisy_phi), 0.0],
            [0.0, np.exp(-1j * noisy_phi)],
        ],
        dtype=complex,
    )


def signal_operator(theta: float) -> Matrix:
    """Return the noiseless signal operator W(theta) = e^{i theta X} from Eq. (1)."""

    cos_theta = np.cos(theta)
    sin_theta = np.sin(theta)
    return np.array(
        [
            [cos_theta, 1j * sin_theta],
            [1j * sin_theta, cos_theta],
        ],
        dtype=complex,
    )

def build_qsp_unitary(
    phases: Sequence[float],
    theta: float,
    epsilon: float = 0.0,
) -> Matrix:
    """Construct U_epsilon(theta; phi) = exp(i phi_0 (1 + epsilon) Z)
                               prod_{j=1}^d W(theta) exp(i phi_j (1 + epsilon) Z)
    """

    if not phases:
        raise ValueError("phases must contain at least one angle")

    unitary = phase_rotation_z(float(phases[0]), epsilon)
    w_theta = signal_operator(theta)
    for phase in phases[1:]:
        unitary = unitary @ w_theta @ phase_rotation_z(float(phase), epsilon)
    return unitary
