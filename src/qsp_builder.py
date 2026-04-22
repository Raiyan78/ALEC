import numpy as np
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
