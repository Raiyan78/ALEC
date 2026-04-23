# ALEC

Experimenting with noisy quantum signal processing (QSP) and the first-order degree-wise recovery construction from the ALEC paper(arXiv:2301.08542).

The implementation follows the pipeline:

- build a noisy QSP unitary from a phase list
- fit the first-order error profile
- build the `k = 1` degree-wise recovery phase list
- merge the original QSP sequence with the recovery sequence

## What is implemented

### `src/qsp_builder.py`

Core QSP evaluator under the multiplicative phase-noise model

\[
U_\epsilon(\theta;\phi)
= e^{i\phi_0(1+\epsilon)Z}
\prod_{j=1}^d W(\theta)e^{i\phi_j(1+\epsilon)Z},
\qquad
W(\theta)=e^{i\theta X}.
\]

Functions:

- `phase_rotation_z(phi, epsilon)`
- `signal_operator(theta)`
- `build_qsp_unitary(phases, theta, epsilon)`

### `src/phase_list.py`

Phase-list algebra used by the recovery construction.

Functions:

- `concatenate_qsp_phase_lists(left, right)`
- `apply_conjugation_to_phase_list(phases, m, n, eta)`
- `_normalize_half_angle(angle)`
- `_conjugation_chain_phase_list(etas, n=...)`

Two points matter here:

1. QSP phase lists do not compose by plain concatenation.
The last phase of the left list and the first phase of the right list merge into one angle.

2. The recovery code works with phase lists symbolically first, and only later evaluates the resulting sequence as a matrix.

### `src/recover.py`

First-order (`k = 1`) degree-wise recovery.

Main public functions:

- `pauli_decomposition(matrix)`
- `corrected_error_operator_from_phase_list(phases, recovery_phase_list, theta, epsilon)`
- `fit_first_order_error_profile(...)`
- `recovery_phase_list_k1_degreewise(...)`
- `build_corrected_qsp_phase_list(phases, ...)`

The recovery loop is:

1. Probe the corrected circuit at small `epsilon`
2. Fit the first-order `X/Y` error profile in the even-power basis of `cos(theta)`
3. Find the highest surviving degree `j_max`
4. Add a recovery block that cancels that degree
5. Repeat until no degree remains above threshold

`build_corrected_qsp_phase_list(...)` returns one merged phase list for the full corrected sequence. It is not just the recovery block.

## What is not implemented

- true higher-order ALEC (`k > 1` in the paper)
- the shorthand Grover-specific branch from Remark S4.11
- paper-level diagnostics / plotting utilities
- a polished package interface

Important: recursively calling

```python
build_corrected_qsp_phase_list(build_corrected_qsp_phase_list(phases))
```

is only recursive first-order recovery. It is not the same thing as the paper's higher-order `k`.

## Repository layout

```text
ALEC/
├── Readme.md
├── main.ipynb
├── benchmark.ipynb
└── src/
    ├── qsp_builder.py
    ├── phase_list.py
    └── recover.py
```

## Basic usage

From the `ALEC/` directory:

```python
import numpy as np

from src.qsp_builder import build_qsp_unitary
from src.recover import build_corrected_qsp_phase_list

base_phases = [np.pi / 3] * 4
theta = 0.4
epsilon = 1e-3

corrected_phases = build_corrected_qsp_phase_list(base_phases)

ideal = build_qsp_unitary(base_phases, theta, 0.0)
noisy = build_qsp_unitary(base_phases, theta, epsilon)
corrected = build_qsp_unitary(corrected_phases, theta, epsilon)
```

If you want to sweep the figure-style x-axis `a^2`, use

\[
a = \cos(\theta), \qquad \theta = \arccos(\sqrt{a^2}).
\]

Example:

```python
a2 = 0.8
theta = np.arccos(np.sqrt(a2))
```

## Notes on the corrected phase list

The corrected list is a merged QSP sequence. That means the original base list does not necessarily appear as a literal prefix.

If

- `base = (a0, ..., ad)`
- `recovery = (r0, ..., rs)`

then the merged list is

- `(a0, ..., a{d-1}, ad + r0, r1, ..., rs)`

So it is normal for the last base phase to change after composition.

## Notebooks

- `main.ipynb`: local numerical experiments
- `benchmark.ipynb`: Qiskit / IBM runtime experiments

The Qiskit hardware side is separate from the core math code. The recovery construction itself only depends on NumPy.


