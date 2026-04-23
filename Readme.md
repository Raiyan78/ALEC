# ALEC

Experimenting with noisy quantum signal processing (QSP) and the first-order degree-wise recovery construction from the ALEC paper(arXiv:2301.08542).

The implementation follows the pipeline:

- build a noisy QSP unitary from a phase list
- fit the first-order error profile
- build the `k = 1` degree-wise recovery phase list
- merge the original QSP sequence with the recovery sequence

Note: we implemented k=1 recovery, not the full k>1 ALEC algorithm. However, one can recursively apply k>1 recovery on the k=1 corrected sequence to get the full ALEC algorithm, however it is not recommended as the degreewise recovery has a tight bound of 2^k d^{2^k}.
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


