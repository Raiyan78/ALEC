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





