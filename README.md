realtime-smm
============

This project is the official codebase for the T-RO paper [A Learning-Based Method for Computing Self-Motion Manifolds of Redundant Robots for Real-Time Fault-Tolerant Motion Planning](https://ieeexplore.ieee.org/abstract/document/10959723).

`realtime-smm` is a learning-based toolkit for computing self-motion manifolds (SMMs) of redundant robots fast enough for realtime global motion planning.

Overview
--------
- Cache-focused data pipeline that stores raw and post-processed SMM samples during grid construction.
- Per-cluster neural regressors trained on Fourier-series targets with optional low-pass filtering and XY half-plane canonicalisation.
- Bundled inference model (`SMMNetworkBundle`) that selects the correct regressor with a classifier, reconstructs joint trajectories, and can upsample output resolutions.
- Optional basin-hopping pretraining (via `pytorch-minimize`) before standard PyTorch optimisers to help escape poor initialisations.
- Example notebook-style script (`examples/smms_3r.py`) that trains, loads, and visualises overlays against analytical solutions.

Features
--------
- **Caching:** Uses `platformdirs` to create per-run cache folders that persist raw grids, post-processed clusters, and trained models.
- **Cluster-centric training:** Each connected workspace cluster yields its own neural network while a classifier predicts the correct cluster at inference time.
- **Fourier targets:** SMMs are transformed to Fourier space, optionally truncated with `fft_cutoff`, and reconstructed via analytic IFFT for smooth representation.
- **XY half-plane canonicalisation:** Input transforms are rotated into a canonical frame, with the orientation offset reapplied after prediction.
- **Upsampling:** `SMMNetworkBundle.__call__` supports `samples=` to evaluate the learned Fourier representation at arbitrary resolutions.
- **Progress feedback:** Training loops use `tqdm` progress bars when available and log final losses per cluster.

Installation
------------
```bash
pip install realtime-smm
```

To include the optional global optimiser support (Basin Hopping via `pytorch-minimize`):

```bash
pip install realtime-smm[optional]
```

Using Poetry:

```bash
poetry install        # core dependencies
poetry install -E optional  # include pytorch-minimize from git
```

Quickstart
----------
`realtime-smm` exposes a narrow public surface in `realtime_smm.main`:

```python
from realtime_smm import (
    Robot,
    GridParams,
    SMMSolverParams,
    TrainingConfig,
    training_pipeline,
    load_trained_bundle,
)

robot = Robot([...])  # Define DH links
grid_params = GridParams(pos_resolution=0.01, use_xy_halfplane=True)
smm_params = SMMSolverParams(samples=64)
config = TrainingConfig(epochs=1000, fft_cutoff=20)

bundle = training_pipeline(
    robot,
    grid_params=grid_params,
    smm_params=smm_params,
    training_config=config,
    name="my_robot_run",
)

bundle = load_trained_bundle(name="my_robot_run")
T = np.eye(4)
ws_smms = bundle(T, samples=256)  # Predict upsampled SMMs
```

Only call the top-level helpers from `main.py` and adjust behaviour through the configuration dataclasses re-exported in `__init__.py`.

Cache Management
----------------
You can clear cached runs when experimenting:

```python
from realtime_smm import clear_cache

clear_cache(name="my_robot_run")  # Remove a single run
clear_cache()  # Prompt before clearing every cache directory
```

Examples
--------
`examples/smms_3r.py` demonstrates the full workflow, including plotting predicted vs. analytical SMM projections. Run it after installation to sanity-check dependencies and visualise results.

License
-------
MIT. See `LICENSE` for details.
