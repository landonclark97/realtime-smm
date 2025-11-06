"""High-level training / loading convenience helpers."""

from __future__ import annotations

from pathlib import Path
import shutil
from typing import Optional, Sequence

import networkx as nx

from .smm_grid import build_smm_grid
from .grid import Grid
from .helpers import GridParams, Robot, SMMSolverParams, TrainingConfig
from .postprocess_grid import postprocess_grid
from .learning import (
    SMMNetworkBundle,
    train_cluster_networks,
)


def training_pipeline(
    robot: Robot,
    *,
    grid_params: GridParams,
    smm_params: Optional[SMMSolverParams] = None,
    name: str,
    training_config: Optional[TrainingConfig] = None,
    use_cache: bool = True,
) -> SMMNetworkBundle:
    """Run grid build→postprocess→cluster training in a single call.

    Args:
        robot: Robot definition used to generate SMMs.
        grid_params: Workspace/grid configuration.
        smm_params: Optional solver parameters (defaults to SMMSolverParams()).
        name: Identifier for cached artifacts and bundle location.
        training_config: Optional override for network training hyperparameters.
        use_cache: When True the underlying SMM pipeline writes to disk cache.

    Returns:
        Trained `SMMNetworkBundle` ready for inference.
    """

    solver = smm_params if smm_params is not None else SMMSolverParams()
    grid: Grid = build_smm_grid(robot, grid_params, solver, use_cache=use_cache)

    clusters: Sequence[nx.Graph] = postprocess_grid(grid, use_cache=use_cache)

    bundle = train_cluster_networks(
        grid,
        clusters,
        name=name,
        config=training_config,
    )
    return bundle


def load_trained_bundle(
    name: str,
    *,
    base_dir: Optional[Path | str] = None,
    device: str = "cpu",
) -> SMMNetworkBundle:
    """Convenience loader that mirrors the training name convention."""

    return SMMNetworkBundle.load(name=name, base_dir=base_dir, device=device)


def clear_cache(name: str = "all", base_dir: Optional[Path | str] = None) -> None:
    """Clear cached artifacts.

    Args:
        name: Specific run directory to remove, or "all" to clear every cache
            entry.
        base_dir: Optional override for the cache root.
    """

    try:
        import platformdirs
    except ImportError:
        raise ImportError("platformdirs is required to clear the cache.")

    root = Path(base_dir) if base_dir is not None else Path(
        platformdirs.user_cache_dir(appname="realtime_smm", appauthor=False)
    )

    if not root.exists():
        return

    if name != "all":
        target = root / name
        if target.exists():
            if target.is_dir():
                shutil.rmtree(target, ignore_errors=True)
            else:
                target.unlink(missing_ok=True)  # type: ignore[arg-type]
        return

    resp = input("[WARNING] Are you sure you want to clear the cache for all realtime-smm runs? (y/N)")
    if resp.lower() != "y":
        return

    for entry in root.iterdir():
        try:
            if entry.is_dir():
                shutil.rmtree(entry, ignore_errors=True)
            else:
                entry.unlink(missing_ok=True)  # type: ignore[arg-type]
        except FileNotFoundError:
            continue
