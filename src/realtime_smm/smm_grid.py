from __future__ import annotations

from typing import List

import numpy as np
import multiprocessing as mp

from .grid import Grid
from .helpers import Robot, TaskSpace, SMMSolverParams, DHLink, JointType, SMMStatus, GridParams, SMMStore, NodeStage

"""Compute SMMs for each node transform in a Grid."""


_POOL_ROBOT: Robot | None = None
_POOL_PARAMS: SMMSolverParams | None = None


def _pool_init(robot: Robot, smm_solver_params: SMMSolverParams) -> None:
    global _POOL_ROBOT, _POOL_PARAMS
    _POOL_ROBOT = robot
    _POOL_PARAMS = smm_solver_params


def _solve_smm_for_T(T_des: np.ndarray):
    assert _POOL_ROBOT is not None and _POOL_PARAMS is not None
    return _POOL_ROBOT.workspace_smms(
        T_des,
        samples=_POOL_PARAMS.samples,
        step=_POOL_PARAMS.step,
        sing_thresh=_POOL_PARAMS.sing_thresh,
        smm_iters=_POOL_PARAMS.smm_iters,
    )


def build_smm_grid(
    robot: Robot,
    grid_params: GridParams = GridParams(),
    smm_solver_params: SMMSolverParams = SMMSolverParams(),
    *,
    use_cache: bool = False,
) -> Grid:
    """Compute SMMs at each grid node transform and store on nodes.

    For each node with attribute 'T' (4x4 desired transform), solves SMMs and
    attaches one of the following attributes per standardized access contract:
      - 'smm_data': in-memory WorkspaceSMMs result (when not using cache)
      - 'smm_store_key': a cache handle indicating data persisted in the store

    This mutates the returned Grid's underlying graph in-place.
    """

    grid = Grid.from_robot(robot, grid_params, smm_solver_params, use_cache=use_cache)

    # Collect node ids and desired transforms
    node_ids: List[int] = []
    transforms: List[np.ndarray] = []
    for n, attrs in grid.nodes(data=True):
        T_des = attrs.get("T")
        if T_des is None:
            raise KeyError("grid node missing 'T' attribute")
        node_ids.append(n)
        transforms.append(T_des)

    # Persist run metadata only for disk-backed runs
    if use_cache:
        grid.store.write_run_metadata(robot, grid_params, smm_solver_params)
        from pathlib import Path
        graph_path = grid.store.run_dir / "graph.pkl"  # type: ignore[attr-defined]
        if Path(graph_path).exists():
            grid.graph = Grid.load_graph(Path(graph_path))

    # Initialize node processing status
    for n in node_ids:
        grid.set_node_attribute(n, "status", NodeStage.WAITING_FOR_SMMS, overwrite=False)

    # Parallel solve using process pool (robot and params passed once to initializer)
    with mp.Pool(initializer=_pool_init, processes=mp.cpu_count(), initargs=(robot, smm_solver_params)) as pool:
        # Determine which nodes still need compute
        pending_pairs = [(n, T) for n, T in zip(node_ids, transforms) if not grid.store.has(n)]

        if pending_pairs:
            pend_ids, pend_Ts = zip(*pending_pairs)
            iterator = pool.imap(_solve_smm_for_T, pend_Ts, chunksize=16)
            try:
                from tqdm import tqdm  # type: ignore
                iterator = tqdm(iterator, total=len(pend_Ts), desc="Solving SMMs", smoothing=0.1)
            except Exception:
                pass
            for n, wk_smms in zip(pend_ids, iterator):
                grid.set_node_smms(n, wk_smms)

    for n in node_ids:
        if grid.graph.nodes[n].get("status") < NodeStage.COMPUTED_SMMS:
            grid.graph.nodes[n]["status"] = NodeStage.COMPUTED_SMMS

    # Persist the graph if caching is enabled via Grid API
    if use_cache:
        from pathlib import Path
        graph_path = grid.store.run_dir / "graph.pkl"  # type: ignore[attr-defined]
        grid.save_graph(Path(graph_path))

    return grid


if __name__ == "__main__":
    # Smoke test: build a grid using TaskSpace + AxisParams and compute SMMs
    ts = TaskSpace.X | TaskSpace.SO2_Z

    dh = [
        DHLink(a=0.20, alpha=0.0, d=0.0, theta=0.0, joint_kind=JointType.REVOLUTE),
        DHLink(a=0.25, alpha=0.0, d=0.0, theta=0.0, joint_kind=JointType.REVOLUTE),
        DHLink(a=0.35, alpha=0.0, d=0.0, theta=0.0, joint_kind=JointType.REVOLUTE),
    ]
    robot = Robot(dh, taskspace=ts)

    grid_params = GridParams(pos_resolution=0.05, orn_resolution=0.1, use_xy_halfplane=False)
    grid = build_smm_grid(robot, grid_params)

    for e in grid.edges():
        u, v = e
        smm_u = grid.graph.nodes[u].get("smm_data")
        smm_v = grid.graph.nodes[v].get("smm_data")
        if smm_u.status != SMMStatus.OK or smm_v.status != SMMStatus.OK:
            grid.graph.remove_edge(u, v)
            continue

        if smm_u.branches != smm_v.branches:
            grid.graph.remove_edge(u, v)
            continue
