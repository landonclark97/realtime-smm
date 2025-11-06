import numpy as np
import networkx as nx
from scipy.optimize import linear_sum_assignment
from typing import Callable

from .grid import Grid
from .helpers import SMMStatus, SMM, WorkspaceSMMs, NodeStage


def _connected_subgraphs(graph: nx.Graph) -> list[nx.Graph]:
    """Return connected subgraphs as standalone graphs with attributes/edges.

    - only_ok: if True, include only nodes whose 'smm_data.status' is OK.
    """
    H = graph

    subgraphs = []
    for comp in nx.connected_components(H):
        sub = H.subgraph(comp).copy()
        subgraphs.append(sub)
    return subgraphs


def _deformation(smm_a: SMM, smm_b: SMM) -> float:
    """Return the deformation between two SMMs."""
    if smm_a.samples != smm_b.samples:
        raise ValueError("SMMs must have the same number of samples")
    smm_a_torus = smm_a.torus
    diff = 0.0
    for sample_b in smm_b.torus:
        diff += np.amin(np.linalg.norm(np.angle(smm_a_torus / sample_b), axis=1))

    return diff / smm_a.samples


def _deformation_matrix(smms_a: list[SMM], smms_b: list[SMM]) -> np.ndarray:
    """Return the deformation matrix between two lists of SMMs."""
    if len(smms_a) != len(smms_b):
        raise ValueError("SMM lists must have the same number of SMMs")
    D = np.zeros((len(smms_a), len(smms_b)))
    for i, smm_a in enumerate(smms_a):
        for j, smm_b in enumerate(smms_b):
            D[i, j] = _deformation(smm_a, smm_b)
    return D


def _align_smms(smms_a: list[SMM], smms_b: list[SMM]) -> list[SMM]:
    """Align two lists of SMMs."""
    deformation_matrix = _deformation_matrix(smms_a, smms_b)
    row_ind, col_ind = linear_sum_assignment(deformation_matrix)
    
    perm = np.empty_like(row_ind)
    perm[row_ind] = col_ind
    
    aligned_smms = [smms_b[i] for i in perm]
    return aligned_smms


def _align_component_smms(
    H: nx.Graph,
    fetch_ws: Callable[[int], WorkspaceSMMs | None],
    update_ws: Callable[[int, WorkspaceSMMs], None],
    subgraph_index: int = 0,
    num_subgraphs: int = 1,
) -> None:
    """Align SMM lists across a connected component using a BFS wavefront.

    For a seed node near the graph center, align each unprocessed neighbor's
    SMM list ordering to the current node, then push that neighbor to the
    processing queue. Mutates node "smm_data" (WorkspaceSMMs) in-place.
    """
    if H.number_of_nodes() == 0:
        return
    # Optional progress bar
    bar = None
    try:
        from tqdm import tqdm  # type: ignore
        bar = tqdm(total=H.number_of_nodes(), desc=f"Align ({subgraph_index + 1}/{num_subgraphs})", smoothing=0.1)
    except Exception:
        bar = None
    # Prefer already-aligned nodes as seeds; otherwise pick center
    aligned_nodes = [n for n in H.nodes if H.nodes[n].get("status") == NodeStage.ALIGNED_SMMS]
    if aligned_nodes:
        processed = set(int(n) for n in aligned_nodes)
        queue = [int(n) for n in aligned_nodes]
        if bar is not None:
            bar.update(len(processed))
    else:
        centers = nx.center(H)
        seed = centers[0]
        processed = set([seed])
        queue = [seed]
        if bar is not None:
            bar.update(1)

    while queue:
        u = queue.pop(0)
        u_ws = fetch_ws(u)
        if u_ws is None:
            continue
        u_smms: list[SMM] = u_ws.data
        for v in H.neighbors(u):
            if v in processed:
                continue
            v_ws = fetch_ws(v)
            if v_ws is None:
                continue
            v_smms: list[SMM] = v_ws.data
            aligned_v = _align_smms(u_smms, v_smms)
            ws_new = WorkspaceSMMs(status=SMMStatus.OK, data=aligned_v)
            update_ws(v, ws_new)
            processed.add(v)
            queue.append(v)
            if bar is not None:
                bar.update(1)
    if bar is not None:
        bar.close()


def postprocess_grid(grid: Grid, *, use_cache: bool = False) -> list[nx.Graph]:
    G = grid.to_networkx()

    to_remove = []
    for n in list(G.nodes()):
        smm = grid.get_node_smms(n)
        if smm is None or smm.status != SMMStatus.OK:
            to_remove.append(n)
    if to_remove:
        G.remove_nodes_from(to_remove)

    for e in G.edges():
        u, v = e
        smm_u = grid.get_node_smms(u)
        smm_v = grid.get_node_smms(v)

        if smm_u is None or smm_v is None:
            G.remove_edge(u, v)
            continue
        if smm_u.branches != smm_v.branches:
            G.remove_edge(u, v)
            continue

    subgraphs = _connected_subgraphs(G)

    # Prepare accessors that respect cache/in-memory and update policy
    def _fetch_ws(node_id: int) -> WorkspaceSMMs | None:
        return grid.get_node_smms(node_id)

    def _update_ws(node_id: int, ws: WorkspaceSMMs) -> None:
        # Persist via Grid API; it will use the attached store if present
        grid.set_node_smms(node_id, ws)
        if grid.graph.nodes[node_id].get("status") < NodeStage.ALIGNED_SMMS:
            grid.graph.nodes[node_id]["status"] = NodeStage.ALIGNED_SMMS

    # Align SMM order consistently within each connected subgraph
    for i, H in enumerate(subgraphs):
        for n, data in H.nodes(data=True):
            if data.get("status") < NodeStage.ALIGNED_SMMS:
                break
        else:
            continue
        _align_component_smms(H, _fetch_ws, _update_ws, subgraph_index=i, num_subgraphs=len(subgraphs))

        # Mark nodes as aligned
        for n in H.nodes:
            if grid.graph.nodes[n].get("status") < NodeStage.ALIGNED_SMMS:
                grid.graph.nodes[n]["status"] = NodeStage.ALIGNED_SMMS
        
        if use_cache:
            from pathlib import Path
            graph_path = grid.store.run_dir / "graph.pkl"  # type: ignore[attr-defined]
            grid.save_graph(Path(graph_path))

    return subgraphs

if __name__ == "__main__":
    from .helpers import TaskSpace, DHLink, JointType, Robot, GridParams, SMMSolverParams
    from .smm_grid import build_smm_grid
    ts = TaskSpace.X | TaskSpace.SO2_Z

    dh = [
        DHLink(a=0.20, alpha=0.0, d=0.0, theta=0.0, joint_type=JointType.REVOLUTE),
        DHLink(a=0.25, alpha=0.0, d=0.0, theta=0.0, joint_type=JointType.REVOLUTE),
        DHLink(a=0.35, alpha=0.0, d=0.0, theta=0.0, joint_type=JointType.REVOLUTE),
    ]
    robot = Robot(dh, taskspace=ts)

    grid_params = GridParams(pos_resolution=0.05, orn_resolution=0.05, use_xy_halfplane=True)

    # # Non-cached path: compute in-memory and postprocess
    grid = build_smm_grid(robot, grid_params)
    Hs = postprocess_grid(grid)
    print("Postprocess (no cache) subgraphs:", len(Hs))

    # Cached path: compute with cache, then postprocess using cache-backed accessor
    smm_solver_params = SMMSolverParams()
    grid_cached = build_smm_grid(robot, grid_params, smm_solver_params, use_cache=True)
    Hs_cached = postprocess_grid(grid_cached, use_cache=True)
    print("Postprocess (cache) subgraphs:", len(Hs_cached))

    import matplotlib.pyplot as plt
    for H in Hs_cached:
        nx.draw(H, with_labels=True)
        plt.show()