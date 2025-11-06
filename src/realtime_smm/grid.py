from __future__ import annotations

import copy
import pickle
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Sequence, Tuple, Union

import networkx as nx
import numpy as np

from .helpers import TaskSpace, AxisParams, Robot, GridParams, WorkspaceSMMs, SMMSolverParams
from .helpers.store import SMMStore, MemorySMMStore

Axis = TaskSpace
Position = Tuple[float, ...]
IndexTuple = Tuple[int, ...]


SO3_K_DEFAULT = 8
SO3_SEED_DEFAULT = 12345


class Grid:
    """Orthogonally-connected N-dimensional grid backed by a networkx.Graph.

    Input API
    ---------
    - taskspace: TaskSpace flags selecting axes (e.g., TaskSpace.X | TaskSpace.Y | TaskSpace.SO2_Z)
    - axis_params: list of AxisParams entries (each includes its TaskSpace axis)

    Notes
    -----
    - SO2_* axes are periodic; edges wrap around.
    - SO3 density uses resolution as a density proxy; neighbors are kNN by geodesic angle.
    - Nodes store only T (4x4) and index.
    """

    def __init__(
        self,
        taskspace: TaskSpace,
        axis_params: list[AxisParams],
        smm_store: Optional[SMMStore] = None,
        *,
        so3_k: int = SO3_K_DEFAULT,
        so3_seed: int = SO3_SEED_DEFAULT,
    ) -> None:
        # Validate SO2 uniqueness and incompatibility with SO3
        so2_flags = [TaskSpace.SO2_X in taskspace, TaskSpace.SO2_Y in taskspace, TaskSpace.SO2_Z in taskspace]
        if sum(so2_flags) > 1:
            raise ValueError("Only one of SO2_X, SO2_Y, SO2_Z may be set in TaskSpace for grid construction")
        if TaskSpace.SO3 in taskspace and any(so2_flags):
            raise ValueError("SO3 may not be combined with SO2_* for grid construction")

        for ax_param in axis_params:
            if ax_param.axis not in taskspace:
                raise ValueError(f"AxisParams for '{ax_param.axis.name}' not in taskspace")

        # Determine ordered axes from flags
        axes: List[TaskSpace] = []
        if TaskSpace.X in taskspace:
            axes.append(TaskSpace.X)
        if TaskSpace.Y in taskspace:
            axes.append(TaskSpace.Y)
        if TaskSpace.Z in taskspace:
            axes.append(TaskSpace.Z)
        if TaskSpace.SO2_X in taskspace:
            axes.append(TaskSpace.SO2_X)
        if TaskSpace.SO2_Y in taskspace:
            axes.append(TaskSpace.SO2_Y)
        if TaskSpace.SO2_Z in taskspace:
            axes.append(TaskSpace.SO2_Z)
        if TaskSpace.SO3 in taskspace:
            axes.append(TaskSpace.SO3)
        if not axes:
            raise ValueError("taskspace must include at least one axis")

        # Build lookup of params by axis and validate presence
        params_by_axis: Dict[TaskSpace, AxisParams] = {ap.axis: ap for ap in axis_params}
        for flag in axes:
            if flag not in params_by_axis:
                raise ValueError(f"missing AxisParams for '{flag.name}'")

        self.axes: Tuple[Axis, ...] = tuple(axes)
        self.axis_to_index: Dict[Axis, int] = {a: i for i, a in enumerate(self.axes)}

        # Store bounds/resolution/counts per axis
        bounds_list: List[Tuple[float, float]] = []
        res_list: List[float] = []
        counts_list: List[int] = []
        for a in self.axes:
            ap = params_by_axis[a]
            r = float(ap.resolution)
            if a in (TaskSpace.X, TaskSpace.Y, TaskSpace.Z):
                lo = float(ap.lower if ap.lower is not None else -1.0)
                hi = float(ap.upper if ap.upper is not None else 1.0)
                bounds_list.append((lo, hi))
                res_list.append(r)
                counts_list.append(self._derive_count(lo, hi, r))
            elif a in (TaskSpace.SO2_X, TaskSpace.SO2_Y, TaskSpace.SO2_Z):
                lo = float(ap.lower if ap.lower is not None else -np.pi)
                hi = float(ap.upper if ap.upper is not None else np.pi)
                bounds_list.append((lo, hi))
                res_list.append(r)
                counts_list.append(self._derive_count(lo, hi, r))
            elif a == TaskSpace.SO3:
                # Bounds unused for SO3 coordinate index
                bounds_list.append((0.0, 1.0))
                res_list.append(r)
                counts_list.append(self._derive_so3_count(r))
            else:
                raise ValueError(f"Unsupported axis: {a}")

        self.bounds = tuple(bounds_list)
        self.resolution = tuple(res_list)
        self.counts = tuple(int(c) for c in counts_list)

        # Coordinates along each axis
        self.axis_coords: Tuple[np.ndarray, ...] = tuple(
            (
                np.arange(self.counts[i], dtype=float)
                if self.axes[i] == TaskSpace.SO3
                else self._coords_via_resolution(self.bounds[i][0], self.resolution[i], self.counts[i], hi=self.bounds[i][1])
            )
            for i in range(len(self.axes))
        )

        # If SO3 used, pre-sample rotations and kNN neighbors (once)
        self.so3_rotations: Optional[List[np.ndarray]] = None
        self.so3_neighbors: Optional[List[List[int]]] = None
        self.so3_k: int = int(so3_k)
        self.so3_seed: int = int(so3_seed)
        if TaskSpace.SO3 in self.axis_to_index:
            so3_idx = self.axis_to_index[TaskSpace.SO3]
            so3_count = int(self.counts[so3_idx])
            self.so3_rotations = self._sample_uniform_so3(so3_count, seed=self.so3_seed)
            self.so3_neighbors = self._build_so3_knn(self.so3_rotations, k=self.so3_k)

        # Build graph with nodes and edges
        self.graph: nx.Graph = nx.Graph()
        self._build_nodes()
        self._build_edges()
        
        if smm_store is not None:
            self.store = smm_store
        else:
            self.store = MemorySMMStore()

        self.use_xy_halfplane: bool = False

    @classmethod
    def from_robot(
        cls,
        robot: Robot,
        grid_params: GridParams,
        smm_solver_params: SMMSolverParams | None = None,
        *,
        use_cache: bool = False,
    ) -> Grid:
        limits = robot.workspace_limits()
        axis_params = []
        for axis, (lo, hi) in limits.items():
            if grid_params.use_xy_halfplane:
                if axis == TaskSpace.X:
                    lo = 0.0
                elif axis == TaskSpace.Y and grid_params.use_xy_halfplane:
                    lo = 0.0
                    hi = 0.0
            axis_params.append(AxisParams(axis=axis, lower=lo, upper=hi, resolution=grid_params.pos_resolution))
        
        if TaskSpace.SO2_X in robot._taskspace:
            axis_params.append(AxisParams(axis=TaskSpace.SO2_X, lower=-np.pi, upper=np.pi, resolution=grid_params.orn_resolution))
        elif TaskSpace.SO2_Y in robot._taskspace:
            axis_params.append(AxisParams(axis=TaskSpace.SO2_Y, lower=-np.pi, upper=np.pi, resolution=grid_params.orn_resolution))
        elif TaskSpace.SO2_Z in robot._taskspace:
            axis_params.append(AxisParams(axis=TaskSpace.SO2_Z, lower=-np.pi, upper=np.pi, resolution=grid_params.orn_resolution))
        elif TaskSpace.SO3 in robot._taskspace:
            axis_params.append(AxisParams(axis=TaskSpace.SO3, resolution=grid_params.orn_resolution))

        if use_cache:
            smm_store = SMMStore.build_for(robot, grid_params, smm_solver_params if smm_solver_params is not None else SMMSolverParams())
        else:
            smm_store = MemorySMMStore()
        grid = cls(robot._taskspace, axis_params, smm_store=smm_store)
        grid.use_xy_halfplane = grid_params.use_xy_halfplane

        return grid


    # ------------------------ Public API ------------------------
    def to_networkx(self) -> nx.Graph:
        return copy.deepcopy(self.graph)

    @classmethod
    def from_graph(cls, graph: nx.Graph, taskspace: TaskSpace, axis_params: list[AxisParams], *, so3_k: int = SO3_K_DEFAULT, so3_seed: int = SO3_SEED_DEFAULT) -> "Grid":
        """Construct a Grid directly from an existing NetworkX graph and params.

        The provided graph is used as-is; axis parameters are kept for metadata
        and utility conversions.
        """
        grid = cls(taskspace, axis_params, so3_k=so3_k, so3_seed=so3_seed)
        grid.graph = graph
        return grid

    def save_graph(self, path: Path) -> None:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        pickle.dump(self.graph, open(str(p), "wb"))

    @staticmethod
    def load_graph(path: Path) -> nx.Graph:
        return pickle.load(open(str(path), "rb"))

    def nodes(self, data: bool = False):
        return self.graph.nodes(data=data)

    def edges(self, data: bool = False):
        return self.graph.edges(data=data)

    def node_index(self, coord_index: IndexTuple) -> int:
        return self._index_tuple_to_flat(coord_index)

    def index_to_pos(self, coord_index: IndexTuple) -> Position:
        return tuple(self.axis_coords[i][idx] for i, idx in enumerate(coord_index))

    def neighbors_index(self, coord_index: IndexTuple) -> List[IndexTuple]:
        node_id = self.node_index(coord_index)
        return [self.graph.nodes[n]["index"] for n in self.graph.neighbors(node_id)]

    # ------------------------ Attribute utilities ------------------------
    def set_node_attribute(self, node: int, key: str, value: object, overwrite: bool = True) -> None:
        """Set a single attribute on a single node.

        Args:
            node: Node id (flat id) to set the attribute for.
            key: Attribute name.
            value: Attribute value.
            overwrite: If False, keeps existing value if key already present.
        """
        if node not in self.graph:
            raise KeyError(f"node {node} does not exist")
        if overwrite or key not in self.graph.nodes[node]:
            self.graph.nodes[node][key] = value

    def set_edge_attribute(self, u: int, v: int, key: str, value: object, overwrite: bool = True) -> None:
        """Set a single attribute on a single edge (u, v).

        Args:
            u: First endpoint.
            v: Second endpoint.
            key: Attribute name.
            value: Attribute value.
            overwrite: If False, keeps existing value if key already present.
        """
        if not self.graph.has_edge(u, v):
            raise KeyError(f"edge ({u}, {v}) does not exist")
        data = self.graph.edges[u, v]
        if overwrite or key not in data:
            data[key] = value

    # ------------------------ Store utilities ------------------------
    def set_node_smms(self, node: int, ws: WorkspaceSMMs) -> None:
        """Persist WorkspaceSMMs for a node via the attached store if present.

        Falls back to in-memory assignment when no store is attached.
        """
        if node not in self.graph:
            raise KeyError(f"node {node} does not exist")
        self.store.put(node, ws)

    def get_node_smms(self, node: int) -> Optional[WorkspaceSMMs]:
        """Retrieve WorkspaceSMMs for a node via the attached store if present."""
        if node not in self.graph:
            raise KeyError(f"node {node} does not exist")

        return self.store.get(node)

    # ------------------------ Internal helpers ------------------------
    @staticmethod
    def _coords_via_resolution(lo: float, res: float, count: int, hi: Optional[float] = None) -> np.ndarray:
        if count <= 0:
            return np.empty((0,), dtype=float)
        if hi is None:
            return lo + res * np.arange(count, dtype=float)
        mid = (lo + hi) / 2.0
        k = (count - 1) // 2
        offsets = np.arange(-k, k + 1, dtype=float) * res
        return mid + offsets

    @staticmethod
    def _derive_count(lo: float, hi: float, res: float) -> int:
        lo = float(lo); hi = float(hi); res = float(res)
        if not np.isfinite(lo) or not np.isfinite(hi) or not np.isfinite(res):
            raise ValueError("non-finite value in derive_count")
        if res <= 0:
            raise ValueError("resolution must be > 0")
        if hi == lo:
            return 1
        half_extent = (hi - lo) / 2.0
        k = int(np.ceil(half_extent / res))
        k = max(k, 1)
        return (2 * k) + 1

    @staticmethod
    def _normalize_resolution(axes: Sequence[Axis], resolution: Union[float, Mapping[Axis, float]]) -> Dict[Axis, float]:
        if isinstance(resolution, (int, float)):
            r = float(resolution)
            return {a: r for a in axes}
        return {a: float(resolution[a]) for a in axes}

    @staticmethod
    def _derive_so3_count(density: float) -> int:
        return max(1, int(np.ceil((4.0 * np.pi) / max(1e-12, float(density) ** 2))))

    def _build_nodes(self) -> None:
        sizes = [int(c) for c in self.counts]
        self.sizes: Tuple[int, ...] = tuple(sizes)
        self.ndim: int = len(self.sizes)
        self.strides: Tuple[int, ...] = self._compute_strides(self.sizes)

        total_nodes = int(np.prod(self.sizes))
        for flat_id in range(total_nodes):
            idx_tuple = self._flat_to_index_tuple(flat_id)

            # Orientation from either SO2 angle or SO3 sample index
            R = np.eye(3, dtype=float)
            if (TaskSpace.SO3 in self.axis_to_index) and (self.so3_rotations is not None):
                so3_i = idx_tuple[self.axis_to_index[TaskSpace.SO3]]
                R = self.so3_rotations[int(so3_i)]
            else:
                if TaskSpace.SO2_X in self.axis_to_index:
                    a = float(self.axis_coords[self.axis_to_index[TaskSpace.SO2_X]][idx_tuple[self.axis_to_index[TaskSpace.SO2_X]]])
                    R = self._rotation_matrix_from_xyz_angles(a, 0.0, 0.0)
                elif TaskSpace.SO2_Y in self.axis_to_index:
                    a = float(self.axis_coords[self.axis_to_index[TaskSpace.SO2_Y]][idx_tuple[self.axis_to_index[TaskSpace.SO2_Y]]])
                    R = self._rotation_matrix_from_xyz_angles(0.0, a, 0.0)
                elif TaskSpace.SO2_Z in self.axis_to_index:
                    a = float(self.axis_coords[self.axis_to_index[TaskSpace.SO2_Z]][idx_tuple[self.axis_to_index[TaskSpace.SO2_Z]]])
                    R = self._rotation_matrix_from_xyz_angles(0.0, 0.0, a)

            T = np.eye(4, dtype=float)
            T[:3, :3] = R
            p = np.zeros(3, dtype=float)
            if TaskSpace.X in self.axis_to_index:
                ix = self.axis_to_index[TaskSpace.X]
                p[0] = float(self.axis_coords[ix][idx_tuple[ix]])
            if TaskSpace.Y in self.axis_to_index:
                iy = self.axis_to_index[TaskSpace.Y]
                p[1] = float(self.axis_coords[iy][idx_tuple[iy]])
            if TaskSpace.Z in self.axis_to_index:
                iz = self.axis_to_index[TaskSpace.Z]
                p[2] = float(self.axis_coords[iz][idx_tuple[iz]])
            T[:3, 3] = p

            self.graph.add_node(flat_id, index=idx_tuple, T=T)

    def _build_edges(self) -> None:
        total_nodes = int(np.prod(self.sizes))
        for flat_id in range(total_nodes):
            idx_tuple = self._flat_to_index_tuple(flat_id)
            for axis in range(self.ndim):
                axis_flag = self.axes[axis]
                if axis_flag == TaskSpace.SO3 and getattr(self, 'so3_neighbors', None) is not None:
                    orient_i = idx_tuple[axis]
                    for orient_j in self.so3_neighbors[orient_i]:
                        neighbor_idx = list(idx_tuple)
                        neighbor_idx[axis] = orient_j
                        neighbor_flat = self._index_tuple_to_flat(tuple(neighbor_idx))
                        self.graph.add_edge(flat_id, neighbor_flat)
                    continue
                if idx_tuple[axis] + 1 < self.sizes[axis]:
                    neighbor_idx = list(idx_tuple)
                    neighbor_idx[axis] += 1
                    neighbor_flat = self._index_tuple_to_flat(tuple(neighbor_idx))
                    self.graph.add_edge(flat_id, neighbor_flat)
                elif (axis_flag in (TaskSpace.SO2_X, TaskSpace.SO2_Y, TaskSpace.SO2_Z)) and self.sizes[axis] > 1:
                    neighbor_idx = list(idx_tuple)
                    neighbor_idx[axis] = 0
                    neighbor_flat = self._index_tuple_to_flat(tuple(neighbor_idx))
                    self.graph.add_edge(flat_id, neighbor_flat)

    @staticmethod
    def _compute_strides(sizes: Sequence[int]) -> Tuple[int, ...]:
        strides: List[int] = [1] * len(sizes)
        for i in range(len(sizes) - 2, -1, -1):
            strides[i] = strides[i + 1] * int(sizes[i + 1])
        return tuple(strides)

    def _index_tuple_to_flat(self, idx: IndexTuple) -> int:
        flat = 0
        for i, v in enumerate(idx):
            flat += int(v) * self.strides[i]
        return int(flat)

    def _flat_to_index_tuple(self, flat: int) -> IndexTuple:
        idx: List[int] = [0] * self.ndim
        remainder = int(flat)
        for i in range(self.ndim):
            stride = self.strides[i]
            size = self.sizes[i]
            idx[i] = remainder // stride
            remainder = remainder % stride
            if idx[i] >= size:
                idx[i] = size - 1
        return tuple(int(v) for v in idx)

    @staticmethod
    def _rotation_matrix_from_xyz_angles(theta_x: float, theta_y: float, theta_z: float) -> np.ndarray:
        cx, sx = np.cos(theta_x), np.sin(theta_x)
        cy, sy = np.cos(theta_y), np.sin(theta_y)
        cz, sz = np.cos(theta_z), np.sin(theta_z)
        Rx = np.array([[1.0, 0.0, 0.0], [0.0, cx, -sx], [0.0, sx, cx]], dtype=float)
        Ry = np.array([[cy, 0.0, sy], [0.0, 1.0, 0.0], [-sy, 0.0, cy]], dtype=float)
        Rz = np.array([[cz, -sz, 0.0], [sz, cz, 0.0], [0.0, 0.0, 1.0]], dtype=float)
        return Rz @ Ry @ Rx

    @staticmethod
    def _rotation_vector_from_matrix(R: np.ndarray) -> np.ndarray:
        tr = np.trace(R)
        cos_theta = (tr - 1.0) / 2.0
        cos_theta = np.clip(cos_theta, -1.0, 1.0)
        theta = float(np.arccos(cos_theta))
        if np.isclose(theta, 0.0):
            return 0.5 * np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]], dtype=float)
        else:
            return (
                theta
                / (2.0 * np.sin(theta))
                * np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]])
            )

    @staticmethod
    def _sample_uniform_so3(count: int, seed: int = 12345) -> List[np.ndarray]:
        rng = np.random.default_rng(seed)
        rotations: List[np.ndarray] = []
        for _ in range(int(count)):
            u1 = rng.random()
            u2 = rng.random() * 2.0 * np.pi
            u3 = rng.random() * 2.0 * np.pi
            s1 = np.sqrt(1.0 - u1)
            s2 = np.sqrt(u1)
            qw = s2 * np.cos(u3)
            qx = s1 * np.sin(u2)
            qy = s1 * np.cos(u2)
            qz = s2 * np.sin(u3)
            qx2, qy2, qz2 = qx * qx, qy * qy, qz * qz
            R = np.array([
                [1 - 2 * (qy2 + qz2), 2 * (qx * qy - qz * qw), 2 * (qx * qz + qy * qw)],
                [2 * (qx * qy + qz * qw), 1 - 2 * (qx2 + qz2), 2 * (qy * qz - qx * qw)],
                [2 * (qx * qz - qy * qw), 2 * (qy * qz + qx * qw), 1 - 2 * (qx2 + qy2)],
            ], dtype=float)
            rotations.append(R)
        return rotations

    @staticmethod
    def _so3_geodesic_angle(R1: np.ndarray, R2: np.ndarray) -> float:
        RtR = R1.T @ R2
        tr = np.trace(RtR)
        cos_theta = (tr - 1.0) / 2.0
        cos_theta = np.clip(cos_theta, -1.0, 1.0)
        return float(np.abs(np.arccos(cos_theta)))

    def _build_so3_knn(self, rotations: List[np.ndarray], k: int) -> List[List[int]]:
        n = len(rotations)
        k = max(1, min(int(k), max(1, n - 1)))
        neighbors: List[List[int]] = [[] for _ in range(n)]
        dists = np.zeros((n, n), dtype=float)
        for i in range(n):
            dists[i, i] = np.inf
            for j in range(i + 1, n):
                d = self._so3_geodesic_angle(rotations[i], rotations[j])
                dists[i, j] = d
                dists[j, i] = d
        for i in range(n):
            order = np.argsort(dists[i, :], kind="mergesort")
            neighbors[i] = [int(idx) for idx in order[:k]]
        return neighbors



if __name__ == "__main__":
    # Basic smoke test using TaskSpace + AxisParams (enum keys)
    ts = TaskSpace.X | TaskSpace.Y
    axis_params = [
        AxisParams(axis=TaskSpace.X, lower=0.0, upper=2.0, resolution=1.0),
        AxisParams(axis=TaskSpace.Y, lower=0.0, upper=2.0, resolution=1.0),
    ]

    grid = Grid(ts, axis_params)
    G = grid.to_networkx()

    # With resolution 1.0 over [0,2], we expect positions at 0,1,2 on each axis.
    # 3x3 = 9 nodes; edges: (3-1)*3 + (3-1)*3 = 12
    assert G.number_of_nodes() == 9
    assert G.number_of_edges() == 12

    # Center has 4 neighbors
    center_idx = (1, 1)
    center_node = grid.node_index(center_idx)
    assert G.degree[center_node] == 4
    assert grid.index_to_pos(center_idx) == (1.0, 1.0)

    # Corners have degree 2
    for idx in [(0, 0), (0, 2), (2, 0), (2, 2)]:
        n = grid.node_index(idx)
        assert G.degree[n] == 2

    # Exercise nodes()/edges() accessors
    assert len(list(grid.nodes())) == 9
    assert len(list(grid.edges())) == 12
    # Node data contains index and T
    some_node, attrs = next(iter(grid.nodes(data=True)))
    assert "index" in attrs and "T" in attrs
    # T should have identity rotation and correct translation for (1,1)
    T_center = G.nodes[center_node]["T"]
    assert np.allclose(T_center[:3, :3], np.eye(3))
    assert np.allclose(T_center[:3, 3], np.array([1.0, 1.0, 0.0]))

    print("Grid __main__ smoke test passed.")

    # One-to-one attribute setters
    n0 = grid.node_index((0, 1))
    grid.set_node_attribute(n0, "marker", "A")
    assert grid.graph.nodes[n0]["marker"] == "A"

    # Pick an existing edge and set a one-to-one attribute
    some_edge = next(iter(G.edges))
    grid.set_edge_attribute(some_edge[0], some_edge[1], "tag", "E1")
    assert grid.graph.edges[some_edge]["tag"] == "E1"

    print("One-to-one attribute setters smoke test passed.")

    # SO2+X smoke test: include X positions and theta_z orientations
    ts2 = TaskSpace.X | TaskSpace.SO2_X
    ap2 = [
        AxisParams(axis=TaskSpace.X, lower=0.0, upper=1.0, resolution=0.3),
        AxisParams(axis=TaskSpace.SO2_X, lower=-np.pi, upper=np.pi, resolution=0.8),
    ]
    grid2 = Grid(ts2, ap2)
    G2 = grid2.to_networkx()

    # Every node should have degree 2 along theta and 1 along x interior; corners adjust naturally
    print("SO2+X grid nodes:", G2.number_of_nodes(), "edges:", G2.number_of_edges())

    # Plot grid2 in (x, theta_z) space if matplotlib is available
    try:
        import matplotlib.pyplot as plt
        # Position mapping: x from T translation; theta from R (atan2(R[1,0], R[0,0]))
        def node_coords(nid: int) -> Tuple[float, float]:
            Tn = grid2.graph.nodes[nid]["T"]
            x = float(Tn[0, 3]) + np.random.randn() * 0.05
            R = Tn[:3, :3]
            theta = float(np.arctan2(R[1, 0], R[0, 0]))
            return x, theta

        pos = {n: node_coords(n) for n in G2.nodes}
        plt.figure()
        nx.draw(G2, pos=pos, node_size=200, with_labels=True)
        plt.xlabel("x")
        plt.ylabel("theta_x (rad)")
        plt.title("Grid in (x, theta_z)")
        plt.show()
    except Exception as e:
        print("Matplotlib not available or failed to plot:", str(e))

    # XY + SO3 3D orientation field smoke test
    ts3 = TaskSpace.X | TaskSpace.Y | TaskSpace.SO3
    ap3 = [
        AxisParams(axis=TaskSpace.X, lower=-1.0, upper=1.0, resolution=0.1),
        AxisParams(axis=TaskSpace.Y, lower=-1.0, upper=1.0, resolution=0.1),
        AxisParams(axis=TaskSpace.SO3, resolution=1.0),
    ]
    grid3 = Grid(ts3, ap3)
    G3 = grid3.to_networkx()

    try:
        import matplotlib.pyplot as plt
        from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

        fig = plt.figure()
        ax = fig.add_subplot(111, projection="3d")

        # Subsample for plotting if grid is too large
        nodes = list(G3.nodes)
        max_plot = 3000
        if len(nodes) > max_plot:
            rng = np.random.default_rng(42)
            nodes = list(rng.choice(nodes, size=max_plot, replace=False))

        axis_len = 0.1
        xs, ys, zs = [], [], []
        for n in nodes:
            Tn = G3.nodes[n]["T"]
            p = Tn[:3, 3]
            R = Tn[:3, :3]
            xs.append(p[0]); ys.append(p[1]); zs.append(p[2])
            # Draw local axes from position
            ends = [p + axis_len * R[:, 0], p + axis_len * R[:, 1], p + axis_len * R[:, 2]]
            cols = ["r", "g", "b"]
            for end, c in zip(ends, cols):
                ax.plot([p[0], end[0]], [p[1], end[1]], [p[2], end[2]], color=c, linewidth=0.5)

        ax.scatter(xs, ys, zs, s=2, c="k", alpha=0.3)
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_zlabel("z")
        ax.set_title("XY+SO3 grid: orientations as RGB axes")
        ax.set_zlim(-1.0, 1.0)
        plt.show()
    except Exception as e:
        print("Matplotlib 3D plot failed:", str(e))

