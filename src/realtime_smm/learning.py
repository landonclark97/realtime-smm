from __future__ import annotations

import os
from pathlib import Path
import platform
from typing import Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import networkx as nx
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from .grid import Grid
from .helpers import SMMStatus, WorkspaceSMMs, TaskSpace, SMM, SchedulerType, TrainingConfig, FFTTargetSpec, ClusterModelBundle
from .helpers.nn import MLP
from .helpers.store import ensure_dir, user_cache_dir

try:  # Optional progress bar
    from tqdm import tqdm
except Exception:  # pragma: no cover
    tqdm = None  # type: ignore


def _rotation_matrix_to_quaternion(R: np.ndarray) -> np.ndarray:
    R = np.asarray(R, dtype=np.float64)
    tr = float(R[0, 0] + R[1, 1] + R[2, 2])
    if tr > 0.0:
        S = np.sqrt(tr + 1.0) * 2.0
        qw = 0.25 * S
        qx = (R[2, 1] - R[1, 2]) / S
        qy = (R[0, 2] - R[2, 0]) / S
        qz = (R[1, 0] - R[0, 1]) / S
    else:
        if (R[0, 0] > R[1, 1]) and (R[0, 0] > R[2, 2]):
            S = np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2.0
            qw = (R[2, 1] - R[1, 2]) / S
            qx = 0.25 * S
            qy = (R[0, 1] + R[1, 0]) / S
            qz = (R[0, 2] + R[2, 0]) / S
        elif R[1, 1] > R[2, 2]:
            S = np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2.0
            qw = (R[0, 2] - R[2, 0]) / S
            qx = (R[0, 1] + R[1, 0]) / S
            qy = 0.25 * S
            qz = (R[1, 2] + R[2, 1]) / S
        else:
            S = np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2.0
            qw = (R[1, 0] - R[0, 1]) / S
            qx = (R[0, 2] + R[2, 0]) / S
            qy = (R[1, 2] + R[2, 1]) / S
            qz = 0.25 * S
    q = np.array([qw, qx, qy, qz], dtype=np.float64)
    norm = np.linalg.norm(q)
    if norm == 0.0:
        q = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
    else:
        q /= norm
    if q[0] < 0.0:
        q *= -1.0
    return q.astype(np.float32)


def _so2_features(R: np.ndarray, axis: TaskSpace) -> List[float]:
    if axis == TaskSpace.SO2_X:
        angle = float(np.arctan2(R[2, 1], R[2, 2]))
    elif axis == TaskSpace.SO2_Y:
        angle = float(np.arctan2(R[0, 2], R[0, 0]))
    elif axis == TaskSpace.SO2_Z:
        angle = float(np.arctan2(R[1, 0], R[0, 0]))
    else:
        return []
    return [float(np.cos(angle)), float(np.sin(angle))]


def _features_from_T(
    T: np.ndarray,
    axes: Optional[Sequence[TaskSpace]],
    *,
    ignore_y: bool = False,
) -> np.ndarray:
    T_arr = np.asarray(T, dtype=np.float32)
    if axes is None:
        return T_arr.reshape(-1)

    R = T_arr[:3, :3]
    p = T_arr[:3, 3]

    features: List[float] = []
    quat_cache: Optional[np.ndarray] = None
    for axis in axes:
        if axis == TaskSpace.X:
            features.append(float(p[0]))
        elif axis == TaskSpace.Y:
            if ignore_y:
                continue
            features.append(float(p[1]))
        elif axis == TaskSpace.Z:
            features.append(float(p[2]))
        elif axis in (TaskSpace.SO2_X, TaskSpace.SO2_Y, TaskSpace.SO2_Z):
            features.extend(_so2_features(R, axis))
        elif axis == TaskSpace.SO3:
            if quat_cache is None:
                quat_cache = _rotation_matrix_to_quaternion(R)
            features.extend(float(v) for v in quat_cache.tolist())

    return np.asarray(features, dtype=np.float32)


def _rotation_z(angle: float) -> np.ndarray:
    c = float(np.cos(angle))
    s = float(np.sin(angle))
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]], dtype=np.float64)


def _canonicalize_xy_transform(T: np.ndarray, *, enabled: bool) -> tuple[np.ndarray, float]:
    T_arr = np.asarray(T, dtype=np.float64)
    if not enabled:
        return T_arr.astype(np.float32), 0.0

    x = float(T_arr[0, 3])
    y = float(T_arr[1, 3])
    radius = float(np.hypot(x, y))
    if radius < 1e-8:
        return T_arr.astype(np.float32), 0.0

    theta = float(np.arctan2(y, x))
    Rz = _rotation_z(-theta)

    T_canon = T_arr.copy()
    T_canon[:3, :3] = Rz @ T_canon[:3, :3]
    position = Rz @ T_arr[:3, 3]
    position[0] = radius
    position[1] = 0.0
    T_canon[:3, 3] = position
    return T_canon.astype(np.float32), theta


def _features_from_transform(
    T: np.ndarray,
    axes: Optional[Sequence[TaskSpace]],
    *,
    use_xy_halfplane: bool,
) -> tuple[np.ndarray, float]:
    T_canon, theta = _canonicalize_xy_transform(T, enabled=use_xy_halfplane)
    feats = _features_from_T(T_canon, axes, ignore_y=use_xy_halfplane)
    return feats, theta


def _node_features(grid: Grid, node: int, *, use_xy_halfplane: bool = False) -> tuple[np.ndarray, float]:
    node_data = grid.graph.nodes[node]
    T = node_data.get("T")
    if T is None:
        return np.empty((0,), dtype=np.float32), 0.0

    axes = getattr(grid, "axes", None)
    return _features_from_transform(np.asarray(T, dtype=np.float32), axes, use_xy_halfplane=use_xy_halfplane)


def _workspace_fft_target(
    ws: WorkspaceSMMs,
    *,
    fft_cutoff: Optional[int] = None,
    angle_shift: float = 0.0,
) -> Tuple[np.ndarray, Optional[FFTTargetSpec]]:
    """Return flattened Fourier coefficients and shape metadata."""

    if ws is None or ws.data is None or len(ws.data) == 0:
        return np.empty((0,), dtype=np.float32), None

    flattened: List[np.ndarray] = []
    coeff_rows: Optional[int] = None
    coeff_cols: Optional[int] = None
    full_rows: Optional[int] = None

    for smm in ws.data:
        smm_used = smm
        if angle_shift != 0.0:
            angles = smm.angle
            if angles is None:
                continue
            angles = angles.astype(np.float32, copy=True)
            angles[:, 0] = angles[:, 0] - angle_shift
            smm_used = SMM(status=smm.status, data=angles)

        fft = smm_used.fft
        if fft is None:
            return np.empty((0,), dtype=np.float32), None
        coeffs = np.asarray(fft, dtype=np.complex64)

        original_rows = int(coeffs.shape[0])

        if fft_cutoff is not None and fft_cutoff > 0 and coeffs.shape[0] > 2 * fft_cutoff:
            keep = int(fft_cutoff)
            coeffs = np.concatenate([coeffs[:keep, :], coeffs[-keep:, :]], axis=0)
            if coeffs.shape[0] % 2 != 0:
                raise ValueError("Trimmed Fourier coefficients must have an even number of rows")

        if coeff_rows is None:
            coeff_rows = int(coeffs.shape[0])
            coeff_cols = int(coeffs.shape[1])
            full_rows = original_rows
        elif coeff_rows != coeffs.shape[0] or coeff_cols != coeffs.shape[1]:
            raise ValueError("Inconsistent Fourier coefficient dimensions across branches")
        elif full_rows is not None and full_rows != original_rows:
            raise ValueError("Inconsistent original FFT dimensions across branches")
        else:
            full_rows = original_rows if full_rows is None else full_rows

        coeffs_flat = coeffs.reshape(-1)
        real_imag = np.concatenate([coeffs_flat.real, coeffs_flat.imag], axis=0).astype(np.float32, copy=False)
        flattened.append(real_imag)

    if not flattened:
        return np.empty((0,), dtype=np.float32), None

    if coeff_rows is None or coeff_cols is None or full_rows is None:
        raise ValueError("Empty coefficient metadata produced")

    meta = FFTTargetSpec(
        branches=len(ws.data),
        coeff_rows=int(coeff_rows),
        coeff_cols=int(coeff_cols),
        full_rows=int(full_rows),
    )

    return np.concatenate(flattened, axis=0).astype(np.float32, copy=False), meta


def _resolve_device(device: Union[str, torch.device]) -> torch.device:
    if isinstance(device, torch.device):
        return device

    key = device.strip().lower()

    if key == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():  # type: ignore[attr-defined]
            return torch.device("mps")
        return torch.device("cpu")

    if key in {"cpu"}:
        return torch.device("cpu")

    if key in {"cuda", "gpu"}:
        return torch.device("cuda")

    if key in {"mac", "macbook", "mps"}:
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():  # type: ignore[attr-defined]
            return torch.device("mps")
        print("[train_networks] Requested MacBook/MPS device but it is unavailable; falling back to CPU")
        return torch.device("cpu")

    return torch.device(device)


def _collect_cluster_samples(
    grid: Grid,
    cluster: nx.Graph,
    cluster_id: int,
    *,
    fft_cutoff: Optional[int] = None,
    use_xy_halfplane: bool = False,
) -> Tuple[np.ndarray, np.ndarray, List[int], FFTTargetSpec]:
    X_list: List[np.ndarray] = []
    Y_list: List[np.ndarray] = []
    node_ids: List[int] = []

    target_dim: Optional[int] = None
    feature_dim: Optional[int] = None
    target_meta: Optional[FFTTargetSpec] = None

    for node in cluster.nodes:
        try:
            ws = grid.get_node_smms(int(node))
        except Exception:  # pragma: no cover - defensive
                ws = None

        if ws is None or ws.status != SMMStatus.OK:
                continue

        x, theta = _node_features(grid, int(node), use_xy_halfplane=use_xy_halfplane)
        if x.size == 0:
            raise ValueError(f"Cluster {cluster_id} node {node} missing transformation features")

        if feature_dim is None:
            feature_dim = x.shape[0]
        elif feature_dim != x.shape[0]:
            raise ValueError(f"Cluster {cluster_id} produced inconsistent feature dimensions")

        y, meta = _workspace_fft_target(ws, fft_cutoff=fft_cutoff, angle_shift=theta if use_xy_halfplane else 0.0)
        if y.size == 0:
            raise ValueError(f"Cluster {cluster_id} node {node} produced empty target")

        if meta is None:
            raise ValueError(f"Cluster {cluster_id} node {node} missing Fourier metadata")

        if target_dim is None:
            target_dim = y.shape[0]
        elif target_dim != y.shape[0]:
            raise ValueError(f"Cluster {cluster_id} produced inconsistent target dimensions")

        if target_meta is None:
            target_meta = meta
        elif target_meta != meta:
            raise ValueError(f"Cluster {cluster_id} produced inconsistent Fourier metadata")

        X_list.append(x)
        Y_list.append(y)
        node_ids.append(int(node))

    if not X_list:
        raise ValueError(f"Cluster {cluster_id} produced no valid training samples")

    if target_meta is None:
        raise ValueError(f"Cluster {cluster_id} did not produce target metadata")

    X = np.stack(X_list, axis=0).astype(np.float32, copy=False)
    Y = np.stack(Y_list, axis=0).astype(np.float32, copy=False)
    return X, Y, node_ids, target_meta


def _resolve_cache_root(base_dir: Path | str | None) -> Path:
    if base_dir is not None:
        return ensure_dir(Path(base_dir))

    if user_cache_dir is not None:
        return ensure_dir(Path(user_cache_dir(appname="realtime_smm", appauthor=False)))

    cache_root = (
        Path.home() / "Library" / "Caches" / "realtime_smm"
        if platform.system() == "Darwin"
        else Path(os.getenv("XDG_CACHE_HOME", Path.home() / ".cache")) / "realtime_smm"
    )
    return ensure_dir(cache_root)


def _train_single_model(
    X: np.ndarray,
    Y: np.ndarray,
    config: TrainingConfig,
    *,
    criterion: Optional[nn.Module] = None,
) -> tuple[MLP, List[float]]:
    device = _resolve_device(config.device)

    dataset = TensorDataset(torch.from_numpy(X), torch.from_numpy(Y))
    loader = DataLoader(dataset, batch_size=config.batch_size, shuffle=config.shuffle)

    model = MLP(
        input_dim=X.shape[1],
        hidden_dims=config.hidden_dims,
        output_dim=Y.shape[1],
        activation=config.activation,
        dropout=config.dropout,
        layer_norm=config.layer_norm,
    )

    loss_fn = criterion if criterion is not None else nn.MSELoss()

    try:
        from pytorch_minimize.optim import BasinHoppingWrapper

        cpu_device = torch.device("cpu")
        model.to(device=cpu_device)

        basinhop = BasinHoppingWrapper(
            model.parameters(),
            {"method": "CG", "options": {"maxiter": 250}},
            {"niter": 1, "T": 0.05, "interval": 20, "stepsize": 0.05},
        )

        full_x = torch.from_numpy(X).to(cpu_device)
        full_y = torch.from_numpy(Y).to(cpu_device)

        def closure() -> torch.Tensor:
            basinhop.zero_grad()  # type: ignore[call-arg]
            preds = model(full_x)
            loss = loss_fn(preds, full_y)
            loss.backward()
            return loss

        try:
            import contextlib
            import io
            dummy_out = io.StringIO()
            with contextlib.redirect_stdout(dummy_out), contextlib.redirect_stderr(dummy_out):
                for _ in range(10):
                    basinhop.step(closure)
        except Exception:
            pass
        finally:
            model.zero_grad(set_to_none=True)
    except ModuleNotFoundError:
        print("pytorch-minimize not installed")
        pass
    except ImportError:
        print("pytorch-minimize not imported")
        pass

    model.to(device)
    optimizer = torch.optim.Adam(
        model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay
    )

    scheduler = None
    if config.scheduler == SchedulerType.STEP:
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer,
            step_size=max(1, config.scheduler_step_size),
            gamma=float(config.scheduler_gamma),
        )
    elif config.scheduler == SchedulerType.COSINE:
        t_max = config.scheduler_t_max if config.scheduler_t_max is not None else max(1, config.epochs)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=int(t_max),
            eta_min=float(config.scheduler_min_lr),
        )
    elif config.scheduler == SchedulerType.PLATEAU:
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=float(config.scheduler_gamma),
            patience=max(1, config.scheduler_patience),
            min_lr=float(config.scheduler_min_lr),
        )

    loss_history: List[float] = []

    use_bar = tqdm is not None
    epoch_iter = range(config.epochs)
    bar = None
    if use_bar:
        bar = tqdm(epoch_iter, desc="training...", leave=False)
        epoch_iter = bar  # type: ignore[assignment]

    for epoch in epoch_iter:
        model.train()
        total_loss = 0.0
        total_samples = 0
        for xb, yb in loader:
            xb = xb.to(device)
            yb = yb.to(device)

            optimizer.zero_grad(set_to_none=True)
            preds = model(xb)
            loss = loss_fn(preds, yb)
            loss.backward()
            optimizer.step()

            total_loss += float(loss.item()) * xb.size(0)
            total_samples += xb.size(0)

        avg_loss = total_loss / max(total_samples, 1)
        loss_history.append(avg_loss)
        if bar is not None:
            bar.set_postfix(loss=f"{avg_loss:.6f}")
        else:
            if (epoch + 1 == config.epochs) or ((epoch + 1) % max(1, config.epochs // 5) == 0):
                print(f"  epoch {epoch + 1}/{config.epochs}: loss={avg_loss:.6f}")

        if scheduler is not None:
            if config.scheduler == SchedulerType.PLATEAU:
                scheduler.step(avg_loss)
            else:
                scheduler.step()

    model.to(torch.device("cpu"))
    if bar is not None:
        bar.close()
    return model, loss_history


def train_cluster_networks(
    grid: Grid,
    clusters: Sequence[nx.Graph],
    *,
    name: str,
    output_root: Path | str | None = None,
    config: Optional[TrainingConfig] = None,
) -> SMMNetworkBundle:
    """Train canonicalized cluster regressors and persist a single bundle.

    Learned models are serialized to ``<cache>/<name>/bundle.pt`` (with the
    cache root derived from ``output_root`` when provided, otherwise the
    runtime cache). When ``grid`` originates from an XY half-plane run, each
    training sample is rotated into the positive X half-plane before feature
    extraction and the target SMMs are expressed in that canonical frame.
    """

    if config is None:
        config = TrainingConfig()

    run_dir = _resolve_cache_root(output_root) / name
    run_dir.mkdir(parents=True, exist_ok=True)

    cluster_models: Dict[str, ClusterModelBundle] = {}
    use_xy_halfplane = bool(getattr(grid, "use_xy_halfplane", False))

    class_X_chunks: List[np.ndarray] = []
    class_idx_chunks: List[np.ndarray] = []
    cluster_keys: List[str] = []
    classifier: Optional[MLP] = None

    print(f"saving models to {run_dir}")

    for cid, cluster in enumerate(clusters):
        X, Y, node_ids, cluster_spec = _collect_cluster_samples(
            grid,
            cluster,
            cid,
            fft_cutoff=config.fft_cutoff,
            use_xy_halfplane=use_xy_halfplane,
        )

        print(f"cluster {cid}: {X.shape[0]} samples")
        model, losses = _train_single_model(X, Y, config)

        cluster_key = f"cluster_{cid:03d}"
        cluster_models[cluster_key] = ClusterModelBundle(
            model=model,
            spec=cluster_spec,
            feature_dim=int(X.shape[1]),
            output_dim=int(Y.shape[1]),
        )

        class_idx = len(cluster_keys)
        cluster_keys.append(cluster_key)
        class_X_chunks.append(X)
        class_idx_chunks.append(np.full((X.shape[0],), class_idx, dtype=np.int64))

        final_loss = losses[-1] if losses else float("nan")
        print(f"cluster {cid}: trained (loss={final_loss:.6f})")

    if not class_X_chunks:
        raise ValueError("No training data collected for classifier")

    print("training classifier")
    class_X = np.concatenate(class_X_chunks, axis=0).astype(np.float32, copy=False)
    class_labels = np.concatenate(class_idx_chunks, axis=0).astype(np.int64, copy=False)
    num_classes = len(cluster_keys)
    class_y = np.zeros((class_labels.shape[0], num_classes), dtype=np.float32)
    class_y[np.arange(class_labels.shape[0]), class_labels] = 1.0

    classifier, classifier_losses = _train_single_model(
        class_X,
        class_y,
        config,
        criterion=nn.BCEWithLogitsLoss(),
    )

    axes = getattr(grid, "axes", tuple())
    bundle = SMMNetworkBundle(
        axes=axes,
        fft_cutoff=config.fft_cutoff,
        clusters=cluster_models,
        classifier=classifier,
        cluster_keys=cluster_keys,
        device="cpu",
        use_xy_halfplane=use_xy_halfplane,
    )
    bundle_path = run_dir / "bundle.pt"
    bundle.save(bundle_path)
    final_classifier_loss = classifier_losses[-1] if classifier_losses else float("nan")
    print(f"classifier trained (loss={final_classifier_loss:.6f})")
    print(f"bundle saved to {bundle_path}")

    return bundle


class SMMNetworkBundle:
    """Container for trained classifier and cluster regressors."""

    def __init__(
        self,
        *,
        axes: Sequence[TaskSpace],
        fft_cutoff: Optional[int],
        clusters: Dict[str, ClusterModelBundle],
        classifier: MLP,
        cluster_keys: Sequence[str] | None = None,
        device: Union[str, torch.device] = "cpu",
        use_xy_halfplane: bool = False,
    ) -> None:
        if not clusters:
            raise ValueError("At least one cluster model is required")

        self.axes: Tuple[TaskSpace, ...] = tuple(axes)
        self.fft_cutoff: Optional[int] = fft_cutoff
        self.clusters: Dict[str, ClusterModelBundle] = dict(clusters)
        self.classifier: MLP = classifier
        self.use_xy_halfplane: bool = bool(use_xy_halfplane)
        keys = list(cluster_keys) if cluster_keys is not None else sorted(clusters.keys())
        if not keys:
            keys = sorted(clusters.keys())
        for key in keys:
            if key not in self.clusters:
                raise KeyError(f"Classifier key '{key}' has no corresponding cluster model")
        self._cluster_keys: List[str] = keys

        self.to(device)

    def to(self, device: Union[str, torch.device]) -> "SMMNetworkBundle":
        torch_device = _resolve_device(device)
        self.device = torch_device
        for entry in self.clusters.values():
            entry.model.to(torch_device)
            entry.model.eval()
        self.classifier.to(torch_device)
        self.classifier.eval()
        return self

    def save(self, path: Path | str) -> None:
        prev_device = self.device
        self.to("cpu")
        torch.save(self, Path(path))
        if prev_device.type != "cpu":
            self.to(prev_device)

    @classmethod
    def load(
        cls,
        name: str | None = None,
        *,
        path: Path | str | None = None,
        base_dir: Path | str | None = None,
        device: Union[str, torch.device] = "cpu",
    ) -> "SMMNetworkBundle":
        if (name is None and path is None) or (name is not None and path is not None):
            raise ValueError("Specify exactly one of 'name' or 'path'")

        if path is None:
            if name is None:
                raise ValueError("'name' must be provided when 'path' is omitted")
            cache_root = _resolve_cache_root(base_dir)
            path = cache_root / name / "bundle.pt"
        else:
            path = Path(path)

        bundle: "SMMNetworkBundle" = torch.load(path, map_location="cpu")
        return bundle.to(device)

    def _features_from_matrix(self, T: np.ndarray) -> tuple[np.ndarray, float]:
        feats, theta = _features_from_transform(
            np.asarray(T, dtype=np.float32),
            self.axes,
            use_xy_halfplane=self.use_xy_halfplane,
        )
        if feats.size == 0:
            raise ValueError("Transformation produced empty feature vector")
        return feats, theta

    def _select_cluster(self, feats: np.ndarray) -> str:
        if not self.clusters:
            raise RuntimeError("No cluster models loaded")

        with torch.no_grad():
            x = torch.from_numpy(feats.reshape(1, -1)).to(self.device)
            logits = self.classifier(x).cpu().numpy()[0]
        idx = int(np.argmax(logits))
        if idx >= len(self._cluster_keys):
            raise IndexError("Classifier predicted index outside available clusters")
        return self._cluster_keys[idx]

    def _fourier_vector_to_workspace(
        self,
        vec: np.ndarray,
        spec: FFTTargetSpec,
        *,
        angle_offset: float = 0.0,
        samples: Optional[int] = None,
    ) -> WorkspaceSMMs:
        if spec.branches <= 0 or spec.coeff_rows <= 0 or spec.coeff_cols <= 0:
            raise ValueError("Invalid Fourier target specification")

        full_rows = getattr(spec, "full_rows", spec.coeff_rows)
        target_rows = int(samples) if samples is not None else int(full_rows)
        if target_rows < full_rows:
            raise ValueError("samples must be greater than or equal to the original resolution")

        branch_len = 2 * spec.coeff_rows * spec.coeff_cols
        expected = spec.branches * branch_len
        if vec.size != expected:
            raise ValueError(f"Vector length {vec.size} does not match expected {expected}")

        smms: List[SMM] = []
        for b in range(spec.branches):
            start = b * branch_len
            segment = vec[start : start + branch_len]
            half = branch_len // 2
            real = segment[:half]
            imag = segment[half:]
            complex_flat = real + 1j * imag
            compact = complex_flat.reshape(spec.coeff_rows, spec.coeff_cols)

            if spec.coeff_rows < full_rows:
                if spec.coeff_rows % 2 != 0:
                    raise ValueError("Fourier coefficient rows must be even when fft_cutoff is applied")
                keep = spec.coeff_rows // 2
                freq_full = np.zeros((full_rows, spec.coeff_cols), dtype=compact.dtype)
                freq_full[:keep, :] = compact[:keep, :]
                freq_full[-keep:, :] = compact[keep:, :]
            else:
                freq_full = compact.astype(np.complex64, copy=False)

            if target_rows == full_rows:
                torus = np.fft.ifft(freq_full, axis=0)
            else:
                freq_indices = (np.fft.fftfreq(full_rows) * full_rows).astype(np.float64)
                theta = (2.0 * np.pi / target_rows) * np.arange(target_rows, dtype=np.float64)
                exponent = np.exp(1j * np.outer(freq_indices, theta))
                values = (freq_full.T @ exponent).T / float(full_rows)
                torus = values.astype(np.complex64, copy=False)

            angles = np.angle(torus).astype(np.float32, copy=False)
            if angle_offset != 0.0:
                angles[:, 0] = angles[:, 0] + angle_offset
                angles = np.angle(np.exp(1j * angles))
            smms.append(SMM(status=SMMStatus.OK, data=angles))

        return WorkspaceSMMs(status=SMMStatus.OK, data=smms)

    def __call__(self, T: np.ndarray, *, samples: Optional[int] = None) -> WorkspaceSMMs:
        feats, theta = self._features_from_matrix(T)

        key = self._select_cluster(feats)
        entry = self.clusters[key]

        model: MLP = entry.model
        spec: FFTTargetSpec = entry.spec
        expected_feat = entry.feature_dim
        if feats.shape[0] != expected_feat:
            raise ValueError(f"Feature dimension {feats.shape[0]} does not match expected {expected_feat}")

        with torch.no_grad():
            x = torch.from_numpy(feats.reshape(1, -1)).to(self.device)
            pred = model(x).cpu().numpy()[0]

        expected_out = entry.output_dim
        if pred.size != expected_out:
            raise ValueError(f"Output dimension {pred.size} does not match expected {expected_out}")

        angle_offset = theta if self.use_xy_halfplane else 0.0
        return self._fourier_vector_to_workspace(
            pred,
            spec,
            angle_offset=angle_offset,
            samples=samples,
        )


# Allows the network bundle to be saved and loaded using PyTorch
torch.serialization.add_safe_globals([
    SMMNetworkBundle,
    ClusterModelBundle,
    FFTTargetSpec,
    TaskSpace,
    MLP,
    nn.Sequential,
    nn.Linear,
    nn.LayerNorm,
    nn.Dropout,
    nn.ReLU,
    nn.GELU,
    nn.LeakyReLU,
    nn.PReLU,
    nn.Tanh,
    nn.Sigmoid,
    nn.SiLU,
])


