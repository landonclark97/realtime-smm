from __future__ import annotations

from pathlib import Path
from typing import Optional, List, Tuple, Any, Dict
import platform
import os

import json
import numpy as np

try:
    import zarr  # type: ignore
except Exception:  # pragma: no cover
    zarr = None  # type: ignore

# No special store imports needed; use filesystem-backed directory store via path

from .types import SMM, SMMStatus, WorkspaceSMMs
from .types import GridParams, SMMSolverParams

try:
    from platformdirs import user_cache_dir  # type: ignore
except Exception:
    user_cache_dir = None  # type: ignore


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


class SMMStore:
    """Persistent storage for WorkspaceSMMs per grid node using a directory-backed Zarr store."""

    def __init__(self, run_dir: Path) -> None:
        self.run_dir: Path = ensure_dir(Path(run_dir))
        if zarr is None:
            raise RuntimeError("Zarr is required for disk caching. Install 'zarr' or set use_cache=False.")
        # Open/create a root group backed by a filesystem directory
        self._dir_path: Path = self.run_dir / "smms.zarr"
        self._zroot = zarr.open_group(str(self._dir_path), mode="a")  # type: ignore
        # Ensure top-level namespace exists
        try:
            if self._zroot.get("nodes") is None:  # type: ignore
                self._zroot.create_group("nodes")  # type: ignore
            if self._zroot.get("aligned") is None:  # type: ignore
                self._zroot.create_group("aligned")  # type: ignore
        except Exception:
            pass


    def _node_group_path(self, node_id: int) -> str:
        return f"nodes/{int(node_id)}"

    def _get_group(self, group_name: str, node_id: int):
        base = self._zroot.get(group_name)  # type: ignore
        if base is None:
            base = self._zroot.create_group(group_name)  # type: ignore
        path = f"{group_name}/{int(node_id)}"
        grp = self._zroot.get(path)  # type: ignore
        if grp is None:
            grp = self._zroot.create_group(path)  # type: ignore
        return grp

    def put(self, node_id: int, ws: WorkspaceSMMs, *, group_name: str = "nodes") -> None:
        # Zarr backend only
        grp = self._get_group(group_name, node_id)
        # Ensure branches group exists (reuse to avoid create_group collision in ZipStore)
        branches_grp = grp.get("branches")
        if branches_grp is None:
            branches_grp = grp.create_group("branches")

        if ws is None or ws.status != SMMStatus.OK:
            # Non-OK: nothing else to write
            return

        branches = len(ws.data)
        samples = ws.data[0].samples if branches > 0 else 0
        # Write branches as datasets
        for bi, smm in enumerate(ws.data):
            arr = np.asarray(smm.angle, dtype=np.float32)
            # Use a single chunk per branch to minimize filesystem file count
            chunk_shape = arr.shape if arr.ndim == 2 else None
            try:
                from numcodecs import Blosc  # type: ignore
                comp = Blosc(cname="zstd", clevel=5)
            except Exception:
                comp = None
            compressors = [comp] if comp is not None else None
            name = f"{bi}"
            # Zarr v3 API
            # If dataset exists, overwrite its contents; otherwise create
            existing = branches_grp.get(name)
            if existing is not None:
                existing[...] = arr
            else:
                try:
                    ds = branches_grp.create_array(
                        name,
                        shape=arr.shape,
                        dtype=arr.dtype,
                        chunk_shape=chunk_shape,
                        compressors=compressors,
                        overwrite=True,
                    )
                    ds[...] = arr
                except TypeError:
                    # Fallback with minimal kwargs for strict v3 implementations
                    ds = branches_grp.create_array(name, shape=arr.shape, dtype=arr.dtype, overwrite=True)
                    ds[...] = arr
        # Avoid setting group attrs to prevent duplicate zarr.json entries in ZipStore

    def get(self, node_id: int) -> Optional[WorkspaceSMMs]:
        # Prefer aligned result if present
        grp = self._zroot.get(f"aligned/{int(node_id)}")  # type: ignore
        if grp is None:
            grp = self._zroot.get(self._node_group_path(node_id))  # type: ignore
        if grp is None:
            return None
        data: List[SMM] = []
        bgrp = grp.get("branches")
        if bgrp is None:
            return WorkspaceSMMs(status=SMMStatus.NONE, data=data)
        # Enumerate datasets under branches instead of relying on attrs
        try:
            keys = list(bgrp.keys())  # type: ignore
        except Exception:
            keys = []
        for name in keys:
            darr = bgrp.get(str(name))
            if darr is None:
                continue
            arr = darr[...]
            data.append(SMM(status=SMMStatus.OK, data=np.asarray(arr)))
        status = SMMStatus.OK if len(data) > 0 else SMMStatus.NONE
        return WorkspaceSMMs(status=status, data=data)

    def has(self, node_id: int) -> bool:
        # Use get() instead of membership; membership only checks immediate children
        if self._zroot.get(self._node_group_path(node_id)) is not None:  # type: ignore
            return True
        return False

    # Convenience for aligned writes
    def put_aligned(self, node_id: int, ws: WorkspaceSMMs) -> None:
        self.put(node_id, ws, group_name="aligned")

    def delete_node(self, node_id: int, *, include_aligned: bool = True) -> None:
        """Delete persisted data for a node from the directory-backed store."""
        try:
            targets = [self._dir_path / "nodes" / str(int(node_id))]
            if include_aligned:
                targets.append(self._dir_path / "aligned" / str(int(node_id)))
            for t in targets:
                if t.exists():
                    for p in sorted(t.rglob("*"), reverse=True):
                        try:
                            if p.is_file():
                                p.unlink()
                            else:
                                p.rmdir()
                        except Exception:
                            pass
                    try:
                        t.rmdir()
                    except Exception:
                        pass
        except Exception:
            pass

    def write_run_metadata(self, robot: Any, grid_params: GridParams, smm_params: SMMSolverParams) -> None:
        """Persist a small JSON blob describing the run configuration.

        This intentionally avoids saving the full Grid object; instead it
        captures enough to reproduce the run directory key and rebuild later.
        """
        dh = getattr(robot, "_dh", None)
        dh_serialized = dh.tolist() if dh is not None else None
        jts = getattr(robot, "_joint_types", None)
        if jts is not None:
            try:
                jt_serialized = [str(j.value) if hasattr(j, "value") else str(j) for j in jts]
            except Exception:
                jt_serialized = [str(j) for j in jts]
        else:
            jt_serialized = None
        ts = getattr(robot, "_taskspace", None)
        ts_serialized = int(ts) if ts is not None else None
        meta = {
            "robot": {
                "dh": dh_serialized,
                "joint_types": jt_serialized,
                "taskspace": ts_serialized,
            },
            "grid_params": self._jsonable(grid_params),
            "smm_params": self._jsonable(smm_params),
        }
        (self.run_dir / "run_meta.json").write_text(json.dumps(meta, sort_keys=True))

    @staticmethod
    def _default_cache_root() -> Path:
        cache_root = None
        if user_cache_dir is not None:
            try:
                cache_root = Path(user_cache_dir(appname="realtime_smm", appauthor=False))
            except Exception:
                cache_root = None
        if cache_root is None:
            if platform.system() == "Darwin":
                cache_root = Path.home() / "Library" / "Caches" / "realtime_smm"
            else:
                cache_root = Path(os.getenv("XDG_CACHE_HOME", Path.home() / ".cache")) / "realtime_smm"
        return ensure_dir(cache_root)

    @staticmethod
    def _jsonable(obj: Any) -> Any:
        if hasattr(obj, "__dict__"):
            try:
                from dataclasses import asdict
                return asdict(obj)
            except Exception:
                return obj.__dict__
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        try:
            return int(obj)
        except Exception:
            return str(obj)

    @classmethod
    def run_key(cls, robot: Any, grid_params: GridParams, smm_params: SMMSolverParams) -> str:
        payload = {
            "dh": getattr(robot, "_dh", None).tolist() if getattr(robot, "_dh", None) is not None else None,
            "joint_types": [str(jt.value) if hasattr(jt, "value") else str(jt) for jt in getattr(robot, "_joint_types", [])] if hasattr(robot, "_joint_types") else None,
            "taskspace": int(getattr(robot, "_taskspace", 0)) if getattr(robot, "_taskspace", None) is not None else 0,
            "grid": cls._jsonable(grid_params),
            "smm": cls._jsonable(smm_params),
        }
        s = json.dumps(payload, sort_keys=True)
        import hashlib
        return hashlib.sha256(s.encode()).hexdigest()[:16]

    @classmethod
    def build_for(cls, robot: Any, grid_params: GridParams, smm_params: SMMSolverParams) -> "SMMStore":
        root = cls._default_cache_root()
        key = cls.run_key(robot, grid_params, smm_params)
        return cls(root / key)

    # ---------------- In-memory store variant ----------------
    @classmethod
    def build_memory(cls) -> "MemorySMMStore":
        return MemorySMMStore()


class MemorySMMStore:
    """In-memory SMM store with the same interface subset as SMMStore.

    Used when use_cache=False to unify code paths while keeping arrays in RAM.
    """

    def __init__(self) -> None:
        self._data: Dict[int, WorkspaceSMMs] = {}

    def put(self, node_id: int, ws: WorkspaceSMMs) -> None:
        self._data[int(node_id)] = ws

    def get(self, node_id: int) -> Optional[WorkspaceSMMs]:
        return self._data.get(int(node_id))

    def has(self, node_id: int) -> bool:
        return int(node_id) in self._data

    def delete_node(self, node_id: int, *, include_aligned: bool = True) -> None:
        try:
            self._data.pop(int(node_id), None)
        except Exception:
            pass
