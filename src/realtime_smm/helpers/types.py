from enum import Enum, IntFlag, IntEnum, auto
from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np

from .nn import MLP


DEFAULT_FLOAT_DTYPE = np.float32
def set_default_float_dtype(dtype: np.dtype) -> None:
    global DEFAULT_FLOAT_DTYPE
    assert isinstance(dtype, np.dtype)
    DEFAULT_FLOAT_DTYPE = dtype


class JointType(Enum):
    REVOLUTE = "R"
    PRISMATIC = "P"


class Axis(IntFlag):
    X = 0
    Y = auto()
    Z = auto()
    SO2 = auto()
    SO3 = auto()


class TaskSpace(IntFlag):
    """Bitmask for selecting task-space components.

    Combine flags with bitwise OR, e.g., TaskSpace.X | TaskSpace.Z | TaskSpace.SO3.
    """
    NONE = 0
    X = auto()
    Y = auto()
    Z = auto()
    SO2_X = auto()  # single-axis orientation
    SO2_Y = auto()  # single-axis orientation
    SO2_Z = auto()  # single-axis orientation
    SO3 = auto()  # full 3D orientation

    def to_mask(self) -> np.ndarray:
        """Convert to a 6D boolean mask [x,y,z,roll,pitch,yaw].

        so2_axis selects which orientation axis to keep for SO2: one of {'X','Y','Z'}.
        If SO3 is present, all three orientation rows are enabled.
        """
        # Enforce SO2 uniqueness and no mixing with SO3
        so2_flags = [TaskSpace.SO2_X in self, TaskSpace.SO2_Y in self, TaskSpace.SO2_Z in self]
        if sum(so2_flags) > 1:
            raise ValueError("Only one of SO2_X, SO2_Y, SO2_Z may be set in TaskSpace")
        if (TaskSpace.SO3 in self) and any(so2_flags):
            raise ValueError("SO3 may not be combined with SO2_* in TaskSpace")
        mask = np.array([False, False, False, False, False, False], dtype=bool)
        if TaskSpace.X in self:
            mask[0] = True
        if TaskSpace.Y in self:
            mask[1] = True
        if TaskSpace.Z in self:
            mask[2] = True
        if TaskSpace.SO3 in self:
            mask[3:6] = True
        elif TaskSpace.SO2_X in self:
            mask[3] = True
        elif TaskSpace.SO2_Y in self:
            mask[4] = True
        elif TaskSpace.SO2_Z in self:
            mask[5] = True
        return mask



@dataclass(frozen=True)
class AxisParams:
    """Typed parameters for a single task-space axis.

    - lower, upper: numeric limits (used for positional and SO2 axes; SO3 ignores)
    - resolution: spacing for positional/SO2 axes; for SO3 this controls sampling density
    """
    axis: TaskSpace
    resolution: float
    lower: Optional[float] = None
    upper: Optional[float] = None

    def __post_init__(self) -> None:
        if self.axis not in TaskSpace:
            raise ValueError("axis must be a valid TaskSpace")
        if self.lower is not None and self.upper is not None and self.lower > self.upper:
            raise ValueError("lower must be less than upper")
        if self.resolution <= 0:
            raise ValueError("resolution must be > 0")
        if self.axis in [TaskSpace.X, TaskSpace.Y, TaskSpace.Z] and self.lower is None and self.upper is None:
            raise ValueError("lower and upper must be set for positional axes")


@dataclass(frozen=True)
class DHLink:
    """Typed Denavit–Hartenberg parameters for a single link.

    Standard DH: [a, alpha, d, theta].
    joint_kind selects whether q adds to theta (REVOLUTE) or d (PRISMATIC).
    """
    a: float
    alpha: float
    d: float
    theta: float
    joint_type: JointType = JointType.REVOLUTE
    lower_limit: float = -np.pi
    upper_limit: float = np.pi


class SMMStatus(IntEnum):
    NONE = 0
    OK = 1
    SINGULAR = 2
    ERROR = 3


class NodeStage(IntEnum):
    """Processing stage for each grid node.

    This drives progress tracking without external completion utilities.
    """
    INIT = 0
    WAITING_FOR_SMMS = 1
    COMPUTED_SMMS = 2
    ALIGNED_SMMS = 3


@dataclass
class SMM:
    """Self-Motion Manifold result container.

    status: SMMStatus indicating result type
    data: ndarray of manifold samples or None when not available
    """
    status: SMMStatus
    data: np.ndarray | None

    def __post_init__(self) -> None:
        if self.data is not None and self.data.ndim != 2:
            raise ValueError("data must have shape (samples, n)")
        if self.data is not None:
            if self.data is not None and np.iscomplexobj(self.data):
                self.data = np.angle(self.data)
            if self.data.dtype != DEFAULT_FLOAT_DTYPE:
                self.data = self.data.astype(DEFAULT_FLOAT_DTYPE, copy=False)

    @property
    def samples(self) -> int:
        """Return the number of samples in the manifold."""
        return self.data.shape[0]

    @property
    def torus(self) -> np.ndarray:
        """Return the torus-wrapped manifold data."""
        if self.data is not None:
            # unnecessary with __post_init__, but who cares...
            if np.iscomplexobj(self.data):
                return np.copy(self.data)
            return np.copy(np.exp(1j * self.data))
        return None

    @property
    def angle(self) -> np.ndarray:
        """Return the real manifold data."""
        if self.data is not None:
            # unnecessary with __post_init__, but who cares...
            if np.iscomplexobj(self.data):
                return np.copy(np.angle(self.data))
            return np.copy(self.data)
        return None

    @property
    def fft(self) -> np.ndarray:
        """Return the FFT of the manifold data."""
        
        if self.data is None:
            return None
        data = self.torus
        fft_data = np.fft.fft(data, axis=0)

        if np.abs(fft_data[1,0]) > 1e-6:
            samples = data.shape[0]

            rot_ang = np.arctan2(np.imag(fft_data[1,0]),np.real(fft_data[1,0]))
            dt = (rot_ang*samples)/(2.0*np.pi)
            dt = round(dt)

            sq = np.linspace(0,samples-1,num=samples)

            r = np.exp(-1j*2.0*np.pi*dt*sq/samples)
            fft_data = (fft_data.T*r).T

        return fft_data


@dataclass
class WorkspaceSMMs:
    """Self-Motion Manifold result container.

    status: SMMStatus indicating result type
    data: ndarray of manifold samples or None when not available
    """
    status: SMMStatus
    data: list[SMM]

    def __post_init__(self) -> None:
        if self.data is not None and not all(isinstance(smm, SMM) for smm in self.data):
            raise ValueError("data must be a list of SMM objects")
        if self.data is not None and not all(smm.status == SMMStatus.OK for smm in self.data):
            raise ValueError("all SMMs must be OK")
        if self.data is not None and not all(smm.samples == self.data[0].samples for smm in self.data):
            raise ValueError("all SMMs must have the same number of samples")

    @property
    def branches(self) -> int:
        """Return the number of branches in the manifold."""
        return len(self.data)

    @property
    def samples(self) -> int:
        """Return the number of samples in the manifold."""
        return self.data[0].samples


@dataclass(frozen=True)
class SMMSolverParams:
    """Parameters for the SMM solver."""
    samples: int = 128
    step: float = 0.05
    sing_thresh: float = 5e-3
    smm_iters: int = 1000


@dataclass(frozen=True)
class GridParams:
    """Parameters for the grid."""
    pos_resolution: float = 0.1
    orn_resolution: float = 0.1
    use_xy_halfplane: bool = True


class SchedulerType(Enum):
    STEP = "step"
    COSINE = "cosine"
    PLATEAU = "plateau"


@dataclass
class TrainingConfig:
    epochs: int = 200
    batch_size: int = 64
    learning_rate: float = 1e-3
    weight_decay: float = 0.0
    hidden_dims: Tuple[int, ...] = (256, 256)
    activation: str = "relu"
    dropout: float = 0.0
    layer_norm: bool = False
    device: str = "auto"
    shuffle: bool = True
    fft_cutoff: Optional[int] = None
    scheduler: Optional[SchedulerType] = None
    scheduler_step_size: int = 50
    scheduler_gamma: float = 0.9
    scheduler_patience: int = 10
    scheduler_min_lr: float = 1e-6
    scheduler_t_max: Optional[int] = None

    def __post_init__(self) -> None:
        if self.fft_cutoff is not None and self.fft_cutoff < 0:
            raise ValueError("fft_cutoff must be greater than or equal to 0")
        if self.scheduler is not None and isinstance(self.scheduler, str):
            self.scheduler = SchedulerType(self.scheduler)


@dataclass
class FFTTargetSpec:
    branches: int
    coeff_rows: int
    coeff_cols: int
    full_rows: int


@dataclass
class ClusterModelBundle:
    model: MLP
    spec: FFTTargetSpec
    feature_dim: int
    output_dim: int