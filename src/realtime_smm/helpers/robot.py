from __future__ import annotations

from typing import Callable, List, Optional, Sequence, Tuple

import numpy as np
from scipy.optimize import minimize

from .types import SMMStatus, SMM, JointType, DHLink, TaskSpace, WorkspaceSMMs


def _ensure_4x4(matrix: np.ndarray) -> np.ndarray:
    if matrix.shape != (4, 4):
        raise ValueError("Transform must be 4x4 homogeneous matrix")
    return matrix


def _tr(x: float | np.ndarray) -> float | np.ndarray:
    return np.cos(x) + 1j * np.sin(x)

def _tr_inv(x: float | np.ndarray) -> float | np.ndarray:
    return np.arctan2(np.imag(x), np.real(x))


class Robot:
    """Simple DH-parameter robot model supporting FK and geometric Jacobian.

    Parameters
    - dh_params: array-like of shape (n, 4) with columns [a, alpha, d, theta]
      using the standard Denavit–Hartenberg convention. Angles in radians.
    - joint_types: optional iterable of length n with values 'R' (revolute)
      or 'P' (prismatic). Defaults to all 'R'. Case-insensitive.
    - base: 4x4 homogeneous transform from world to joint-0 frame. Defaults I.
    - tool: 4x4 homogeneous transform from last link frame to tool frame. I.

    Notes
    - For revolute joints, configuration q adds to theta: theta_i + q_i.
    - For prismatic joints, configuration q adds to d: d_i + q_i.

    Example
    -------
    >>> dh = np.array([
    ...     [0.0, 0.0, 0.3, 0.0],  # a, alpha, d, theta
    ...     [0.2, 0.0, 0.0, 0.0],
    ... ])
    >>> robot = Robot(dh)
    >>> q = np.array([np.pi/4, -np.pi/6])
    >>> T = robot.forward_kinematics(q)
    >>> J = robot.jacobian(q)
    """

    def __init__(
        self,
        dh_params: Sequence[DHLink],
        base: Optional[np.ndarray] = None,
        tool: Optional[np.ndarray] = None,
        taskspace: Optional[TaskSpace] = TaskSpace.X | TaskSpace.Y | TaskSpace.Z | TaskSpace.SO3,
    ) -> None:
        # Require typed DHLink sequence
        if not isinstance(dh_params, (list, tuple)) or len(dh_params) == 0 or not all(isinstance(ln, DHLink) for ln in dh_params):
            raise ValueError("dh_params must be a non-empty sequence of DHLink instances")
        links: Sequence[DHLink] = dh_params
        self._n = len(links)
        self._dh = np.array([[ln.a, ln.alpha, ln.d, ln.theta] for ln in links], dtype=float)
        self._joint_types: List[JointType] = [ln.joint_type for ln in links]
        self._joint_limits: np.ndarray = np.array([(ln.lower_limit, ln.upper_limit) for ln in links], dtype=float)

        self._base: np.ndarray = _ensure_4x4(np.eye(4) if base is None else np.asarray(base, dtype=float))
        self._tool: np.ndarray = _ensure_4x4(np.eye(4) if tool is None else np.asarray(tool, dtype=float))
        self._taskspace: Optional[TaskSpace] = taskspace

    @property
    def num_joints(self) -> int:
        return self._n

    # Alias commonly used in kinematics literature
    @property
    def n(self) -> int:
        return self._n

    @property
    def max_manifs(self) -> int:
        """Heuristic upper bound on number of manifolds to discover."""
        if self._n == 3:
            return 2
        if self._n == 4:
            return 4
        return 16

    def sample_random_q(self) -> np.ndarray:
        """Return a random joint vector (n,1). Revolute in [-pi,pi], prismatic in [-0.5,0.5]."""
        return np.random.uniform(self._joint_limits[:, 0], self._joint_limits[:, 1], size=(self._n,))

    @staticmethod
    def dh_transform(a: float, alpha: float, d: float, theta: float) -> np.ndarray:
        """Compute standard DH transform A_i (i-1 -> i)."""
        ca, sa = np.cos(alpha), np.sin(alpha)
        ct, st = np.cos(theta), np.sin(theta)
        return np.array([
            [ct, -st * ca,  st * sa, a * ct],
            [st,  ct * ca, -ct * sa, a * st],
            [0.0,     sa,       ca,      d],
            [0.0,    0.0,      0.0,    1.0],
        ], dtype=float)

    def _link_transform(self, index: int, q_i: float) -> np.ndarray:
        a, alpha, d, theta = self._dh[index]
        if self._joint_types[index] == JointType.REVOLUTE:
            theta = theta + q_i
        elif self._joint_types[index] == JointType.PRISMATIC:
            d = d + q_i
        else:
            raise ValueError(f"Invalid joint type: {self._joint_types[index]}")
        return Robot.dh_transform(a, alpha, d, theta)

    def _forward_chain(self, q: np.ndarray) -> List[np.ndarray]:
        """Compute cumulative transforms T_0_i for i=0..n.

        Returns list with length n+1: [T_0_0, T_0_1, ..., T_0_n].
        """
        if q.shape != (self._n,):
            raise ValueError(f"q must have shape ({self._n},)")
        Ts: List[np.ndarray] = [self._base]
        T = self._base
        for i in range(self._n):
            A_i = self._link_transform(i, float(q[i]))
            T = T @ A_i
            Ts.append(T)
        return Ts

    def forward_kinematics(
        self,
        q: np.ndarray | Sequence[float],
        include_intermediate: bool = False,
        base: Optional[np.ndarray] = None,
        tool: Optional[np.ndarray] = None,
    ) -> np.ndarray | Tuple[np.ndarray, List[np.ndarray]]:
        """Compute base-to-tool homogeneous transform for configuration q.

        If include_intermediate=True, also returns the list of cumulative
        transforms [T_0_0, ..., T_0_n] (without tool applied).
        """
        q_arr = np.asarray(q, dtype=float).reshape(-1)
        if base is not None:
            orig_base = self._base
            self._base = _ensure_4x4(np.asarray(base, dtype=float))
        else:
            orig_base = None
        if tool is not None:
            orig_tool = self._tool
            self._tool = _ensure_4x4(np.asarray(tool, dtype=float))
        else:
            orig_tool = None
        try:
            Ts = self._forward_chain(q_arr)
            T_0_n = Ts[-1]
            T = T_0_n @ self._tool
        finally:
            if orig_base is not None:
                self._base = orig_base
            if orig_tool is not None:
                self._tool = orig_tool

        if include_intermediate:
            return T, Ts
        return T

    def workspace_limits(self, seeds: int = 10) -> dict[TaskSpace, tuple[float, float]]:
        """Return workspace limits for each task space component."""
        limits = {}
        def _solve_for_limit(fk: Callable[[np.ndarray], float], axis: TaskSpace) -> tuple[float, float]:
            min_val = np.inf
            max_val = -np.inf

            for _ in range(seeds):
                q0 = self.sample_random_q()
                min_res = minimize(fk, x0=q0, bounds=self._joint_limits, method="L-BFGS-B")
                min_val = min(min_val, min_res.fun)

                q0 = self.sample_random_q()
                max_res = minimize(lambda q: -fk(q), x0=q0, bounds=self._joint_limits, method="L-BFGS-B")
                max_val = max(max_val, -max_res.fun)

            return min_val, max_val
        
        if TaskSpace.X in self._taskspace:
            limits[TaskSpace.X] = _solve_for_limit(lambda q: self.forward_kinematics(q)[0, 3], TaskSpace.X)
        if TaskSpace.Y in self._taskspace:
            limits[TaskSpace.Y] = _solve_for_limit(lambda q: self.forward_kinematics(q)[1, 3], TaskSpace.Y)
        if TaskSpace.Z in self._taskspace and self._joint_types[2] == JointType.REVOLUTE:
            limits[TaskSpace.Z] = _solve_for_limit(lambda q: self.forward_kinematics(q)[2, 3], TaskSpace.Z)

        return limits

    @staticmethod
    def _so3_log(R: np.ndarray) -> np.ndarray:
        """Return vee(log(R)) for R in SO(3) as a 3-vector.

        Robust to small angles.
        """
        # Clamp trace for numerical stability
        tr = np.trace(R)
        cos_theta = (tr - 1.0) / 2.0
        cos_theta = np.clip(cos_theta, -1.0, 1.0)
        theta = np.arccos(cos_theta)
        if np.isclose(theta, 0.0):
            return 0.5 * np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]])
        else:
            return (
                theta
                / (2.0 * np.sin(theta))
                * np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]])
            )

    def fk_err(
        self,
        q: np.ndarray | Sequence[float],
        x_d: np.ndarray,
    ) -> np.ndarray:
        """Task-space error dx (masked).

        - x_d: desired 4x4 transform
        - mask: True -> use default 6D mask; sequence of 6 bools -> custom; False -> full 6D

        Returns column vector with dimension m x 1 where m is number of selected rows.
        """
        if x_d.shape != (4, 4):
            raise ValueError("x_d must be 4x4")

        q_arr = np.asarray(q, dtype=float).reshape(-1)
        if q_arr.shape != (self._n,):
            raise ValueError(f"q must have shape ({self._n},)")

        # Current pose at q
        T = self.forward_kinematics(q_arr)  # 4x4
        R_c = T[:3, :3]
        p_c = T[:3, 3]
        R_d = x_d[:3, :3]
        p_d = x_d[:3, 3]

        # Position error (desired - current)
        e_p = p_d - p_c
        # Orientation error via log map: e_o = log(R_c^T R_d)
        R_err = R_c.T @ R_d
        e_o = Robot._so3_log(R_err)
        dx6 = np.concatenate([e_p, e_o], axis=0)

        mask_vec = self._taskspace.to_mask()
        return dx6[mask_vec]

    def jacobian(
        self,
        q: np.ndarray | Sequence[float],
        base: Optional[np.ndarray] = None,
        tool: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """Compute 6xN geometric Jacobian at the tool frame origin.

        Stacks [Jv; Jw] where Jv,Jw are 3xN. Columns are expressed in the
        base/world frame. Uses the standard cross-product formulation.
        """
        q_arr = np.asarray(q, dtype=float).reshape(-1)
        if q_arr.shape != (self._n,):
            raise ValueError(f"q must have shape ({self._n},)")

        if base is not None:
            orig_base = self._base
            self._base = _ensure_4x4(np.asarray(base, dtype=float))
        else:
            orig_base = None
        if tool is not None:
            orig_tool = self._tool
            self._tool = _ensure_4x4(np.asarray(tool, dtype=float))
        else:
            orig_tool = None

        try:
            Ts = self._forward_chain(q_arr)  # [T_0_0, ..., T_0_n]
            T_0_n = Ts[-1]
            T_0_ee = T_0_n @ self._tool
            p = T_0_ee[:3, 3]

            Jv = np.zeros((3, self._n), dtype=float)
            Jw = np.zeros((3, self._n), dtype=float)

            for i in range(self._n):
                T_0_im1 = Ts[i]  # base to frame i-1
                z = T_0_im1[:3, 2]
                o = T_0_im1[:3, 3]
                if self._joint_types[i] == JointType.REVOLUTE:
                    Jw[:, i] = z
                    Jv[:, i] = np.cross(z, (p - o))
                else:
                    Jw[:, i] = np.array([0.0, 0.0, 0.0])
                    Jv[:, i] = z

            J = np.vstack((Jv, Jw))
        finally:
            if orig_base is not None:
                self._base = orig_base
            if orig_tool is not None:
                self._tool = orig_tool

        return J

    def masked_jacobian(self, q: np.ndarray | Sequence[float]) -> np.ndarray:
        """Return masked Jacobian (m x n) for configuration q."""
        q_arr = np.asarray(q, dtype=float).reshape(-1)
        J6 = self.jacobian(q_arr)  # 6 x n
        mask_vec = self._taskspace.to_mask()
        return J6[mask_vec, :]

    def null(self, J: np.ndarray) -> np.ndarray:
        """Return a unit right-singular vector spanning the (approximate) nullspace.

        If the nullspace is trivial, returns the right singular vector of the
        smallest singular value.
        """
        _, _, Vh = np.linalg.svd(J, full_matrices=True)
        v_min = Vh[-1, :]
        v = v_min
        nrm = np.linalg.norm(v)
        if nrm == 0.0:
            return np.zeros((J.shape[1],))
        return v / nrm

    def ik(
        self,
        x_d: np.ndarray,
        q0: Optional[np.ndarray | Sequence[float]] = None,
        ik_iters: int = 500,
        dls_lambda: float = 0.1,
        thresh: float = 1e-8,
        rej_limit: int = 50,
    ) -> Tuple[bool, np.ndarray]:
        assert x_d.shape == (4, 4)

        if q0 is None:
            q = self.sample_random_q()
        else:
            q_arr = np.asarray(q0, dtype=float)
            if q_arr.shape == (self._n,):
                q_arr = q_arr.reshape(self._n)
            if q_arr.shape != (self._n,):
                raise ValueError(f"q0 must have shape ({self._n},) or ({self._n},1)")
            q = q_arr

        success = False
        rej_cnt = 0
        dt = 1.0

        for _ in range(ik_iters):
            dx = self.fk_err(q, x_d)
            J = self.masked_jacobian(q)
            m_dim = J.shape[0]
            JJt = J @ J.T
            I = np.eye(m_dim)
            dq = (J.T @ np.linalg.inv(JJt + (dls_lambda * I))) @ dx  # (n x 1)

            q = q + (dq * dt)

            dx_new = self.fk_err(q, x_d)

            new_err = np.linalg.norm(dx_new)
            err = np.linalg.norm(dx)

            if new_err < err:
                dls_lambda /= 2.0
                rej_cnt = 0
            else:
                dls_lambda *= 2.0
                dls_lambda = min(1.0, dls_lambda)
                rej_cnt += 1
                if rej_cnt >= rej_limit:
                    break

            if new_err < thresh:
                success = True
                break

        return success, q

    def smm(
        self,
        x_d: np.ndarray,
        q0: np.ndarray | Sequence[float],
        smm_iters: int = 1000,
        step: float = 0.05,
        samples: int = 128,
        ik_thresh: float = 1e-8,
        sing_thresh: float = 5e-3,
    ) -> SMM:
        
        assert x_d.shape == (4, 4)
        q_arr = np.asarray(q0, dtype=float).reshape(-1)
        if q_arr.shape != (self._n,):
            raise ValueError(f"q0 must have shape ({self._n},)")

        if np.linalg.norm(self.fk_err(q_arr, x_d)) > ik_thresh:
            s_ik, q_arr = self.ik(x_d, q0=q_arr)
            if not s_ik:
                return SMM(
                    status=SMMStatus.NONE,
                    data=None,
                )

        J = self.masked_jacobian(q_arr)
        if np.amin(np.linalg.svd(J, compute_uv=False)) <= sing_thresh:
            return SMM(
                status=SMMStatus.SINGULAR,
                data=np.zeros((samples, self.n)),
            )

        orig_nl_vec = self.null(J)
        prev_nl_vec = np.copy(orig_nl_vec)

        orient = np.zeros((self.n))
        for i in range(self.n):
            Ji = np.delete(J, i, axis=1)
            if Ji.shape[0] == Ji.shape[1] and Ji.shape[0] > 0:
                orient[i] = ((-1.0) ** (i + 1)) * np.linalg.det(Ji)
            else:
                orient[i] = 0.0

        if (orig_nl_vec @ orient) < 0:
            orig_nl_vec = -orig_nl_vec
            prev_nl_vec = -prev_nl_vec

        start_q = q_arr

        smm = np.empty((1, self.n))
        smm[0, :] = np.copy(start_q)

        for _ in range(smm_iters):
            J = self.masked_jacobian(start_q)

            sig = np.amin(np.linalg.svd(J)[1])
            if sig <= sing_thresh:
                return SMM(
                    status=SMMStatus.SINGULAR,
                    data=None,
                )

            nl_vec = self.null(J)
            if (nl_vec @ prev_nl_vec) < 0:
                nl_vec = -nl_vec
            prev_nl_vec = np.copy(nl_vec)

            x_err = self.fk_err(start_q, x_d)

            dq = nl_vec + np.linalg.pinv(J) @ x_err
            dq = (dq / max(1e-12, np.linalg.norm(dq))) * step

            start_q = start_q + dq
            smm = np.vstack((smm, start_q[None, :]))

            # have we returned to the starting configuration?
            err = _tr_inv(_tr(smm[0]) / _tr(start_q))
            if np.linalg.norm(err) < (9.0 / 5.0) * step and smm.shape[0] > 5:
                break
        
        # max iterations reached    
        else:
            return SMM(
                status=SMMStatus.ERROR,
                data=None,
            )

        smm_int = np.empty((samples, self.n))
        for l in range(self.n):
            smm_int[:, l] = np.interp(
                np.linspace(0, smm.shape[0] - 1, samples, endpoint=True),
                np.linspace(0, smm.shape[0] - 1, smm.shape[0], endpoint=True),
                smm[:, l],
            )
        smm_int = _tr_inv(_tr(smm_int))

        return SMM(
            status=SMMStatus.OK,
            data=smm_int,
        )

    def workspace_smms(
        self,
        x_d: np.ndarray,
        samples: int = 128,
        step: float = 0.05,
        sing_thresh: float = 5e-3,
        smm_iters: int = 1000,
        without_new: int = 25,
        max_fails: int = 5,
        diff_thresh: float = 0.15,
        adapt_diff: bool = False,
    ) -> Tuple[SMMStatus, np.ndarray]:
        assert x_d.shape == (4, 4)

        max_manifs = self.max_manifs
        current_without = 0
        fails = 0
        success = SMMStatus.NONE

        smms: list[SMM] = []

        while current_without < without_new:
            q_init = self.sample_random_q()
            s, q_sol = self.ik(x_d, q0=q_init)

            if s:
                fails = 0
                add_manifold = True
                for smm in smms:
                    if adapt_diff:
                        diff_thresh = 2.0 * np.mean(
                            np.linalg.norm(_tr_inv(smm.torus[:, :-1] / smm.torus[:, 1:]), axis=1)
                        )
                    if np.amin(
                        np.linalg.norm(_tr_inv(smm.torus / _tr(q_sol)), axis=1)
                    ) < diff_thresh:
                        add_manifold = False
                        current_without += 1
                        break

                if not add_manifold:
                    continue

                new_smm = self.smm(
                    x_d,
                    q0=q_sol,
                    samples=samples,
                    smm_iters=smm_iters,
                    step=step,
                    sing_thresh=sing_thresh,
                )

                if new_smm.status == SMMStatus.OK:
                    current_without = 0
                    smms.append(new_smm)

                elif new_smm.status == SMMStatus.SINGULAR:
                    return WorkspaceSMMs(
                        status=SMMStatus.SINGULAR,
                        data=[],
                    )

                else:
                    current_without += 1

                if len(smms) > max_manifs:
                    return WorkspaceSMMs(
                        status=SMMStatus.ERROR,
                        data=[],
                    )

            else:
                fails += 1
                if fails > max_fails:
                    break

        if len(smms) > 0 and len(smms) <= max_manifs:
            success = SMMStatus.OK
        
        return WorkspaceSMMs(
            status=success,
            data=smms,
        )


if __name__ == "__main__":
    # Build a simple 3R planar arm (single degree of redundancy for 3D position task)
    taskspace = TaskSpace.X | TaskSpace.Y
    links = [
        DHLink(a=0.30, alpha=0.0, d=0.0, theta=0.0, joint_type=JointType.REVOLUTE),
        DHLink(a=0.25, alpha=0.0, d=0.0, theta=0.0, joint_type=JointType.REVOLUTE),
        DHLink(a=0.20, alpha=0.0, d=0.0, theta=0.0, joint_type=JointType.REVOLUTE),
    ]
    robot = Robot(links, taskspace=taskspace)

    print(robot.workspace_limits())

    # Create a desired pose by forward kinematics at a chosen configuration
    q_des = np.array([0.6, -0.8, 0.3])
    T_des = robot.forward_kinematics(q_des)

    # Start from zero and solve IK
    ok, q_sol = robot.ik(T_des, q0=np.zeros(robot.n), thresh=1e-10)
    err_norm = float(np.linalg.norm(robot.fk_err(q_sol, T_des)))
    print("IK success:", ok)
    print("IK error norm:", err_norm)

    # Jacobian at the solution (masked)
    Jm = robot.masked_jacobian(q_sol)
    svals = np.linalg.svd(Jm, compute_uv=False)
    print("Masked Jacobian shape:", Jm.shape)
    print("Min singular value:", float(np.min(svals)))

    # Trace a single self-motion manifold (SSM)
    smm = robot.smm(T_des, q0=q_sol, samples=64, step=0.05, sing_thresh=5e-3)
    print("SSM status:", smm.status)
    if smm.data is not None:
        print("SSM points shape:", smm.data.shape)
        # Plot 2D projection onto theta2 (joint 2) vs theta3 (joint 3) if available
        try:
            import matplotlib.pyplot as plt
            if smm.data.shape[1] >= 3:
                theta2 = smm.data[:, 1]
                theta3 = smm.data[:, 2]
                plt.figure()
                plt.plot(theta2, theta3, "-o", markersize=3)
                plt.xlabel("theta2 (rad)")
                plt.ylabel("theta3 (rad)")
                plt.title("SSM projection onto (theta2, theta3)")
                plt.grid(True)
                plt.show()
        except Exception as e:
            print("Matplotlib plot failed:", str(e))

    # Discover multiple manifolds around the target
    workspace_smms = robot.workspace_smms(T_des, samples=32, step=0.05, sing_thresh=5e-3, without_new=5)
    print("Workspace SSMs status:", workspace_smms.status)
    try:
        num = len(workspace_smms.data) if workspace_smms.data is not None else 0
        print("Number of manifolds:", num)
    except Exception:
        pass


