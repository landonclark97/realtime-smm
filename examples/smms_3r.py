import matplotlib.pyplot as plt
import numpy as np

from realtime_smm import training_pipeline, load_trained_bundle, clear_cache
from realtime_smm import Robot, TaskSpace, SMMSolverParams, DHLink, JointType, GridParams
from realtime_smm import TrainingConfig


def _plot_overlays(robot: Robot, bundle, T: np.ndarray, smm_params: SMMSolverParams) -> None:
    exact = robot.workspace_smms(
        T,
        samples=1000,
        step=smm_params.step,
        sing_thresh=1e-3,
        smm_iters=smm_params.smm_iters,
    )

    predicted = bundle(T, samples=5000)

    def _branches(ws):
        return [branch.angle.astype(np.float32, copy=False) for branch in ws.data if branch.angle is not None]

    exact_branches = _branches(exact)
    predicted_branches = _branches(predicted)

    for pred_branch in predicted_branches:
        diff = np.abs(pred_branch[1:] - pred_branch[:-1])
        mask = np.amax(diff, axis=1) > np.pi
        pred_branch[:-1][mask] = np.nan

    for exact_branch in exact_branches:
        diff = np.abs(exact_branch[1:] - exact_branch[:-1])
        mask = np.amax(diff, axis=1) > np.pi
        exact_branch[:-1][mask] = np.nan

    fig, axes = plt.subplots(1, 2, figsize=(10, 4), tight_layout=True)
    pairs = ((0, 1, r"$\theta_1$ vs $\theta_2$"), (1, 2, r"$\theta_2$ vs $\theta_3$"))

    for idx, branch in enumerate(exact_branches):
        for ax, (i, j, label) in zip(axes, pairs):
            ax.plot(
                branch[:, i],
                branch[:, j],
                color="dodgerblue",
                linewidth=6.0,
                zorder=1,
                alpha=0.2,
                label="Exact" if idx == 0 else None,
            )

    for idx, branch in enumerate(predicted_branches):
        for ax, (i, j, label) in zip(axes, pairs):
            ax.plot(
                branch[:, i],
                branch[:, j],
                color="dodgerblue",
                linewidth=1.5,
                zorder=2,
                label="Bundle" if idx == 0 else None,
            )

    for ax, (_, _, label) in zip(axes, pairs):
        ax.set_xlabel(label.split(" vs ")[0])
        ax.set_ylabel(label.split(" vs ")[1])
        ax.set_xlim(-np.pi, np.pi)
        ax.set_ylim(-np.pi, np.pi)
        ax.grid(True, alpha=0.3)

    axes[0].legend(loc="best")
    plt.suptitle("SMM Overlay")
    plt.show()


def main():

    clear_cache()

    robot = Robot(
        [
            DHLink(a=0.20, alpha=0.0, d=0.0, theta=0.0, joint_type=JointType.REVOLUTE),
            DHLink(a=0.25, alpha=0.0, d=0.0, theta=0.0, joint_type=JointType.REVOLUTE),
            DHLink(a=0.35, alpha=0.0, d=0.0, theta=0.0, joint_type=JointType.REVOLUTE),
        ],
        taskspace=TaskSpace.X | TaskSpace.Y,
    )
    grid_params = GridParams(pos_resolution=0.005, use_xy_halfplane=True)
    smm_solver_params = SMMSolverParams(samples=64, step=0.0075, sing_thresh=3e-2, smm_iters=2500)
    training_config = TrainingConfig(epochs=5000, learning_rate=0.001, weight_decay=0.0001, fft_cutoff=20)

    training_pipeline(robot, grid_params=grid_params, smm_params=smm_solver_params, training_config=training_config, name="3r_smms")
    bundle = load_trained_bundle(name="3r_smms")

    T = np.eye(4)
    T[0, 3] = 0.7
    T[1, 3] = 0.1

    _plot_overlays(robot, bundle, T, smm_params=smm_solver_params)

if __name__ == "__main__":
    main()