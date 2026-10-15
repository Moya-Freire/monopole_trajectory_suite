import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter

from Monopole import Monopole


def animate_monopoles(positions, directions, energies, field,
                      filename="monopoles.gif", interval=100,
                      arrow_length=40):

    colors = ["red", "blue", "green"]
    n_monopoles = len(positions)

    positions = [np.asarray(p) for p in positions]
    directions = [np.asarray(d) for d in directions]
    energies = [np.asarray(e) for e in energies]

    n_frames = min(
        min(len(p) for p in positions),
        min(len(d) for d in directions),
        min(len(e) for e in energies)
    )

    positions = [p[:n_frames] for p in positions]
    directions = [d[:n_frames] for d in directions]
    energies = [e[:n_frames] for e in energies]

    initial_energies = [e[0] for e in energies]

    fig, ax = plt.subplots(figsize=(12, 7))
    fig.subplots_adjust(right=0.72)

    # Uniform magnetic field arrows in the x-y plane.
    xg, yg = np.meshgrid(
        np.linspace(-100, 1100, 9),
        np.linspace(-100, 1100, 9)
    )

    B = np.asarray(field, dtype=float)[:2]
    B_norm = np.linalg.norm(B)

    if B_norm > 0:
        B = B / B_norm
        ax.quiver(
            xg, yg,
            np.full_like(xg, B[0]),
            np.full_like(yg, B[1]),
            color="gray",
            alpha=0.3,
            angles="xy",
            scale_units="xy",
            scale=0.04
        )

    # Complete trajectories as faint reference lines.
    for i in range(n_monopoles):
        ax.plot(
            positions[i][:, 0],
            positions[i][:, 1],
            color=colors[i],
            alpha=0.2
        )

    trails = []
    balls = []
    arrows = []

    for i in range(n_monopoles):
        trail, = ax.plot([], [], color=colors[i], lw=2)

        ball, = ax.plot(
            [], [], "o",
            color=colors[i],
            markersize=9,
            label=f"Monopole {i+1}"
        )

        arrow = ax.quiver(
            [0], [0], [1], [0],
            color=colors[i],
            angles="xy",
            scale_units="xy",
            scale=1,
            width=0.006
        )

        trails.append(trail)
        balls.append(ball)
        arrows.append(arrow)

    # Animated kinetic-energy readout outside the axes.
    energy_text = fig.text(
        0.76, 0.85, "",
        verticalalignment="top",
        family="monospace",
        fontsize=9,
        bbox=dict(
            facecolor="white",
            alpha=0.9,
            edgecolor="gray"
        )
    )

    all_xy = np.concatenate(
        [p[:, :2] for p in positions], axis=0
    )

    margin = max(
        50,
        0.05 * max(
            np.ptp(all_xy[:, 0]),
            np.ptp(all_xy[:, 1]),
            1
        )
    )

    ax.set_xlim(
        all_xy[:, 0].min() - margin,
        all_xy[:, 0].max() + margin
    )
    ax.set_ylim(
        all_xy[:, 1].min() - margin,
        all_xy[:, 1].max() + margin
    )

    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.set_title("Monopole trajectories")
    ax.legend(loc="upper right")
    ax.grid(alpha=0.2)

    def update(frame):
        for i in range(n_monopoles):
            path = positions[i][:frame + 1]
            pos = positions[i][frame]
            d = directions[i][frame]

            trails[i].set_data(path[:, 0], path[:, 1])
            balls[i].set_data([pos[0]], [pos[1]])

            arrows[i].set_offsets([pos[:2]])
            arrows[i].set_UVC(
                arrow_length * d[0],
                arrow_length * d[1]
            )

        lines = []

        for i in range(n_monopoles):
            kinetic = energies[i][frame]
            delta = kinetic - initial_energies[i]

            lines.append(
                f"M{i+1}: Ek = {kinetic:.6e}\n"
                f"    dEk = {delta:+.6e}"
            )

        energy_text.set_text("\n".join(lines))
        ax.set_title(f"Monopole trajectories — step {frame}")

        return trails + balls + arrows + [energy_text]

    animation = FuncAnimation(
        fig,
        update,
        frames=n_frames,
        interval=interval,
        blit=False,
        repeat=False
    )

    animation.save(
        filename,
        writer=PillowWriter(
            fps=max(1, round(1000 / interval))
        )
    )

    plt.close(fig)
    return filename