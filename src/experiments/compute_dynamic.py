import pandas as pd
import matplotlib.pyplot as plt

INPUT_FILE = "experiment_results/dynamic_comparison.csv"

df = pd.read_csv(INPUT_FILE)

df["success"] = df["success"].astype(int)

# Success rate over ALL runs
success_summary = (
    df.groupby(["size", "algorithm"], as_index=False)
      .agg(success_rate=("success", "mean"))
)

# Only successful runs for the remaining metrics
successful = df[df["success"] == 1]

metrics_summary = (
    successful.groupby(["size", "algorithm"], as_index=False)
    .agg(
        avg_steps=("steps", "mean"),
        avg_collisions=("obstacle_collisions", "mean"),
        avg_bonuses=("bonuses_collected", "mean")
    )
)

summary = success_summary.merge(
    metrics_summary,
    on=["size", "algorithm"],
    how="left"
)

summary.to_csv("experiment_results/results/dynamic_comparison.csv")

print(summary)

algorithms = ["A*", "Q", "R"]
labels = {
    "A*": "A*",
    "Q": "Q-learning",
    "R": "Reactive"
}

linestyles = {
    "A*": "-",
    "Q": (0, (3, 1, 1, 1)),
    "R": ":"
}

markers = {
    "A*": "o",
    "Q": "P",
    "R": "^"
}

fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharex=True)

plots = [
    ("success_rate", "Success Rate"),
    ("avg_steps", "Average Steps"),
    ("avg_collisions", "Average Obstacle Collisions"),
    ("avg_bonuses", "Average Bonuses Collected")
]

for ax, (metric, ylabel) in zip(axes.flat, plots):

    for algo in algorithms:
        data = (
            summary[summary["algorithm"] == algo]
            .sort_values("size")
        )

        ax.plot(
            data["size"],
            data[metric],
            marker=markers[algo],
            linestyle=linestyles[algo],
            linewidth=2,
            label=labels[algo]
        )

    ax.set_xlabel("Maze Size")
    ax.set_ylabel(ylabel)
    ax.set_title(ylabel)
    ax.grid(True)

axes[0, 0].legend(loc="best")

plt.tight_layout()
plt.savefig("experiment_results/figures/dynamic_comparison.png", dpi=300, bbox_inches="tight")
plt.show()