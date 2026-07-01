import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

INPUT_FILE = "experiment_results/path_optimality.csv"

df = pd.read_csv(INPUT_FILE)

df["success"] = df["success"].astype(int)

# Best run for each maze/algorithm
best = (
    df.sort_values(["size", "maze_id", "algorithm", "path_cost"])
    .groupby(["size", "maze_id", "algorithm"], as_index=False)
    .first()
)

# Success rate (ALL runs)
success_summary = (
    df.groupby(["size", "algorithm"], as_index=False)
      .agg(success_rate=("success", "mean"))
)

# Keep only successful runs for path quality metrics
best_success = best[best["success"] == 1].copy()

baseline = (
    best_success.groupby(["size", "maze_id"], as_index=False)
    .agg(
        best_cost=("path_cost", "min"),
        best_length=("path_length", "min")
    )
)

merged = best_success.merge(baseline, on=["size", "maze_id"])

merged["cost_ratio"] = merged["path_cost"] / merged["best_cost"]
merged["length_ratio"] = merged["path_length"] / merged["best_length"]

quality_summary = (
    merged.groupby(["size", "algorithm"], as_index=False)
    .agg(
        avg_cost_ratio=("cost_ratio", "mean"),
        std_cost_ratio=("cost_ratio", "std"),
        avg_length_ratio=("length_ratio", "mean")
    )
)

summary = quality_summary.merge(
    success_summary,
    on=["size", "algorithm"]
)

summary.to_csv("experiment_results/results/path_optimality.csv")

print(summary)

algorithms = ["A*", "BFS", "DFS", "Dijkstra", "Q-learning"]

linestyles = {
    "A*": "-",
    "BFS": "--",
    "DFS": ":",
    "Dijkstra": "-.",
    "Q-learning": (0, (3, 1, 1, 1))
}

markers = {
    "A*": "o",
    "BFS": "s",
    "DFS": "^",
    "Dijkstra": "D",
    "Q-learning": "P"
}

fig, axes = plt.subplots(1, 3, figsize=(15, 4), sharex=True)

plots = [
    ("avg_cost_ratio", "std_cost_ratio", "Average Cost Ratio"),
    ("avg_length_ratio", None, "Average Path Length Ratio"),
    ("success_rate", None, "Success Rate"),
]

for ax, (metric, std_metric, ylabel) in zip(axes, plots):

    for algo in algorithms:
        data = (
            summary[summary["algorithm"] == algo]
            .sort_values("size")
        )

        if std_metric is None:
            ax.plot(
                data["size"],
                data[metric],
                label=algo,
                marker=markers[algo],
                linestyle=linestyles[algo],
                linewidth=2
            )
        else:
            ax.errorbar(
                data["size"],
                data[metric],
                yerr=data[std_metric],
                label=algo,
                marker=markers[algo],
                linestyle=linestyles[algo],
                linewidth=2,
                capsize=4
            )

    ax.set_xlabel("Maze Size")
    ax.set_ylabel(ylabel)
    ax.set_title(ylabel)
    ax.grid(True)

axes[0].legend(loc="best")

plt.tight_layout()
plt.savefig("experiment_results/figures/path_optimality.png", dpi=300, bbox_inches="tight")
plt.show()