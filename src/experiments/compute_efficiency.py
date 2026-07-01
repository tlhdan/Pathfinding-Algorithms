import pandas as pd
import matplotlib.pyplot as plt

para = "density"
file = f"experiment_results/{para}_efficiency.csv"
group = [para, "algorithm"]
rows = ["runtime", "nodes_expanded", "max_frontier_size"]

def summarize_results(file, group, metrics, agg="mean"):
    df = pd.read_csv(file)

    return (
        df.groupby(group)[metrics]
        .agg(agg)
        .reset_index()
    )

stats = summarize_results(file, group, rows, agg=["mean", "std"])
stats.to_csv(f"experiment_results/results/efficiency_{para}.csv")
print(stats)

algorithms = stats["algorithm"].unique()
paras = sorted(stats[para].unique())

linestyles = {
    "A*": "-",
    "BFS": "--",
    "DFS": ":",
    "Dijkstra": "-."
}

markers = {
    "A*": "o",
    "BFS": "s",
    "DFS": "^",
    "Dijkstra": "D"
}

def plot_metric(metric, title, ylabel, log_scale=False):
    plt.figure()

    for algo in algorithms:
        subset = stats[stats["algorithm"] == algo].sort_values(para)

        x = subset[para].values
        y = subset[(metric, "mean")].values
        y_std = subset[(metric, "std")].values

        plt.plot(
            x, y,
            label=algo,
            linestyle=linestyles.get(algo, "-"),
            marker=markers.get(algo, "o"),
            linewidth=2
        )
        plt.fill_between(
            x,
            y - y_std,
            y + y_std,
            alpha=0.2
        )

    plt.title(title)
    plt.xlabel(f"Maze {para.capitalize()}")
    plt.ylabel(ylabel)
    plt.legend()

    if log_scale:
        plt.yscale("log")

    plt.grid(True)
    plt.savefig(f"experiment_results/figures/{metric}_vs_{para}.png", dpi=300, bbox_inches="tight")
    plt.show()


plot_metric(
    "runtime",
    f"Runtime vs Maze {para.capitalize()}",
    "Runtime (seconds)",
    log_scale=True
)

plot_metric(
    "nodes_expanded",
    f"Nodes Expanded vs Maze {para.capitalize()}",
    "Nodes Expanded"
)

plot_metric(
    "max_frontier_size",
    f"Max Frontier Size vs Maze {para.capitalize()}",
    "Frontier Size"
)