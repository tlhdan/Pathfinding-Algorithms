import pandas as pd
import numpy as np

FILE = "experiment_results/qlearning_dynamic_training.csv"
WINDOW = 3

df = pd.read_csv(FILE)

results = []

for (size, maze_id, run_id), run in df.groupby(["size","maze_id","run_id"]):

    run = run.sort_values("episode")

    costs = run["eval_reward"].to_numpy()
    success = run["eval_success"].to_numpy()
    episodes = run["episode"].to_numpy()

    converged = False
    convergence_episode = None

    for i in range(len(costs)-WINDOW+1):

        if np.all(success[i:i+WINDOW] == 1):

            if np.all(costs[i:i+WINDOW] == costs[i]):

                converged = True
                convergence_episode = episodes[i]
                break

    results.append({
        "size": size,
        "maze_id": maze_id,
        "run_id": run_id,
        "converged": converged,
        "episode": convergence_episode
    })

results = pd.DataFrame(results)

summary = results.groupby("size").agg(
    runs=("converged", "count"),
    converged=("converged", "sum"),
    mean_episode=("episode", "mean"),
    median_episode=("episode", "median"),
    std_episode=("episode", "std"),
)

summary["convergence_%"] = 100 * summary["converged"] / summary["runs"]

summary.to_csv("experiment_results/qlearning_dynamic_convergence.csv")

print(summary)