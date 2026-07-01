import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import csv
import random
import numpy as np

from maze_generator import create_maze
from algorithms.bfs import bfs
from algorithms.dfs import dfs
from algorithms.dijkstra import dijkstra
from algorithms.astar import astar
from algorithms.qlearning_static import q_learning, get_path

SIZES = [11, 21, 31]

DENSITY = 0.8
MIN_COST = 1
MAX_COST = 50

NUM_MAZES = 20
NUM_Q_RUNS = 5

EPISODES = 1000

ALPHA = 0.1
GAMMA = 0.99
EPSILON = 1
EPSILON_DECAY = 0.995
EPSILON_MIN = 0.01

def reconstruct_path(parent, maze, start, finish):

    if finish not in parent and finish != start:
        return None, None

    node = finish
    path = [finish]

    while node != start:
        node = parent[node]
        path.append(node)

    path.reverse()

    cost = sum(maze[cell] for cell in path[1:])

    return path, cost

def optimality_test(
    sizes=SIZES,
    density=DENSITY,
    min_cost=MIN_COST,
    max_cost=MAX_COST,
    num_mazes=NUM_MAZES,
    num_q_runs=NUM_Q_RUNS,
    episodes=EPISODES,
    alpha=ALPHA,
    gamma=GAMMA,
    epsilon=EPSILON,
    epsilon_decay=EPSILON_DECAY,
    epsilon_min=EPSILON_MIN,
    file="experiment_results/path_optimality.csv"
):

    with open(file, "w", newline="") as f:

        writer = csv.writer(f)

        writer.writerow([
            "size",
            "maze_id",
            "algorithm",
            "run_id",
            "success",
            "path_cost",
            "path_length"
        ])

        for size in sizes:

            for maze_id in range(num_mazes):

                maze_seed = size * 1000 + maze_id

                random.seed(maze_seed)
                np.random.seed(maze_seed)

                maze, start, finish = create_maze(
                    size,
                    size,
                    density,
                    weighted=True,
                    min_cost=min_cost,
                    max_cost=max_cost,
                    min_sf_distance=size // 2
                )

                # BFS

                parent = bfs(maze, start, finish)[1]
                path, cost = reconstruct_path(parent, maze, start, finish)

                writer.writerow([
                    size,
                    maze_id,
                    "BFS",
                    0,
                    int(path is not None),
                    cost,
                    len(path)-1 if path else None
                ])

                # DFS

                parent = dfs(maze, start, finish)[1]
                path, cost = reconstruct_path(parent, maze, start, finish)

                writer.writerow([
                    size,
                    maze_id,
                    "DFS",
                    0,
                    int(path is not None),
                    cost,
                    len(path)-1 if path else None
                ])

                # Dijkstra

                parent = dijkstra(maze, start, finish)[1]
                path, cost = reconstruct_path(parent, maze, start, finish)

                writer.writerow([
                    size,
                    maze_id,
                    "Dijkstra",
                    0,
                    int(path is not None),
                    cost,
                    len(path)-1 if path else None
                ])

                # A*

                parent = astar(maze, start, finish)[1]
                path, cost = reconstruct_path(parent, maze, start, finish)

                writer.writerow([
                    size,
                    maze_id,
                    "A*",
                    0,
                    int(path is not None),
                    cost,
                    len(path)-1 if path else None
                ])

                # Q-learning

                for run_id in range(num_q_runs):

                    run_seed = maze_seed * 100 + run_id

                    random.seed(run_seed)
                    np.random.seed(run_seed)

                    Q = q_learning(
                        maze,
                        start,
                        finish,
                        episodes=episodes,
                        alpha=alpha,
                        gamma=gamma,
                        epsilon=epsilon,
                        epsilon_decay=epsilon_decay,
                        epsilon_min=epsilon_min
                    )

                    path, cost, success, _ = get_path(
                        Q,
                        maze,
                        start,
                        finish
                    )

                    writer.writerow([
                        size,
                        maze_id,
                        "Q-learning",
                        run_id,
                        int(success),
                        cost,
                        len(path)-1 if success else None
                    ])