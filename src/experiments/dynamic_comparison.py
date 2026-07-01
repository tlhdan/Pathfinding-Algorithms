import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import csv
import random
import numpy as np
from concurrent.futures import ProcessPoolExecutor

from maze_generator import (
    create_maze,
    generate_bonuses,
    generate_obstacles,
)

from algorithms.qlearning_dynamic import q_learning, get_path
from algorithms.astar_replanning import astar_replanning
from algorithms.reactive import reactive_agent


SIZES = [11, 21, 31]
DENSITY = 0.8
NUM_BONUSES = 5
NUM_OBSTACLES = 5
NUM_MAZES = 20
NUM_RUNS = 5


def worker(args):
    (
        algo,
        size,
        maze_id,
        run_id,
        density,
        num_bonuses,
        num_obstacles,
    ) = args

    seed = size * 1000 + maze_id * 100 + run_id
    random.seed(seed)
    np.random.seed(seed)

    maze, start, finish = create_maze(
        size,
        size,
        density,
        min_sf_distance=size // 2
    )

    bonuses = generate_bonuses(maze, start, finish, num_bonuses)
    obstacles = generate_obstacles(maze, start, finish, num_obstacles)

    if algo == "Q":
        Q = q_learning(
            maze,
            start,
            finish,
            bonuses,
            obstacles,
            mode="run"
        )

        result = get_path(Q, maze, start, finish, bonuses, obstacles)

    elif algo == "A*":
        result = astar_replanning(
            maze,
            start,
            finish,
            bonuses,
            obstacles
        )

    elif algo == "R":
        result = reactive_agent(
            maze,
            start,
            finish,
            bonuses,
            obstacles
        )

    return [
        algo,
        size,
        maze_id,
        run_id,
        result["success"],
        result["steps"],
        result["bonuses_collected"],
        result["obstacle_collisions"]
    ]


def run_experiment(
    file="experiment_results/dynamic_comparison.csv"
):

    tasks = []

    algos = ["Q", "A*", "R"]

    for algo in algos:
        for size in SIZES:
            for maze_id in range(NUM_MAZES):
                for run_id in range(NUM_RUNS):
                    tasks.append((
                        algo,
                        size,
                        maze_id,
                        run_id,
                        DENSITY,
                        NUM_BONUSES,
                        NUM_OBSTACLES
                    ))

    with open(file, "w", newline="") as f:
        writer = csv.writer(f)

        writer.writerow([
            "algorithm",
            "size",
            "maze_id",
            "run_id",
            "success",
            "steps",
            "bonuses_collected",
            "obstacle_collisions"
        ])

        with ProcessPoolExecutor(max_workers=6) as executor:
            for row in executor.map(worker, tasks):
                writer.writerow(row)


if __name__ == "__main__":
    pass