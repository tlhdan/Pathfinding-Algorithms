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
from algorithms import qlearning_dynamic
from algorithms import qlearning_static

SIZES = [11, 21, 31]
DENSITY = 0.8
MIN_COST = 1
MAX_COST = 50
NUM_BONUSES = 5
NUM_OBSTACLES = 5
NUM_MAZES = 20
NUM_RUNS = 5
EPISODES = 2000
EVAL_INTERVAL = 50
ALPHA = 0.1
GAMMA = 0.99
EPSILON = 1
EPSILON_DECAY = 0.995
EPSILON_MIN = 0.01

def dynamic_worker(args):
    (
        size, maze_id, run_id,
        density, num_bonuses, num_obstacles,
        episodes, eval_interval,
        alpha, gamma,
        epsilon, epsilon_decay, epsilon_min
    ) = args

    maze_seed = size * 1000 + maze_id

    random.seed(maze_seed)
    np.random.seed(maze_seed)

    maze, start, finish = create_maze(
        size,
        size,
        density,
        min_sf_distance=size // 2
    )

    bonuses = generate_bonuses(maze, start, finish, num_bonuses)
    obstacles = generate_obstacles(maze, start, finish, num_obstacles)

    run_seed = maze_seed * 100 + run_id
    random.seed(run_seed)
    np.random.seed(run_seed)

    _, history = qlearning_dynamic.q_learning(
        maze,
        start,
        finish,
        bonuses,
        obstacles,
        mode="test",
        eval_interval=eval_interval,
        episodes=episodes,
        alpha=alpha,
        gamma=gamma,
        epsilon=epsilon,
        epsilon_decay=epsilon_decay,
        epsilon_min=epsilon_min
    )

    rows = []

    for record in history:
        rows.append([
            size,
            maze_id,
            run_id,
            record["episode"],
            record["eval_success"],
            record["eval_step"],
            record["eval_reward"]
        ])

    return rows

def dynamic_test(
    sizes=SIZES,
    density=DENSITY,
    num_bonuses=NUM_BONUSES,
    num_obstacles=NUM_OBSTACLES,
    num_mazes=NUM_MAZES,
    num_runs=NUM_RUNS,
    episodes=EPISODES,
    eval_interval=EVAL_INTERVAL,
    alpha=ALPHA,
    gamma=GAMMA,
    epsilon=EPSILON,
    epsilon_decay=EPSILON_DECAY,
    epsilon_min=EPSILON_MIN,
    file="experiment_results/qlearning_dynamic_training.csv"
):

    tasks = []

    for size in sizes:
        for maze_id in range(num_mazes):
            for run_id in range(num_runs):
                tasks.append((
                    size,
                    maze_id,
                    run_id,
                    density,
                    num_bonuses,
                    num_obstacles,
                    episodes,
                    eval_interval,
                    alpha,
                    gamma,
                    epsilon,
                    epsilon_decay,
                    epsilon_min
                ))

    with open(file, "w", newline="") as f:
        writer = csv.writer(f)

        writer.writerow([
            "size",
            "maze_id",
            "run_id",
            "episode",
            "eval_success",
            "eval_steps",
            "eval_reward"
        ])

        with ProcessPoolExecutor(max_workers=6) as executor:
            for rows in executor.map(dynamic_worker, tasks):
                writer.writerows(rows)

def static_test(
    sizes=SIZES,
    density=DENSITY,
    num_mazes=NUM_MAZES,
    min_cost=MIN_COST,
    max_cost=MAX_COST,
    num_runs=NUM_RUNS,
    episodes=EPISODES,
    eval_interval=EVAL_INTERVAL,
    alpha=ALPHA,
    gamma=GAMMA,
    epsilon=EPSILON,
    epsilon_decay=EPSILON_DECAY,
    epsilon_min=EPSILON_MIN,
    file="experiment_results/qlearning_static_training.csv"
):

    with open(file, "w", newline="") as f:
        writer = csv.writer(f)

        writer.writerow([
            "size",
            "maze_id",
            "run_id",
            "episode",
            "eval_success",
            "eval_cost"
        ])

        for size in sizes:

            for maze_id in range(num_mazes):

                # Generate a reproducible maze
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

                for run_id in range(num_runs):

                    # Independent training run
                    run_seed = maze_seed * 100 + run_id
                    random.seed(run_seed)
                    np.random.seed(run_seed)

                    _, training_history = qlearning_static.q_learning(
                        maze,
                        start,
                        finish,
                        mode="test",
                        eval_interval=eval_interval,
                        episodes=episodes,
                        alpha=alpha,
                        gamma=gamma,
                        epsilon=epsilon,
                        epsilon_decay=epsilon_decay,
                        epsilon_min=epsilon_min
                    )

                    for record in training_history:
                        writer.writerow([
                            size,
                            maze_id,
                            run_id,
                            record["episode"],
                            record["eval_success"],
                            record["eval_cost"]
                        ])

if __name__ == "__main__":
    pass