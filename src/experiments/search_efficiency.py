import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import csv
from maze_generator import create_maze
from algorithms.bfs import bfs
from algorithms.dfs import dfs
from algorithms.dijkstra import dijkstra
from algorithms.astar import astar

# size test
SIZES = [21, 41, 61, 81]
DENSITY = 0.75

# density test
SIZE = 61
DENSITIES = [0.25, 0.5, 0.75, 1]

NUM_MAZES = 100

def size_test(sizes=SIZES, density=DENSITY, num_mazes=NUM_MAZES, file="experiment_results/size_efficiency.csv"):
    with open(file, "w", newline="") as f:
        writer = csv.writer(f)

        writer.writerow([
            "maze_id",
            "size",
            "density",
            "algorithm",
            "runtime",
            "nodes_expanded",
            "max_frontier_size"
        ])
    
    f.close()
    
    with open(file, "a", newline="") as f:
        writer = csv.writer(f)

        for size in sizes:
            for maze_id in range(num_mazes):
                maze, start, finish = create_maze(size, size, density, min_sf_distance=size // 2)

                stats = bfs(maze, start, finish, mode="test")
                writer.writerow([
                    maze_id,
                    size,
                    density,
                    "BFS",
                    stats["runtime"],
                    stats["nodes_expanded"],
                    stats["max_frontier_size"]
                ])

                stats = dfs(maze, start, finish, mode="test")
                writer.writerow([
                    maze_id,
                    size,
                    density,
                    "DFS",
                    stats["runtime"],
                    stats["nodes_expanded"],
                    stats["max_frontier_size"]
                ])

                stats = dijkstra(maze, start, finish, mode="test")
                writer.writerow([
                    maze_id,
                    size,
                    density,
                    "Dijkstra",
                    stats["runtime"],
                    stats["nodes_expanded"],
                    stats["max_frontier_size"]
                ])

                stats = astar(maze, start, finish, mode="test")
                writer.writerow([
                    maze_id,
                    size,
                    density,
                    "A*",
                    stats["runtime"],
                    stats["nodes_expanded"],
                    stats["max_frontier_size"]
                ])

def density_test(densities=DENSITIES, size=SIZE, num_mazes=NUM_MAZES,  file="experiment_results/density_efficiency.csv"):
    with open(file, "w", newline="") as f:
        writer = csv.writer(f)

        writer.writerow([
            "maze_id",
            "size",
            "density",
            "algorithm",
            "runtime",
            "nodes_expanded",
            "max_frontier_size"
        ])
    
    with open(file, "a", newline="") as f:
        writer = csv.writer(f)

        for density in densities:
            for maze_id in range(num_mazes):
                maze, start, finish = create_maze(size, size, density, min_sf_distance=size // 2)

                stats = bfs(maze, start, finish, mode="test")
                writer.writerow([
                    maze_id,
                    size,
                    density,
                    "BFS",
                    stats["runtime"],
                    stats["nodes_expanded"],
                    stats["max_frontier_size"]
                ])

                stats = dfs(maze, start, finish, mode="test")
                writer.writerow([
                    maze_id,
                    size,
                    density,
                    "DFS",
                    stats["runtime"],
                    stats["nodes_expanded"],
                    stats["max_frontier_size"]
                ])

                stats = dijkstra(maze, start, finish, mode="test")
                writer.writerow([
                    maze_id,
                    size,
                    density,
                    "Dijkstra",
                    stats["runtime"],
                    stats["nodes_expanded"],
                    stats["max_frontier_size"]
                ])

                stats = astar(maze, start, finish, mode="test")
                writer.writerow([
                    maze_id,
                    size,
                    density,
                    "A*",
                    stats["runtime"],
                    stats["nodes_expanded"],
                    stats["max_frontier_size"]
                ])