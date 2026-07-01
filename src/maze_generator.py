import random
import numpy as np


def create_maze(num_rows, num_cols, density=1, weighted=False, min_cost=1, max_cost=50, min_sf_distance=None):
    num_rows, num_cols = (num_rows + 1) // 2, (num_cols + 1) // 2

    maze = np.zeros((num_rows * 2 + 1, num_cols * 2 + 1), dtype=int)

    row = random.randrange(num_rows) * 2 + 1
    col = random.randrange(num_cols) * 2 + 1

    maze[row, col] = 1

    random_dfs(maze, num_rows, num_cols, (row, col))

    remove_wall(maze, 1 - density)

    valid_positions = list(zip(*np.where(maze > 0)))

    start, finish = random.sample(valid_positions, 2)

    if min_sf_distance is not None:
        while True:
            start, finish = random.sample(valid_positions, 2)

            if abs(start[0]-finish[0]) + abs(start[1]-finish[1]) >= min_sf_distance:
                break
    else:
        start, finish = random.sample(valid_positions, 2)

    if weighted:
        maze = add_random_weights(maze, min_cost, max_cost)

    return maze, start, finish


def random_dfs(maze, num_rows, num_cols, start):
    stack = [start]

    while stack:
        r, c = stack[-1]

        directions = [(-2, 0), (2, 0), (0, -2), (0, 2)]
        random.shuffle(directions)

        carved = False

        for dr, dc in directions:
            nr, nc = r + dr, c + dc

            if 1 <= nr < num_rows * 2 and 1 <= nc < num_cols * 2:

                if maze[nr, nc] == 0:
                    maze[(r + nr) // 2, (c + nc) // 2] = 1
                    maze[nr, nc] = 1

                    stack.append((nr, nc))

                    carved = True
                    break

        if not carved:
            stack.pop()


def remove_wall(maze, chance):
    rows, cols = maze.shape

    for r in range(1, rows - 1):
        for c in range(1, cols - 1):

            if maze[r, c] == 0:

                if not (r % 2 == 1 and c % 2 == 1):

                    if random.random() < chance:
                        maze[r, c] = 1


def generate_obstacles(maze, start, finish, num_obstacles=0):
    rows, cols = maze.shape

    directions = [
        (-1, 0),
        (0, 1),
        (1, 0),
        (0, -1),
        (0, 0)
    ]

    obstacles = {}

    obstacle_id = 0

    while len(obstacles) < num_obstacles:
        r = random.randrange(1, rows - 1)
        c = random.randrange(1, cols - 1)

        if maze[r, c] == 0:
            continue

        if (r, c) == start or (r, c) == finish:
            continue

        if any(obstacle_data["pos"] == (r, c) for obstacle_data in obstacles.values()):
            continue

        obstacles[obstacle_id] = {
            "pos": (r, c),
            "dir": random.choice(directions)
        }

        obstacle_id += 1

    return obstacles


def move_obstacles(maze, obstacles):
    new_obstacles = {}
    
    for obstacle_id, obstacle_data in obstacles.items():
        r, c = obstacle_data["pos"]
        dr, dc = obstacle_data["dir"]

        nr, nc = r + dr, c + dc

        if maze[nr, nc] == 0:
            dr, dc = -dr, -dc

            nr, nc = r + dr, c + dc
        
            if maze[nr, nc] == 0:
                nr, nc = r, c
    
        new_obstacles[obstacle_id] = {
            "pos": (nr, nc),
            "dir": (dr, dc)
        }

    return new_obstacles


def generate_bonuses(maze, start, finish, num_bonuses=0):
    rows, cols = maze.shape

    bonuses = set()

    while len(bonuses) < num_bonuses:
        r = random.randrange(1, rows - 1)
        c = random.randrange(1, cols - 1)

        if maze[r, c] == 0:
            continue

        if (r, c) == start or (r, c) == finish:
            continue

        if (r, c) in bonuses:
            continue

        bonuses.add((r, c))

    return bonuses

def add_random_weights(maze, min_cost=1, max_cost=50):
    weighted_maze = maze.copy()

    rows, cols = maze.shape

    for r in range(rows):
        for c in range(cols):

            if maze[r, c] > 0:
                weighted_maze[r, c] = random.randint(min_cost, max_cost)

    return weighted_maze