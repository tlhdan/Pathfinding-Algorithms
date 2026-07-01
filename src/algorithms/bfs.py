from collections import deque
import time


def get_neighbors(maze, node):
    r, c = node

    directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]

    neighbors = []

    for dr, dc in directions:
        nr, nc = r + dr, c + dc

        if 0 <= nr < maze.shape[0] and 0 <= nc < maze.shape[1]:
            if maze[nr, nc] != 0:
                neighbors.append((nr, nc))

    return neighbors


def bfs(maze, start, finish, mode="run"):
    start_time = time.perf_counter()

    queue = deque([start])
    visited = {start}
    parent = {}
    history = []

    nodes_expanded = 0
    max_frontier_size = 1

    while queue:
        history.append((set(visited), set(queue)))

        current = queue.popleft()
        nodes_expanded += 1

        if current == finish:
            break

        for neighbor in get_neighbors(maze, current):
            if neighbor not in visited:
                visited.add(neighbor)
                parent[neighbor] = current
                queue.append(neighbor)
        
        max_frontier_size = max(max_frontier_size, len(queue))
    
    runtime = time.perf_counter() - start_time

    stats = {
        "runtime": runtime,
        "nodes_expanded": nodes_expanded,
        "max_frontier_size": max_frontier_size,
    }

    if mode == "test":
        return stats

    return history, parent