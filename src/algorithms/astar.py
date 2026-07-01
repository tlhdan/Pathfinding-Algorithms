import heapq
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


def heuristic(a, b):
    return abs(a[0] - b[0]) + abs(a[1] - b[1])

def reconstruct_path(parent, start, finish):
    if finish != start and finish not in parent:
        return None

    path = [finish]

    while path[-1] != start:
        path.append(parent[path[-1]])

    path.reverse()
    return path

def astar(maze, start, finish, mode="run"):
    start_time = time.perf_counter()

    pq = [(0, start)]
    g_score = {start: 0}
    parent = {}
    visited = set()
    history = []

    nodes_expanded = 0
    max_frontier_size = 1

    while pq:
        frontier = {node for _, node in pq}
        history.append((set(visited), frontier))

        _, current = heapq.heappop(pq)
        nodes_expanded += 1

        if current in visited:
            continue

        visited.add(current)

        if current == finish:
            break

        for neighbor in get_neighbors(maze, current):
            tentative_g = g_score[current] + maze[neighbor]

            if neighbor not in g_score or tentative_g < g_score[neighbor]:
                g_score[neighbor] = tentative_g
                f_score = tentative_g + heuristic(neighbor, finish)

                parent[neighbor] = current
                heapq.heappush(pq, (f_score, neighbor))
            
        max_frontier_size = max(max_frontier_size, len(pq))

    runtime = time.perf_counter() - start_time

    stats = {
        "runtime": runtime,
        "nodes_expanded": nodes_expanded,
        "max_frontier_size": max_frontier_size,
    }

    if mode == "test":
        return stats
    
    elif mode == "path":
        return reconstruct_path(parent, start, finish)

    return history, parent