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


def dijkstra(maze, start, finish, mode="run"):
    start_time = time.perf_counter()

    pq = [(0, start)]
    distances = {start: 0}
    parent = {}
    visited = set()
    history = []

    nodes_expanded = 0
    max_frontier_size = 1

    while pq:
        frontier = {node for _, node in pq}
        history.append((set(visited), frontier))

        cost, current = heapq.heappop(pq)
        nodes_expanded += 1

        if current in visited:
            continue

        visited.add(current)

        if current == finish:
            break

        for neighbor in get_neighbors(maze, current):
            new_cost = cost + maze[neighbor]

            if neighbor not in distances or new_cost < distances[neighbor]:
                distances[neighbor] = new_cost
                parent[neighbor] = current
                heapq.heappush(pq, (new_cost, neighbor))
        
        max_frontier_size = max(max_frontier_size, len(pq))
    
    runtime = time.perf_counter() - start_time

    stats = {
        "runtime": runtime,
        "nodes_expanded": nodes_expanded,
        "max_frontier_size": max_frontier_size,
    }

    if mode == "test":
        return stats

    return history, parent