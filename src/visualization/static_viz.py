import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pygame
from algorithms.bfs import bfs
from algorithms.dfs import dfs
from algorithms.dijkstra import dijkstra
from algorithms.astar import astar


def get_cell_color(cost):
    if cost <= 10:
        return "grey95"

    elif cost <= 20:
        return "grey90"

    elif cost <= 30:
        return "grey85"

    elif cost <= 40:
        return "grey80"

    else:
        return "grey75"
    

def draw_maze(screen, maze, start, finish, rect, visited, frontier, path):
    maze_rows, maze_cols = maze.shape

    available_width = rect.width
    available_height = rect.height

    cell_size = min(
        available_width // maze_cols,
        available_height // maze_rows
    )

    cell_size = max(cell_size, 1)

    maze_pixel_width = maze_cols * cell_size
    maze_pixel_height = maze_rows * cell_size

    offset_x = rect.x + (rect.width - maze_pixel_width) // 2
    offset_y = rect.y + (available_height - maze_pixel_height) // 2

    for r in range(maze_rows):
        for c in range(maze_cols):
            cell = maze[r, c]

            if cell == 0:
                color = "black"

            else:
                color = get_cell_color(cell)

                if (r, c) in visited:
                    color = "blue"

                if (r, c) in frontier:
                    color = "yellow"

                if (r, c) in path:
                    color = "purple"

                if (r, c) == start:
                    color = "red"

                if (r, c) == finish:
                    color = "green"

            pygame.draw.rect(
                screen,
                color,
                (
                    offset_x + c * cell_size,
                    offset_y + r * cell_size,
                    cell_size,
                    cell_size
                )
            )


def reconstruct_path(parent, start, finish):
    path = []

    if finish not in parent:
        return path

    current = finish

    while current != start:
        path.append(current)
        current = parent[current]

    path.append(start)
    path.reverse()

    return path


def visualize(maze, start, finish):
    # initialize
    pygame.init()
    width, height = pygame.display.get_desktop_sizes()[0]
    screen = pygame.display.set_mode((width, height))
    pygame.display.set_caption("Pathfinding Algorithm Visualization")
    font = pygame.font.SysFont(None, width // 40)
    clock = pygame.time.Clock()

    panel_height = height // 16

    viz_y = panel_height
    viz_height = height - panel_height
    half_width = width // 2
    half_height = viz_height // 2

    bfs_rect = pygame.Rect(0, viz_y, half_width, half_height)
    dfs_rect = pygame.Rect(half_width, viz_y, half_width, half_height)
    dijkstra_rect = pygame.Rect(0, viz_y + half_height, half_width, half_height)
    astar_rect = pygame.Rect(half_width, viz_y + half_height, half_width, half_height)

    run_button = pygame.Rect(width // 36 * 25, panel_height // 5, width // 36 * 2, panel_height // 5 * 3)

    exit_button = pygame.Rect(width // 36 * 31, panel_height // 5, width // 36 * 2, panel_height // 5 * 3)

    bfs_history, bfs_parent = bfs(maze, start, finish)
    dfs_history, dfs_parent = dfs(maze, start, finish)
    dijkstra_history, dijkstra_parent = dijkstra(maze, start, finish)
    astar_history, astar_parent = astar(maze, start, finish)

    bfs_path = reconstruct_path(bfs_parent, start, finish)
    dfs_path = reconstruct_path(dfs_parent, start, finish)
    dijkstra_path = reconstruct_path(dijkstra_parent, start, finish)
    astar_path = reconstruct_path(astar_parent, start, finish)

    step = 0

    # main loop
    running = True

    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False


            if event.type == pygame.MOUSEBUTTONDOWN:
                if run_button.collidepoint(event.pos):
                    step = 0

                elif exit_button.collidepoint(event.pos):
                    running = False

        screen.fill("white")

        # panel
        pygame.draw.rect(
            screen,
            "azure3",
            (0, 0, width, panel_height)
        )

        pygame.draw.rect(screen, "lightgreen", run_button)
        pygame.draw.rect(screen, "black", run_button, 2)
        run_text = font.render("Run", True, "black")
        screen.blit(run_text, (
                run_button.x + (run_button.width - run_text.get_width()) // 2,
                run_button.y + (run_button.height - run_text.get_height()) // 2
            )
        )

        pygame.draw.rect(screen, "salmon", exit_button)
        pygame.draw.rect(screen, "black", exit_button, 2)
        exit_text = font.render("Exit", True, "black")
        screen.blit(
            exit_text,
            (
                exit_button.x + (exit_button.width - exit_text.get_width()) // 2,
                exit_button.y + (exit_button.height - exit_text.get_height()) // 2
            )
        )

        # visualization
        bfs_visited, bfs_frontier = bfs_history[min(step, len(bfs_history)-1)]
        dfs_visited, dfs_frontier = dfs_history[min(step, len(dfs_history)-1)]
        dijkstra_visited, dijkstra_frontier = dijkstra_history[min(step, len(dijkstra_history)-1)]
        astar_visited, astar_frontier = astar_history[min(step, len(astar_history)-1)]

        draw_maze(screen, maze, start, finish, bfs_rect, bfs_visited, bfs_frontier, bfs_path if step >= len(bfs_history) - 1 else [])
        draw_maze(screen, maze, start, finish, dfs_rect, dfs_visited, dfs_frontier, dfs_path if step >= len(dfs_history) - 1 else [])
        draw_maze(screen, maze, start, finish, dijkstra_rect, dijkstra_visited, dijkstra_frontier, dijkstra_path if step >= len(dijkstra_history) - 1 else [])
        draw_maze(screen, maze, start, finish, astar_rect, astar_visited, astar_frontier, astar_path if step >= len(astar_history) - 1 else [])

        step += 1

        pygame.draw.rect(screen, "azure4", bfs_rect, 2)
        bfs_label = font.render("BFS", True, "azure3")
        screen.blit(bfs_label, (bfs_rect.x + 10, bfs_rect.y + 10))

        pygame.draw.rect(screen, "azure4", dfs_rect, 2)
        dfs_label = font.render("DFS", True, "azure3")
        screen.blit(dfs_label, (dfs_rect.x + 10, dfs_rect.y + 10))

        pygame.draw.rect(screen, "azure4", dijkstra_rect, 2)
        dijkstra_label = font.render("Dijkstra", True, "azure3")
        screen.blit(dijkstra_label, (dijkstra_rect.x + 10, dijkstra_rect.y + 10))

        pygame.draw.rect(screen, "azure4", astar_rect, 2)
        astar_label = font.render("A*", True, "azure3")
        screen.blit(astar_label, (astar_rect.x + 10, astar_rect.y + 10))
        
        pygame.draw.rect(screen, "azure4", bfs_rect, 2)
        pygame.draw.rect(screen, "azure4", dfs_rect, 2)
        pygame.draw.rect(screen, "azure4", dijkstra_rect, 2)
        pygame.draw.rect(screen, "azure4", astar_rect, 2)

        pygame.display.flip()

        clock.tick(60)

    pygame.quit()