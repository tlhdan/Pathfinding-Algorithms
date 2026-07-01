import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pygame
import copy
from algorithms.qlearning_dynamic import *
from algorithms.astar_replanning import astar_replanning
from algorithms.reactive import reactive_agent


def draw_panel(
    screen,
    maze,
    start,
    finish,
    result,
    step,
    rect,
    initial_bonuses,
    initial_obstacles,
    cell_size,
    perspective=True
):
    path = result["path"]
    rows, cols = maze.shape
    perception_radius = 5
    half = perception_radius // 2
    current_step = min(step, len(path) - 1)

    bonuses = initial_bonuses.copy()
    obstacles = copy.deepcopy(initial_obstacles)

    for _ in range(current_step):
        obstacles = move_obstacles(maze, obstacles)

    for pos in path[:current_step + 1]:
        if pos in bonuses:
            bonuses.remove(pos)

    visited_cells = set(path[:current_step + 1])
    
    for r in range(rows):
        for c in range(cols):
            cell_rect = pygame.Rect(rect.x + c * cell_size, rect.y + r * cell_size, cell_size, cell_size)

            if maze[r, c] == 0:
                pygame.draw.rect(screen, "black", cell_rect)
            
            else:
                pygame.draw.rect(screen, "white", cell_rect)

    for vr, vc in visited_cells:
        cell_rect = pygame.Rect(rect.x + vc * cell_size, rect.y + vr * cell_size, cell_size, cell_size)
        pygame.draw.rect(screen, "lightblue", cell_rect)
    
    sr, sc = start
    pygame.draw.rect(screen, "red", (rect.x + sc * cell_size, rect.y + sr * cell_size, cell_size, cell_size))
    
    fr, fc = finish
    pygame.draw.rect(screen, "green", (rect.x + fc * cell_size, rect.y + fr * cell_size, cell_size, cell_size))

    for br, bc in bonuses:
        center = (rect.x + bc * cell_size + cell_size // 2, rect.y + br * cell_size + cell_size // 2)
        pygame.draw.circle(screen, "yellow", center, cell_size // 4)

    for obstacle_data in obstacles.values():
        orow, ocol = obstacle_data["pos"]
        center = (rect.x + ocol * cell_size + cell_size // 2, rect.y + orow * cell_size + cell_size // 2)
        pygame.draw.circle(screen, "brown", center, cell_size // 3)
    
    agent_pos = path[current_step]
    ar, ac = agent_pos
    pygame.draw.circle(screen, "blue", (rect.x + ac * cell_size + cell_size // 2, rect.y + ar * cell_size + cell_size // 2), cell_size // 3)

    if perspective:
        top_left_x = (ac - half) * cell_size
        top_left_y = (ar - half) * cell_size
        box_size = perception_radius * cell_size
        perception_rect = pygame.Rect(
            rect.x + top_left_x,
            rect.y + top_left_y,
            box_size,
            box_size
        )
        pygame.draw.rect(screen, "cyan", perception_rect, 3)


def draw_status(screen, font, x, y, title, log, success):
    title_text = font.render(title, True, "black")
    screen.blit(title_text, (x, y))

    success_text = font.render(f"Success: {'True' if success == 1 else 'False'}", True, "black")
    step_text = font.render(f"Step: {log['step']}", True, "black")
    bonus_text = font.render(f"Bonuses: {log['bonuses_collected']}", True, "black")
    collision_text = font.render(f"Collisions: {log['obstacle_collisions']}", True, "black")

    screen.blit(success_text, (x, y + 20))
    screen.blit(step_text, (x, y + 40))
    screen.blit(bonus_text, (x, y + 60))
    screen.blit(collision_text, (x, y + 80))


def visualize(initial_maze, start, finish, initial_bonuses, initial_obstacles, cell_size=25):
    # initialize
    pygame.init()

    maze = initial_maze.copy()
    rows, cols = maze.shape
    panel_height = 150
    maze_width = cols * cell_size
    gap = 20
    width = maze_width * 3 + gap * 4
    height = rows * cell_size + panel_height
    font = pygame.font.SysFont(None, cell_size)
    clock = pygame.time.Clock()

    screen = pygame.display.set_mode((width, height))
    pygame.display.set_caption("Dynamic Maze Visualization")

    next_button = pygame.Rect(5, 5, cell_size // 4 * 7, cell_size)
    run_button = pygame.Rect(cell_size // 4 * 7 + 10, 5, cell_size * 3, cell_size)
    exit_button = pygame.Rect(cell_size // 4 * 7 + cell_size * 3 + 15, 5, cell_size // 4 * 7, cell_size)

    q_rect = pygame.Rect(
        gap,
        panel_height,
        maze_width,
        rows * cell_size
    )

    astar_rect = pygame.Rect(
        gap * 2 + maze_width,
        panel_height,
        maze_width,
        rows * cell_size
    )

    reactive_rect = pygame.Rect(
        gap * 3 + maze_width * 2,
        panel_height,
        maze_width,
        rows * cell_size
    )

    Q = q_learning(initial_maze, start, finish, initial_bonuses, initial_obstacles)

    q_result = get_path(
        Q,
        maze,
        start,
        finish,
        initial_bonuses,
        initial_obstacles
    )

    astar_result = astar_replanning(
        maze,
        start,
        finish,
        initial_bonuses,
        initial_obstacles
    )

    reactive_result = reactive_agent(
        maze,
        start,
        finish,
        initial_bonuses,
        initial_obstacles
    )

    step = 0

    running = True

    # main loop
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            
            if event.type == pygame.MOUSEBUTTONDOWN:
                if run_button.collidepoint(event.pos):
                    step = 0
                
                elif next_button.collidepoint(event.pos):
                    max_steps = max(
                        len(q_result["path"]),
                        len(astar_result["path"]),
                        len(reactive_result["path"])
                    )

                    if step < max_steps - 1:
                        step += 1
                
                elif exit_button.collidepoint(event.pos):
                    running = False
        
        screen.fill("white")

        # panel
        pygame.draw.rect(
            screen,
            "azure3",
            (0, 0, width, panel_height)
        )

        pygame.draw.rect(screen, "lightgreen", next_button)
        pygame.draw.rect(screen, "black", next_button, 2)
        next_text = font.render("Next", True, "black")
        screen.blit(
            next_text,
            (
                next_button.x + (next_button.width - next_text.get_width()) // 2,
                next_button.y + (next_button.height - next_text.get_height()) // 2
            )
        )

        pygame.draw.rect(screen, "khaki", run_button)
        pygame.draw.rect(screen, "black", run_button, 2)
        run_text = font.render("Restart", True, "black")
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

        q_log = q_result["history"][min(step - 1, len(q_result["history"]) - 1)]
        astar_log = astar_result["history"][min(step - 1, len(astar_result["history"]) - 1)]
        reactive_log = reactive_result["history"][min(step - 1, len(reactive_result["history"]) - 1)]

        draw_status(screen, font, q_rect.x, 40, "Q-Learning", q_log, q_result["success"])
        draw_status(screen, font, astar_rect.x, 40, "A* Replanning", astar_log, astar_result["success"])
        draw_status(screen, font, reactive_rect.x, 40, "Reactive", reactive_log, reactive_result["success"])

        # visualization
        draw_panel(
            screen,
            maze,
            start,
            finish,
            q_result,
            step,
            q_rect,
            initial_bonuses,
            initial_obstacles,
            cell_size
        )

        draw_panel(
            screen,
            maze,
            start,
            finish,
            astar_result,
            step,
            astar_rect,
            initial_bonuses,
            initial_obstacles,
            cell_size,
            perspective=False
        )

        draw_panel(
            screen,
            maze,
            start,
            finish,
            reactive_result,
            step,
            reactive_rect,
            initial_bonuses,
            initial_obstacles,
            cell_size,
            perspective=False
        )

        pygame.display.flip()

        clock.tick(60)

    pygame.quit()
