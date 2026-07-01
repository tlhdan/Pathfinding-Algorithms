import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import copy
from algorithms.astar import astar
from maze_generator import move_obstacles

OBSTACLE_COST = 5
BONUS_COST = 0.1


def build_cost_map(maze, bonuses, obstacles):
    cost_map = maze.astype(float).copy()

    for obstacle in obstacles.values():
        r, c = obstacle["pos"]
        cost_map[r, c] = OBSTACLE_COST

    for r, c in bonuses:
        cost_map[r, c] = BONUS_COST

    return cost_map


def astar_replanning(maze, start, finish, bonuses, obstacles):
    agent = start
    bonuses = bonuses.copy()
    obstacles = copy.deepcopy(obstacles)

    path = [agent]

    steps = 0
    bonuses_collected = 0
    obstacle_collisions = 0
    success = False

    max_steps = maze.shape[0] * maze.shape[1] * 2

    current_obstacles = copy.deepcopy(obstacles)
    previous_obstacles = copy.deepcopy(obstacles)

    history = []

    while steps < max_steps:

        if agent == finish:
            success = True
            break

        cost_map = build_cost_map(maze, bonuses, current_obstacles)

        planned_path = astar(cost_map, agent, finish, mode="path")

        if planned_path is None or len(planned_path) < 2:
            break

        next_pos = planned_path[1]

        previous_agent = agent
        agent = next_pos
        path.append(agent)

        steps += 1

        if agent in bonuses:
            bonuses.remove(agent)
            bonuses_collected += 1

        previous_obstacles = current_obstacles
        current_obstacles = move_obstacles(maze, current_obstacles)

        step_collisions = 0

        for oid, obs in current_obstacles.items():
            old_pos = previous_obstacles[oid]["pos"]
            new_pos = obs["pos"]

            if agent == new_pos:
                step_collisions += 1
            elif agent == old_pos and previous_agent == new_pos:
                step_collisions += 1

        obstacle_collisions += step_collisions

        history.append({
            "step": steps,
            "agent_pos": agent,
            "bonuses_collected": bonuses_collected,
            "obstacle_collisions": obstacle_collisions,
            "success": success
        })

    return {
        "path": path,
        "success": int(success),
        "steps": steps,
        "bonuses_collected": bonuses_collected,
        "obstacle_collisions": obstacle_collisions,
        "history": history
    }