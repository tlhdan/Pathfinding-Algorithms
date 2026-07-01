import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import random
import copy
import numpy as np
from collections import defaultdict
from maze_generator import move_obstacles


STEP_PENALTY = -1
IDLE_PENALTY = 0.5
INVALID_MOVE_PENALTY = -100
OBSTACLE_PENALTY = -5
BONUS_REWARD = 25
FINISH_REWARD = 100


actions = {
    0: (-1, 0),
    1: (0, 1),
    2: (1, 0),
    3: (0, -1),
    4: (0, 0)
}


def reset_environment(start, initial_maze, initial_bonuses, initial_obstacles):
    return (
        start,
        initial_maze.copy(),
        initial_bonuses.copy(),
        copy.deepcopy(initial_obstacles)
    )


def extract_local_patch(grid, center, radius=5):
    rows, cols = grid.shape
    patch = np.zeros((radius, radius), dtype=grid.dtype)

    cr, cc = center
    offset = radius // 2

    for r in range(radius):
        for c in range(radius):
            gr = cr + (r - offset)
            gc = cc + (c - offset)

            if 0 <= gr < rows and 0 <= gc < cols:
                patch[r, c] = grid[gr, gc]

    return patch


def get_obstacle_grid(shape, obstacles):
    grid = np.zeros(shape, dtype=np.int8)
    for o in obstacles.values():
        r, c = o["pos"]
        grid[r, c] = 1
    return grid


def get_bonus_grid(shape, bonuses):
    grid = np.zeros(shape, dtype=np.int8)
    for b in bonuses:
        grid[b] = 1
    return grid


def get_state(agent_pos, maze, bonuses, current_obstacles, previous_obstacles):
    return (
        agent_pos,
        tuple(extract_local_patch(maze, agent_pos).flatten()),
        tuple(extract_local_patch(get_bonus_grid(maze.shape, bonuses), agent_pos).flatten()),
        tuple(extract_local_patch(get_obstacle_grid(maze.shape, current_obstacles), agent_pos).flatten()),
        tuple(extract_local_patch(get_obstacle_grid(maze.shape, previous_obstacles), agent_pos).flatten())
    )


def move_agent(pos, action):
    return tuple(np.add(pos, action))


def get_reward(a, maze, proposed_pos, agent_pos,
               current_obstacles, previous_obstacles,
               bonuses, finish):

    reward = STEP_PENALTY
    collisions = 0
    bonus_collected = False
    success = False

    if a == 4:
        reward += IDLE_PENALTY

    # invalid move (wall)
    if maze[proposed_pos] == 0:
        proposed_pos = agent_pos
        reward += INVALID_MOVE_PENALTY

    # obstacle collisions
    for oid, obs in current_obstacles.items():
        old_pos = previous_obstacles[oid]["pos"]
        new_pos = obs["pos"]

        if proposed_pos == new_pos:
            collisions += 1
        elif proposed_pos == old_pos and agent_pos == new_pos:
            collisions += 1

    reward += collisions * OBSTACLE_PENALTY

    # bonus
    if proposed_pos in bonuses:
        bonuses.remove(proposed_pos)
        reward += BONUS_REWARD
        bonus_collected = True

    # goal
    if proposed_pos == finish:
        reward += FINISH_REWARD
        success = True

    return reward, proposed_pos, success, bonus_collected, collisions


def q_learning(initial_maze, start, finish,
               initial_bonuses, initial_obstacles,
               mode="run",
               eval_interval=50,
               episodes=1000,
               alpha=0.1,
               gamma=0.99,
               epsilon=1,
               epsilon_decay=0.995,
               epsilon_min=0.01):

    Q = defaultdict(lambda: np.zeros(len(actions)))
    max_steps = initial_maze.shape[0] * initial_maze.shape[1] * 2

    eval_history = []

    for episode in range(episodes):

        agent_pos, maze, bonuses, obstacles = reset_environment(
            start, initial_maze, initial_bonuses, initial_obstacles
        )

        current_obstacles = copy.deepcopy(obstacles)
        previous_obstacles = copy.deepcopy(obstacles)

        done = False

        for step in range(max_steps):
            if done:
                break

            s = get_state(agent_pos, maze, bonuses,
                           current_obstacles, previous_obstacles)

            if random.random() < epsilon:
                a = random.choice(list(actions.keys()))
            else:
                a = int(np.argmax(Q[s]))

            proposed_pos = move_agent(agent_pos, actions[a])

            new_obstacles = move_obstacles(maze, current_obstacles)
            previous_obstacles = current_obstacles
            current_obstacles = new_obstacles

            reward, proposed_pos, reward_done, _, _ = get_reward(
                a, maze, proposed_pos, agent_pos,
                current_obstacles, previous_obstacles,
                bonuses, finish
            )

            ns = get_state(proposed_pos, maze, bonuses,
                           current_obstacles, previous_obstacles)

            if reward_done:
                done = True
                Q[s][a] += alpha * (reward - Q[s][a])
            else:
                Q[s][a] += alpha * (reward + gamma * np.max(Q[ns]) - Q[s][a])

            agent_pos = proposed_pos

        epsilon = max(epsilon_min, epsilon * epsilon_decay)

        # evaluation snapshot
        if (episode + 1) % eval_interval == 0:
            result = get_path(
                Q,
                initial_maze,
                start,
                finish,
                initial_bonuses,
                initial_obstacles
            )

            eval_history.append({
                "episode": episode + 1,
                "eval_success": result["success"],
                "eval_step": result["steps"],
                "eval_reward": result["reward"]
            })

    if mode == "test":
        return Q, eval_history

    return Q


def get_path(Q, maze, start, finish, bonuses, obstacles):

    agent_pos = start
    bonuses = bonuses.copy()
    obstacles = copy.deepcopy(obstacles)

    current_obstacles = copy.deepcopy(obstacles)
    previous_obstacles = copy.deepcopy(obstacles)

    path = [agent_pos]
    history = []

    steps = 0
    total_reward = 0
    bonuses_collected = 0
    obstacle_collisions = 0
    success = False

    max_steps = maze.shape[0] * maze.shape[1] * 2

    for _ in range(max_steps):

        if agent_pos == finish:
            success = True
            break

        s = get_state(agent_pos, maze, bonuses,
                       current_obstacles, previous_obstacles)

        a = int(np.argmax(Q[s]))

        proposed_pos = move_agent(agent_pos, actions[a])

        new_obstacles = move_obstacles(maze, current_obstacles)
        previous_obstacles = current_obstacles
        current_obstacles = new_obstacles

        reward, proposed_pos, reward_success, bonus, collisions = get_reward(
            a, maze, proposed_pos, agent_pos,
            current_obstacles, previous_obstacles,
            bonuses, finish
        )

        total_reward += reward

        if reward_success:
            success = True

        agent_pos = proposed_pos
        path.append(agent_pos)

        steps += 1
        bonuses_collected += int(bonus)
        obstacle_collisions += collisions

        history.append({
            "step": steps,
            "agent_pos": agent_pos,
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
        "history": history,
        "reward": total_reward
    }