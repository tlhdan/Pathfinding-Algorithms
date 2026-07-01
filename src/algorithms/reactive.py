import copy
from maze_generator import move_obstacles

actions = {
    0: (-1, 0),
    1: (0, 1),
    2: (1, 0),
    3: (0, -1),
    4: (0, 0)
}


def manhattan(a, b):
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


def get_valid_moves(pos, maze):
    moves = []
    for a, (dr, dc) in actions.items():
        nr, nc = pos[0] + dr, pos[1] + dc
        if 0 <= nr < maze.shape[0] and 0 <= nc < maze.shape[1]:
            if maze[nr, nc] != 0:
                moves.append((a, (nr, nc)))
    return moves


def reactive_agent(maze, start, finish, bonuses, obstacles):

    agent = start
    bonuses = bonuses.copy()
    obstacles = copy.deepcopy(obstacles)

    path = [agent]
    history = []

    steps = 0
    bonuses_collected = 0
    obstacle_collisions = 0
    success = False

    max_steps = maze.shape[0] * maze.shape[1] * 2

    for _ in range(max_steps):

        if agent == finish:
            success = True
            break

        prev_agent = agent
        current_obstacles = obstacles
        next_obstacles = move_obstacles(maze, obstacles)

        if agent in bonuses:
            bonuses.remove(agent)
            bonuses_collected += 1

        danger_cells = {o["pos"] for o in next_obstacles.values()}

        valid_moves = get_valid_moves(agent, maze)

        best_action = 4
        best_score = float("inf")

        for a, pos in valid_moves:

            collision_risk = 1 if pos in danger_cells else 0
            dist = manhattan(pos, finish)
            bonus_score = -5 if pos in bonuses else 0

            score = dist + collision_risk * 100 + bonus_score

            if score < best_score:
                best_score = score
                best_action = a

        dr, dc = actions[best_action]
        new_pos = (agent[0] + dr, agent[1] + dc)

        if maze[new_pos] == 0:
            new_pos = agent

        agent = new_pos
        path.append(agent)

        for oid, obs in next_obstacles.items():
            old_pos = current_obstacles[oid]["pos"]
            new_pos_obs = obs["pos"]

            if agent == new_pos_obs:
                obstacle_collisions += 1
            elif agent == old_pos and prev_agent == new_pos_obs:
                obstacle_collisions += 1

        obstacles = next_obstacles
        steps += 1

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