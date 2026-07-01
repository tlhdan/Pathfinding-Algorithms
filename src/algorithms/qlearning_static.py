import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import random
import numpy as np
from collections import defaultdict


actions = {
    0: (-1, 0),
    1: (0, 1),
    2: (1, 0),
    3: (0, -1)
}


def move_agent(s, a):
    ns = tuple(np.add(s, a))

    return ns


def get_reward(maze, proposed_pos, agent_pos, finish, invalid_move_penalty, finish_reward):
    success = False

    if maze[proposed_pos] == 0:
        proposed_pos = agent_pos
        reward = invalid_move_penalty
    
    else:
        reward = -maze[proposed_pos]
    
    if proposed_pos == finish:
        reward += finish_reward
        success = True
    
    return reward, proposed_pos, success


def q_learning(maze, start, finish, mode="run", eval_interval=50,
               episodes=1000, alpha=0.1, gamma=0.99, epsilon=1, epsilon_decay=0.995, epsilon_min=0.01):
    maze_size = maze.shape

    rows, cols = maze_size

    Q = defaultdict(lambda: np.zeros(4))

    invalid_move_penalty = -2 * np.max(maze)
    finish_reward = np.sum(maze)

    max_steps = rows * cols * 2

    training_history = []
    
    for episode in range(episodes):
        s = start

        done = False
        total_reward = 0

        for step in range(max_steps):
            if done:
                break
            
            # choose action
            if random.random() < epsilon:
                a = random.choice(list(actions.keys()))
            else:
                a = np.argmax(Q[s])

            # act
            proposed_pos = move_agent(s, actions[a])

            # get reward
            reward, proposed_pos, done = get_reward(maze, proposed_pos, s, finish, invalid_move_penalty, finish_reward)
            
            # update Q table
            if done:
                Q[s][a] += alpha * (reward - Q[s][a])
            else:
                ns = proposed_pos

                Q[s][a] += alpha * (reward + gamma * np.max(Q[ns]) - Q[s][a])

            # finalize
            s = proposed_pos

            total_reward += reward

        if (episode + 1) % eval_interval == 0:
            _, path_cost, success, _ = get_path(Q, maze, start, finish)
            
            training_history.append({
                "episode": episode + 1,
                "eval_success": int(success),
                "eval_cost": path_cost
            })

        epsilon = max(epsilon_min, epsilon * epsilon_decay)
    
    if mode == "test":
        return Q, training_history
    
    return Q


def get_path(Q, maze, start, finish):
    # set up
    rows, cols = maze.shape

    max_steps = rows * cols * 4

    # initialize
    s = start

    maze = maze.copy()

    # tracking
    path = [s]

    path_cost = 0

    history = []

    success = False

    # traverse
    for step in range(max_steps):
        if success:
            break

        # choose action
        if s not in Q:
            a = random.choice(list(actions.keys()))

        else:
            a = int(np.argmax(Q[s]))

        # act
        proposed_pos = move_agent(s, actions[a])

        _, proposed_pos, success = get_reward(maze, proposed_pos, s, finish, 0, 0)

        if proposed_pos != s:
            path_cost += maze[proposed_pos]
            path.append(proposed_pos)
        
        #finalize
        s = proposed_pos

        # log
        history.append({
            "step": step,
            "agent_pos": s,
            "action": a,
            "cost": path_cost
        })

    return path, path_cost, success, history