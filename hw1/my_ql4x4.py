import argparse
import os
import sys
import csv
import numpy as np
from collections import defaultdict
import pandas as pd

if "SUMO_HOME" in os.environ:
    tools = os.path.join(os.environ["SUMO_HOME"], "tools")
    sys.path.append(tools)
else:
    sys.exit("Please declare the environment variable 'SUMO_HOME'")

from sumo_rl import SumoEnvironment


def initialize_q_table(n_action_space):
    return defaultdict(lambda: np.zeros(n_action_space, dtype=float))


def select_action_eps_greedy(Q, state, epsilon):
    q_values = Q[state]
    n_actions = q_values.shape[0]
    if np.random.rand() < epsilon:
        return int(np.random.randint(n_actions))
    else:
        return int(np.argmax(q_values))


def update_Q(Q, s, a, r, next_s, alpha, gamma):
    V_next = float(np.max(Q[next_s]))
    td_error = r + gamma * V_next - Q[s][a]
    Q[s][a] += alpha * td_error


class MyQLAgent:
    def __init__(
        self,
        starting_state,
        action_space,
        alpha=0.1,
        gamma=0.99,
        initial_epsilon=0.05,
        min_epsilon=0.005,
        decay=1.0,
    ):
        self.state = starting_state
        self.n_actions = action_space.n
        self.Q = initialize_q_table(self.n_actions)
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = initial_epsilon
        self.min_epsilon = min_epsilon
        self.decay = decay
        self.last_action = None

    def act(self):
        action = select_action_eps_greedy(self.Q, self.state, self.epsilon)
        self.last_action = action
        return action

    def learn(self, next_state, reward):
        update_Q(
            self.Q,
            s=self.state,
            a=self.last_action,
            r=reward,
            next_s=next_state,
            alpha=self.alpha,
            gamma=self.gamma,
        )
        self.state = next_state
        self.epsilon = max(self.min_epsilon, self.epsilon * self.decay)


if __name__ == "__main__":
    alpha = 0.1
    gamma = 0.99
    decay = 1.0
    runs = 30
    episodes = 4

    env = SumoEnvironment(
        net_file="sumo_rl/nets/4x4-Lucas/4x4.net.xml",
        route_file="sumo_rl/nets/4x4-Lucas/4x4c1c2c1c2.rou.xml",
        use_gui=False,
        num_seconds=500,
        min_green=5,
        delta_time=5,
    )

    os.makedirs("outputs/my_4x4_reward", exist_ok=True)
    reward_log_path = "outputs/my_4x4_reward/episode_rewards.csv"

    if not os.path.exists(reward_log_path):
        with open(reward_log_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["run", "episode", "agent_id", "total_reward"])

    for run in range(1, runs + 1):
        initial_states = env.reset()

        ql_agents = {
            ts: MyQLAgent(
                starting_state=env.encode(initial_states[ts], ts),
                action_space=env.action_space,
                alpha=alpha,
                gamma=gamma,
                initial_epsilon=0.05,
                min_epsilon=0.005,
                decay=decay,
            )
            for ts in env.ts_ids
        }

        for episode in range(1, episodes + 1):
            if episode != 1:
                initial_states = env.reset()
                for ts in initial_states.keys():
                    ql_agents[ts].state = env.encode(initial_states[ts], ts)

            done = {"__all__": False}
            episode_rewards = {ts: 0.0 for ts in env.ts_ids}

            while not done["__all__"]:
                actions = {ts: ql_agents[ts].act() for ts in ql_agents}
                s, r, done, info = env.step(action=actions)

                for agent_id in s:
                    reward = r[agent_id]
                    next_state = env.encode(s[agent_id], agent_id)
                    episode_rewards[agent_id] += reward
                    ql_agents[agent_id].learn(next_state, reward)

            with open(reward_log_path, "a", newline="") as f:
                writer = csv.writer(f)
                for agent_id, total_r in episode_rewards.items():
                    writer.writerow([run, episode, agent_id, total_r])

            env.save_csv(f"outputs/my_4x4/ql-4x4grid_run{run}", episode)

    env.close()
