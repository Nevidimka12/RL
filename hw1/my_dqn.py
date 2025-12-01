import os
import sys
from collections import deque

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim

if "SUMO_HOME" in os.environ:
    tools = os.path.join(os.environ["SUMO_HOME"], "tools")
    sys.path.append(tools)
else:
    sys.exit("Please declare the environment variable 'SUMO_HOME'")

from sumo_rl import SumoEnvironment


def create_network(input_dim, hidden_dims, output_dim):
    layers = []
    in_dim = input_dim
    for h in (hidden_dims or []):
        layers.append(nn.Linear(in_dim, h))
        layers.append(nn.ReLU())
        in_dim = h
    layers.append(nn.Linear(in_dim, output_dim))
    return nn.Sequential(*layers)


def select_action_eps_greedy(Q, state, epsilon):
    if not isinstance(state, torch.Tensor):
        state = torch.tensor(state, dtype=torch.float32)
    q_values = Q(state).detach().numpy()
    n = q_values.shape[-1]
    if np.random.rand() < epsilon:
        return int(np.random.randint(n))
    return int(np.argmax(q_values))


def to_tensor(x, dtype=np.float32):
    if isinstance(x, torch.Tensor):
        return x
    return torch.from_numpy(np.asarray(x, dtype=dtype))


def compute_td_target(Q, rewards, next_states, terminateds, gamma=0.99):
    r = to_tensor(rewards)
    s_next = to_tensor(next_states)
    term = to_tensor(terminateds, bool)
    q_sn = Q(s_next)
    v_sn = torch.max(q_sn, dim=1).values
    return r + (1.0 - term.float()) * gamma * v_sn


def compute_td_loss(Q, states, actions, td_target, regularizer=0.1, out_non_reduced_losses=False):
    s = to_tensor(states)
    a = to_tensor(actions, int).long()
    q_s = Q(s)
    q_s_a = q_s.gather(1, a.view(-1, 1)).squeeze(1)
    td_error = td_target.detach() - q_s_a
    td_losses = td_error ** 2
    loss = torch.mean(td_losses)
    loss += regularizer * torch.abs(q_s_a).mean()
    if out_non_reduced_losses:
        return loss, td_losses.detach()
    return loss


def symlog(x):
    x = np.asarray(x, dtype=float)
    return np.sign(x) * np.log(np.abs(x) + 1.0)


def softmax(xs, temp=1.0):
    xs = np.asarray(xs, dtype=float)
    e = np.exp((xs - xs.max()) / temp)
    return e / e.sum()


def sample_prioritized_batch(replay_buffer, n_samples):
    priorities = np.array([s[0] for s in replay_buffer], dtype=float)
    logits = symlog(priorities)
    probs = softmax(logits)
    rng = np.random.default_rng()
    replace = n_samples > len(replay_buffer)
    indices = rng.choice(len(replay_buffer), size=n_samples, replace=replace, p=probs)
    states, actions, rewards, next_states, terms = [], [], [], [], []
    for idx in indices:
        _, s, a, r, ns, t = replay_buffer[idx]
        states.append(s)
        actions.append(a)
        rewards.append(r)
        next_states.append(ns)
        terms.append(t)
    batch = (
        np.array(states),
        np.array(actions),
        np.array(rewards),
        np.array(next_states),
        np.array(terms),
    )
    return batch, indices


def update_batch(replay_buffer, indices, batch, new_priority):
    states, actions, rewards, next_states, terms = batch
    for i in range(len(indices)):
        replay_buffer[indices[i]] = (
            float(new_priority[i]),
            states[i],
            actions[i],
            rewards[i],
            next_states[i],
            terms[i],
        )


def sort_replay_buffer(replay_buffer):
    new_rb = deque(maxlen=replay_buffer.maxlen)
    new_rb.extend(sorted(replay_buffer, key=lambda x: x[0]))
    return new_rb


def linear(st, end, duration, t):
    if t >= duration:
        return end
    return st + (end - st) * (t / duration)


def run_dqn_prioritized_rb_sumo(
    total_max_steps=10000,
    hidden_dims=(256, 256),
    lr=1e-3,
    gamma=0.99,
    eps_st=0.4,
    eps_end=0.02,
    eps_dur=0.25,
    train_schedule=4,
    replay_buffer_size=1000,
    batch_size=64,
    sort_every=40,
    reward_csv_path="outputs/big-intersection/dqn_prioritized_rewards.csv",
):
    env = SumoEnvironment(
        net_file="sumo_rl/nets/big-intersection/big-intersection.net.xml",
        route_file="sumo_rl/nets/big-intersection/routes.rou.xml",
        single_agent=True,
        use_gui=False,
        num_seconds=5400,
        yellow_time=3,
        min_green=5,
        max_green=20,
    )

    os.makedirs(os.path.dirname(reward_csv_path), exist_ok=True)
    if not os.path.exists(reward_csv_path):
        pd.DataFrame(columns=["episode", "total_reward"]).to_csv(reward_csv_path, index=False)

    obs = env.observation_space
    act = env.action_space
    state_dim = int(np.prod(obs.shape))
    n_actions = act.n

    Q = create_network(state_dim, hidden_dims, n_actions)
    optimizer = optim.Adam(Q.parameters(), lr=lr)

    replay_buffer = deque(maxlen=replay_buffer_size)

    s, _ = env.reset()
    s = np.asarray(s, dtype=np.float32).flatten()
    episode_reward = 0.0
    episode_idx = 0
    eps_steps = int(eps_dur * total_max_steps)

    for step in range(1, total_max_steps + 1):
        epsilon = linear(eps_st, eps_end, eps_steps, step)
        a = select_action_eps_greedy(Q, s, epsilon)

        s_next, r, terminated, truncated, _ = env.step(a)
        s_next = np.asarray(s_next, dtype=np.float32).flatten()
        done = terminated or truncated
        episode_reward += r

        with torch.no_grad():
            s_t = to_tensor(s).float().unsqueeze(0)
            ns_t = to_tensor(s_next).float().unsqueeze(0)
            r_t = torch.tensor([r], dtype=torch.float32)
            a_t = torch.tensor([a], dtype=torch.long)
            term_t = torch.tensor([terminated], dtype=torch.bool)
            td_target = compute_td_target(Q, r_t, ns_t, term_t, gamma)
            loss_single = compute_td_loss(Q, s_t, a_t, td_target)
            priority = float(loss_single.detach().cpu().numpy())

        replay_buffer.append((priority, s, a, r, s_next, terminated))

        if len(replay_buffer) >= batch_size and step % train_schedule == 0:
            batch, indices = sample_prioritized_batch(replay_buffer, batch_size)
            states, actions, rewards, next_states, terms = batch
            optimizer.zero_grad()
            td_target_b = compute_td_target(Q, rewards, next_states, terms, gamma)
            loss, td_losses = compute_td_loss(Q, states, actions, td_target_b, out_non_reduced_losses=True)
            loss.backward()
            optimizer.step()
            update_batch(replay_buffer, indices, batch, td_losses.cpu().numpy())

        if len(replay_buffer) >= batch_size and step % (sort_every * train_schedule) == 0:
            replay_buffer = sort_replay_buffer(replay_buffer)

        if done:
            episode_idx += 1
            df = pd.read_csv(reward_csv_path)
            df.loc[len(df)] = [episode_idx, float(episode_reward)]
            df.to_csv(reward_csv_path, index=False)
            print(f"[episode {episode_idx}] total_reward = {episode_reward:.3f}, eps = {epsilon:.3f}")
            s, _ = env.reset()
            s = np.asarray(s, dtype=np.float32).flatten()
            episode_reward = 0.0
        else:
            s = s_next

    env.close()
    print("DONE")


if __name__ == "__main__":
    run_dqn_prioritized_rb_sumo(
        total_max_steps=10000,
        hidden_dims=(256, 256),
        lr=1e-3,
        gamma=0.99,
        eps_st=0.4,
        eps_end=0.02,
        eps_dur=0.25,
        train_schedule=4,
        replay_buffer_size=1000,
        batch_size=64,
        sort_every=10,
        reward_csv_path="outputs/big-intersection/dqn_prioritized_rewards.csv",
    )
