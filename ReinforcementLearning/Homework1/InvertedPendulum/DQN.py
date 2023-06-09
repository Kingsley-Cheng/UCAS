from agent import Agent
from torch import nn
import torch
from torch.nn import functional as F
import numpy as np
import collections
import random
import matplotlib.pyplot as plt
from matplotlib.pyplot import MultipleLocator
import seaborn as sns

np.random.seed(200)


class Replaybuffer:
    def __init__(self, capacity):
        self.buffer = collections.deque(maxlen=capacity)

    def add(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        transitions = random.sample(self.buffer, batch_size)
        state, action, reward, next_state, done = zip(*transitions)
        return np.array(state), action, reward, np.array(next_state), done

    def size(self):
        return len(self.buffer)


class Qnet(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.fc1 = nn.Linear(2, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, 3)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        return self.fc3(x)


class DQN(Agent):
    def __init__(self, hidden_dim, num_angle=200, num_angle_velocity=200, num_actions=3, gamma=0.98):
        super().__init__(num_angle, num_angle_velocity, num_actions, gamma)
        self.hidden_dim = hidden_dim
        self.q_net = Qnet(self.hidden_dim)
        self.target_q_net = Qnet(self.hidden_dim)
        self.optimizer = torch.optim.Adam(self.q_net.parameters(), lr=0.01)

    def epsilon_greedy_action(self, state):
        if np.random.random() < 0.01:
            action = np.random.choice([0, 1, 2])
        else:
            state = torch.tensor([state], dtype=torch.float)
            action = self.q_net(state).argmax().item()
        return action

    def train(self, steps=2000, episode_length=20000):
        batch_size = 64
        minimal_size = 500
        buffer_size = 10000
        replay_buffer = Replaybuffer(buffer_size)
        ep_returns = []
        for episode in range(steps):
            s0_v = self.env.reset()
            iters = 0
            returns = 0
            loss = 0
            while iters < episode_length:
                a0_idx = self.epsilon_greedy_action(s0_v)
                a0_v = np.array([self.action_list[a0_idx]])
                s1_v, r, done, info = self.env.step(a0_v)
                replay_buffer.add(s0_v, a0_idx, r, s1_v, done)
                s0_v = s1_v
                returns += r
                if replay_buffer.size() > minimal_size:
                    b_s, b_a, b_r, b_ns, b_d = replay_buffer.sample(batch_size)
                    states = torch.tensor(b_s, dtype=torch.float)
                    actions = torch.tensor(b_a).view(-1, 1)
                    rewards = torch.tensor(b_r, dtype=torch.float).view(-1, 1)
                    next_states = torch.tensor(b_ns, dtype=torch.float)
                    dones = torch.tensor(b_d, dtype=torch.float).view(-1, 1)

                    q_values = self.q_net(states).gather(1, actions)
                    max_next_q_values = self.target_q_net(next_states).max(1)[0].view(-1, 1)
                    delta = rewards + self.gamma * max_next_q_values * (1 - dones)
                    dqn_loss = torch.mean(F.mse_loss(q_values, delta))
                    self.optimizer.zero_grad()
                    dqn_loss.backward()
                    self.optimizer.step()
                    if iters % 10 == 0:
                        self.target_q_net.load_state_dict(self.q_net.state_dict())
                    iters += 1
                if done:
                    break
            ep_returns.append(returns)
            torch.save(self.q_net, "./InvertedPendulum/result/dqn")
            print(episode, returns)
            np.save("diff_list2.npy", ep_returns)

    def best_action(self, state):
        state = torch.tensor([state], dtype=torch.float)
        with torch.no_grad():
            action = self.target_q_net(state).argmax().item()
        return np.array([self.action_list[action]])


if __name__ == "__main__":
    agent = DQN(hidden_dim=128)
    agent.target_q_net = torch.load("./InvertedPendulum/result/dqn")
    # agent.q_net = torch.load("./InvertedPendulum/result/dqn")
    # agent.train()
    for i in range(agent.num_angle):
        for j in range(agent.num_angle_velocity):
            state = torch.tensor([i, j], dtype=torch.float)
            agent.Q[j, i, 0] = agent.target_q_net(state)[0]
    sns.heatmap(agent.Q[:, :, 0], xticklabels=False, yticklabels=False)
    plt.ylabel("Angle")
    plt.xlabel("Angle Velocity")
    plt.title("Voltage = -3V")
    plt.show()
    # diff_list = np.load("diff_list2.npy")
    # plt.plot(list(np.arange(155)),diff_list)
    # plt.xlabel("epochs")
    # plt.ylabel("Rewards")
    # plt.grid()
    # plt.show()
    # agent.demo()
