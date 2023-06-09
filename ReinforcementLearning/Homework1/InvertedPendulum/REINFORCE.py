import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from agent import Agent


class PolicyNet(nn.Module):
    def __init__(self, hidden_dim):
        super(PolicyNet, self).__init__()
        self.fc1 = nn.Linear(2, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, 3)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        return F.softmax(self.fc2(x), dim=1)


class REINFORCE(Agent):
    def __init__(self, hidden_dim, num_angle=200, num_angle_velocity=200, num_actions=3, gamma=0.98):
        super().__init__(num_angle, num_angle_velocity, num_actions, gamma)
        self.policy_net = PolicyNet(hidden_dim)
        self.optimizer = torch.optim.Adam(self.policy_net.parameters(), lr=100)

    def action(self, state):
        state = torch.tensor([state], dtype=torch.float)
        probs = self.policy_net(state)
        action_dist = torch.distributions.Categorical(probs)
        action = action_dist.sample()
        return action.item()

    def update(self, transition_dict):
        reward_list = transition_dict['rewards']
        state_list = transition_dict['states']
        action_list = transition_dict['actions']
        G = 0
        self.optimizer.zero_grad()
        ep_loss = 0
        for i in reversed(range(len(reward_list))):
            reward = reward_list[i]
            state = torch.tensor([state_list[i]], dtype=torch.float)
            action = torch.tensor(action_list[i]).view(-1, 1)
            log_prob = torch.log(self.policy_net(state).gather(1, action))
            G = self.gamma * G + reward
            loss = -log_prob * G
            ep_loss += loss
            loss.backward()
        self.optimizer.step()
        return ep_loss

    def train(self, steps=2000, episode_length=5000):
        ep_returns = []
        loss_list = []
        for episode in range(steps):
            transition_dict = {'states': [], 'actions': [], "next_states": [], 'rewards': [], "dones": []}
            s0_v = self.env.reset()
            iters = 0
            returns = 0
            while iters < episode_length:
                iters += 1
                a0_idx = self.action(s0_v)
                a0_v = np.array([self.action_list[a0_idx]])
                s1_v, r, done, info = self.env.step(a0_v)
                transition_dict['actions'].append(a0_idx)
                transition_dict['states'].append(s0_v)
                transition_dict['rewards'].append(r)
                transition_dict['dones'].append(done)
                transition_dict['next_states'].append(s1_v)
                s0_v = s1_v
                returns += r
                if done:
                    break
            loss = self.update(transition_dict)
            ep_returns.append(returns)
            loss_list.append(loss)
            torch.save(self.policy_net, "./InvertedPendulum/result/REINFORCE")
            print(episode, loss, returns)

    def best_action(self, state):
        state = torch.tensor([state], dtype=torch.float)
        probs = self.policy_net(state)
        action_dist = torch.distributions.Categorical(probs)
        action = action_dist.sample()
        return np.array([self.action_list[action.item()]])


if __name__ == "__main__":
    agent = REINFORCE(hidden_dim=128)
    agent.policy_net = torch.load("./InvertedPendulum/result/REINFORCE")
    # agent.train()
    agent.demo()
