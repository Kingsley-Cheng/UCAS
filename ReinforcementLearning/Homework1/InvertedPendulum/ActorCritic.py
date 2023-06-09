import numpy as np
import torch
from torch import nn 
from torch.nn import functional as F
from agent import Agent

class PolicyNet(nn.Module):
    def __init__(self, hidden_dim):
        super(PolicyNet,self).__init__()
        self.fc1 = nn.Linear(2, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim,3)
    
    def forward(self, x):
        x = F.relu(self.fc1(x))
        return F.softmax(self.fc2(x),dim=1)
    
class ValueNet(nn.Module):
    def __init__(self,hidden_dim):
        super().__init__()
        self.fc1 = nn.Linear(2, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim,3)
    
    def forward(self, x):
        x = F.relu(self.fc1(x))
        return self.fc2(x)
    
class ActorCritic(Agent):
    def __init__(self,hidden_dim, num_angle=200, num_angle_velocity=200, num_actions=3, gamma=0.98):
        super().__init__(num_angle, num_angle_velocity, num_actions, gamma)
        self.actor = PolicyNet(hidden_dim)
        self.critic = ValueNet(hidden_dim)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(),lr = 0.001)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr= 0.01)

    def action(self, state):
        state = torch.tensor([state], dtype=torch.float)
        probs = self.actor(state)
        # action_dist = torch.distributions.Categorical(probs)
        # action = action_dist.sample()
        action = probs.argmax()
        return action.item()
    
    def update(self, transition_dict):
        states = torch.tensor(transition_dict['states'],dtype=torch.float)
        actions = torch.tensor(transition_dict['actions']).view(-1,1)
        rewards = torch.tensor(transition_dict['rewards'], dtype=torch.float).view(-1,1)
        next_states = torch.tensor(transition_dict['next_states'], dtype=torch.float)
        dones = torch.tensor(transition_dict['dones'], dtype=torch.float).view(-1,1)
        td_error = rewards + self.gamma*self.critic(next_states)*(1-dones)
        td_delta = td_error - self.critic(states)
        log_probs = torch.log(self.actor(states).gather(1, actions))
        actor_loss = torch.mean(-log_probs*td_delta.detach())
        critic_loss = torch.mean(F.mse_loss(self.critic(states), td_error.detach()))
        self.actor_optimizer.zero_grad()
        self.critic_optimizer.zero_grad()
        actor_loss.backward()
        critic_loss.backward()
        self.actor_optimizer.step()
        self.critic_optimizer.step()

    def train(self, steps=20000, episode_length = 10000):
        ep_returns = []
        for episode in range(steps):
            transition_dict = {'states':[], 'actions':[], "next_states":[],'rewards':[],"dones":[]}
            s0_v = self.env.reset()
            iters = 0
            returns = 0
            while iters < episode_length:
                iters +=1
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
            self.update(transition_dict)
            ep_returns.append(returns)
            torch.save(self.actor,"./InvertedPendulum/result/actor")
            torch.save(self.critic,"./InvertedPendulum/result/critic")
            print(episode,returns)

    def best_action(self, state):
        state = torch.tensor([state], dtype=torch.float)
        probs = self.actor(state)
        # action_dist = torch.distributions.Categorical(probs)
        # action = action_dist.sample()
        action = probs.argmax()
        return np.array([self.action_list[action.item()]])

if __name__ == "__main__":
    agent = ActorCritic(hidden_dim=128)
    agent.critic = torch.load("./InvertedPendulum/result/critic")
    agent.actor = torch.load("./InvertedPendulum/result/actor")
    agent.actor_optimizer = torch.optim.Adam(agent.actor.parameters(),lr = 0.05)
    agent.critic_optimizer = torch.optim.Adam(agent.critic.parameters(), lr= 0.5)
    agent.train()
    agent.demo()
    