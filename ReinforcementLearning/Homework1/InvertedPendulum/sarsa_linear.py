"""
基于线性逼近器的Sarsa算法
"""
import numpy as np
from agent import Agent
from encoder import Encoder
import matplotlib.pyplot as plt
from matplotlib.pyplot import MultipleLocator
import seaborn as sns

np.random.seed(0)


class Sarsa_lam(Agent):

    def train(self, steps=100, episode_length=1000):
        self.encoder = Encoder(self.num_angle, self.num_angle_velocity)
        epsilon = 0.1
        return_list = []
        lr = 0.8
        for episode in range(steps):
            returns = 0
            s0_v = self.env.reset()
            a0_idx = self.epsilon_greedy_action(s0_v, epsilon)
            a0_v = np.array([self.action_list[a0_idx]])
            s0 = self.encoder.encode(s0_v, a0_v)
            iters = 0
            X = []
            reward = []
            X.append(s0)
            while iters < episode_length:
                iters += 1
                s1_v, r, done, info = self.env.step(a0_v)
                a1_idx = self.epsilon_greedy_action(s1_v, epsilon)
                a1_v = np.array([self.action_list[a1_idx]])
                s1 = self.encoder.encode(s1_v, a1_v)
                reward.append(r)
                X.append(s1)
                # delta = r + self.gamma * float(np.matmul(s1, self.weight)) - float(np.matmul(s0, self.weight))
                # self.weight = self.weight + np.matrix(lr * delta * s0).T
                returns += r
                # if done:
                #     break
                a0_v = a1_v
            X1 = np.array(X)[1:,:]
            X2 = np.array(X)[:-1,:]
            reward = np.array(reward)
            self.weight = np.matmul(np.matmul(np.linalg.inv(np.matmul(X1.T,(X1-self.gamma*X2))+0.000001*np.eye(300)),X1.T),reward)
            # print(self.weight)
        #     return_list.append(returns)
            print(episode, returns)
            agent.save_weight("sarsa_lam_linear.npy")
        # np.save("diff_list", return_list)

    def epsilon_greedy_action(self, state, epsilon):
        if np.random.random() < epsilon:
            a_idx = np.random.choice([0, 1, 2])
        else:
            q = []
            for a in self.action_list:
                s = self.encoder.encode(state, a)
                q.append(float(np.matmul(s, self.weight)))
            a_idx = np.argmax(q)
        return a_idx

    def best_action(self, state):
        self.encoder = Encoder(self.num_angle, self.num_angle_velocity)
        q = []
        for a in self.action_list:
            s = self.encoder.encode(state, a)
            q.append(float(np.matmul(s, self.weight)))
        a_idx = np.argmax(q)
        print(q)
        return np.array([self.action_list[a_idx]])


if __name__ == "__main__":
    agent = Sarsa_lam(10, 10)
    agent.weight = np.load("result/sarsa_lam_linear.npy")
    # print(agent.weight)
    encoder = Encoder(agent.num_angle, agent.num_angle_velocity)
    for i in range(agent.num_angle):
        for j in range(agent.num_angle_velocity):
            feature = encoder.encode(agent.idx2value(i, j), -3)
            agent.Q[i, j, 0] = float(np.matmul(feature, agent.weight))
            print(float(np.matmul(feature, agent.weight)))
    sns.heatmap(agent.Q[:, :, 0], xticklabels=False, yticklabels=False)
    plt.ylabel("Angle")
    plt.xlabel("Angle Velocity")
    plt.title("Voltage = -3V")
    plt.show()
    # agent.load_weight("sarsa_lam_linear.npy")
    # agent.train()
    # agent.demo()
