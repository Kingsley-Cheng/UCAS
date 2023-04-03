import numpy as np
from agent import Agent
import matplotlib.pyplot as plt
from matplotlib.pyplot import MultipleLocator
import seaborn as sns
np.random.seed(200)


class Sarsa(Agent):
    def train(self, steps=500, episode_length=10000):
        epsilon = 0.1
        return_list = []
        lr = 0.8
        for episode in range(steps):
            old_Q = self.Q.copy()
            s0_v = self.env.reset()
            s0 = self.value2idx(s0_v[0], s0_v[1])
            iters = 0
            returns = 0
            while iters < episode_length:
                iters += 1
                a0_idx = self.epsilon_greedy_action(s0_v, epsilon)
                a0_v = np.array([self.action_list[a0_idx]])

                s1_v, r, done, info = self.env.step(a0_v)
                s1 = self.value2idx(s1_v[0], s1_v[1])
                a1_idx = np.argmax(self.Q[s1[0], s1[1]])
                delta = r + self.gamma * self.Q[s1[0], s1[1], a1_idx] - self.Q[s0[0], s0[1], a0_idx]
                returns += r
                self.Q[s0[0], s0[1], a0_idx] += lr * delta
                if done:
                    break
                s0_v = s1_v
                s0 = s1
            return_list.append(returns)
            print(episode, returns)
            agent.save_Q("Qlearning.npy")
        np.save("diff_list", return_list)

    def epsilon_greedy_action(self, state, epsilon):
        if np.random.random() < epsilon:
            a_idx = np.random.choice([0, 1, 2])
        else:
            state_idx = self.value2idx(state[0], state[1])
            a_idx = np.argmax(self.Q[state_idx[0], state_idx[1]])
        return a_idx

    def best_action(self, state):
        state_idx = self.value2idx(state[0], state[1])
        action_idx = np.argmax(self.Q[state_idx[0], state_idx[1]])
        return np.array([self.action_list[action_idx]])


if __name__ == "__main__":
    # agent = Sarsa(300, 300)
    # agent.load_Q("Qlearning.npy")
    # # agent.train()
    # step = np.load("step.npy")
    # diff_list = np.load("diff_list.npy")
    # x_major_locator = MultipleLocator(50)
    # plt.plot(range(1, step + 1), diff_list)
    # ax = plt.gca()
    # ax.xaxis.set_major_locator(x_major_locator)
    # ax.grid()
    # plt.xlabel("Epochs")
    # plt.ylabel("Rewards")
    # plt.show()
    # print(step)
    Q = np.load("result/Qlearning.npy")
    sns.heatmap(Q[:, :, 0], xticklabels=False, yticklabels=False)
    plt.ylabel("Angle")
    plt.xlabel("Angle Velocity")
    plt.title("Voltage = -3V")
    plt.show()
    # agent.demo()