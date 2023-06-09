"""
离散Q价值迭代
"""
import numpy as np
from agent import Agent
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.pyplot import MultipleLocator


class QIteration(Agent):
    def train(self, error=0.1):
        """
        重写父类方法
        :param error: Q函数阈值
        :return: None
        """
        step = 0  # 迭代次数计数器
        diff_list = []  # 记录每次迭代后与前一次 Q 值的变化
        while True:
            step += 1
            # 记录迭代后的 Q 值
            new_Q = np.zeros_like(self.Q)
            # 遍历所有离散状态空间
            for i in range(self.num_angle):
                for j in range(self.num_angle_velocity):
                    # 获取状态的连续空间表示
                    s0_v = self.idx2value(i, j)
                    # 遍历所有可能的动作
                    for a_idx in range(3):
                        # 获取当前动作
                        a = self.action_list[a_idx]
                        # 由动作更新下一个状态
                        s1_v = self.env.transition(s0_v, a)
                        # 获取新状态离散空间表示
                        s1_idx = self.value2idx(s1_v[0], s1_v[1])
                        # 计算在新状态下最大 Q 值
                        q = max(self.Q[s1_idx[0], s1_idx[1]])
                        # 计算根据前一状态和采取动作获得回报
                        r = self.env.reward(s0_v, a)
                        # 更新 Q 表格，即回报+新状态下的最大Q值
                        new_Q[i, j, a_idx] = r + self.gamma * q
            # 遍历一次后，计算更新前后，Q变化量
            diff = np.linalg.norm(new_Q - self.Q)
            # 将变化量记录下来
            diff_list.append(diff)
            # 更新全局 Q 为更新后的Q值
            self.Q = new_Q
            # 打印周期与每次更新量
            print(step, diff)
            # 每 10 次记录一次 Q 表格
            if step % 10 == 0:
                self.save_Q("Q300.npy")
            # 若 Q 变化量小于给定阈值，迭代完成
            if diff < error:
                print("finish")
                break
        np.save("step", step)
        np.save("diff_list", diff_list)

    def best_action(self, state):
        """
        重写父类方法
        :param state: 状态
        :return: 状态下最佳动作
        """
        # 动作的离散空间表示
        state_idx = self.value2idx(state[0], state[1])
        # 返回最大贪心动作 index
        action_idx = np.argmax(self.Q[state_idx[0], state_idx[1]])
        # 返回动作
        return np.array([self.action_list[action_idx]])


if __name__ == "__main__":
    agent = QIteration(300,300)
    # agent.load_Q("Q300.npy")
    agent.Q = np.load("result/Q300.npy")
    # sns.heatmap(Q[:, :, 0], xticklabels=False, yticklabels=False)
    # plt.ylabel("Angle")
    # plt.xlabel("Angle Velocity")
    # plt.title("Voltage = -3V")
    # plt.show()
    # agent.train()
    # step=np.load("step.npy")
    # diff_list = np.load("diff_list.npy")
    # x_major_locator=MultipleLocator(50)
    # plt.plot(range(1,step+1),diff_list)
    # ax=plt.gca()
    # ax.xaxis.set_major_locator(x_major_locator)
    # ax.grid()
    # plt.xlabel("epochs")
    # plt.ylabel("Q weight difference")
    # plt.show()
    # print(step)
    agent.demo()
