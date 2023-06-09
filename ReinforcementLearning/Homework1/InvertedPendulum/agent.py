# imports
import os
import time

import numpy as np

import InvertedPendulum


class Agent:
    """
    策略基类，定义基本成员和需要实现的方法
    离散Q值迭代，离散Sarsa和线性逼近器的Sarsa（lambda）算法继承
    """

    def __init__(self, num_angle=200, num_angle_velocity=200, num_actions=3, gamma=0.98):
        self.gamma = gamma  # 折扣因子
        self.num_angle = num_angle  # 离散化分割角度数量，逼近器下采样中心
        self.num_angle_velocity = num_angle_velocity  # 离散化分割角速度数量，逼近器下采样中心
        self.num_actions = num_actions  # 动作数量
        self.encoder = None  # 线性逼近器 rbf 编码器

        self.env = InvertedPendulum.InvertedPendulum()  # 策略所需要解决的环境
        self.MAX_angle = self.env.MAX_angle  # 最大角度
        self.MAX_angle_velocity = self.env.MAX_angle_velocity  # 最大角速度
        self.MAX_voltage = self.env.MAX_voltage  # 最大电压
        self.action_list = np.array([-3, 0, 3])  # 离散化的特征动作列表

        self.Q = np.zeros((self.num_angle, self.num_angle_velocity, self.num_actions))  # Q函数表格（状态空间*动作空间）
        self.weight = np.zeros((num_angle * num_angle_velocity * num_actions, 1))  # 线性逼近器的权重

        self.path = "./InvertedPendulum/result/"  # 存储权重与Q函数表格的位置

        # 判断是否已存在该文件夹
        if not os.path.exists(self.path):
            os.makedirs(self.path)

    def train(self):
        """
        各个策略算法的核心实现
        :return:
        """
        pass

    def best_action(self, state):
        """
        每个状态下的最优动作index
        :param state: 状态
        :return: 最优动作index
        """
        pass

    def demo(self, iterations=1000):
        """
        实现在一个策略下，环境结果
        :param iterations: 一次实验最多动作采样次数
        :return:
        """
        # 初始化环境，获取初始状态
        s0 = self.env.reset()
        # 进行渲染
        self.env.render()
        # 停 1 s
        time.sleep(1)
        # 动作采样计数器
        step = 0
        while True:
            step += 1
            self.env.render()
            # 根据状态采样动作
            a = self.best_action(s0)
            # 根据动作获得下一个状态与此动作产生的收益
            s0, r, done, info = self.env.step(a)
            # 判断是否达到目标或者达到最大动作采样
            if done or step > iterations:
                print("Finished!")
                break
        # 停 2 s 用于显示结果
        time.sleep(30)
        # 关闭环境
        self.env.close()

    def save_Q(self, filename='Q.npy'):
        """
        存储Q表格
        :param filename: 文件名称，默认为"Q.npy"
        :return:
        """
        np.save(self.path + filename, self.Q)

    def load_Q(self, file=""):
        """
        加载 Q 表格，将其赋值给 self.Q
        :param file: 文件名称，默认为"Q.npy"
        :return:
        """
        file = self.path + file if file else self.path + "Q.npy"
        self.Q = np.load(file)

    def save_weight(self, filename='Q.npy'):
        """
        存储线性逼近器权重
        :param filename: 文件名称，默认为"Q.npy"
        :return:
        """
        np.save(self.path + filename, self.weight)

    def load_weight(self, file=""):
        """
        加载线性逼近器权重，将其赋值给 self.weight
        :param file: 文件名称，默认为"Q.npy"
        :return:
        """
        file = self.path + file if file else self.path + "Q.npy"
        self.weight = np.load(file)

    def value2idx(self, angle, angle_velocity):
        """
        给定连续空间下状态，返回离散空间下状态表示
        :param angle: 连续空间下角度
        :param angle_velocity: 连续空间下角速度
        :return: 离散空间下状态表示 [离散角度表示，离散角速度表示]
        """
        # 角度规范化
        angle = (angle + self.MAX_angle) % (2 * self.MAX_angle) - self.MAX_angle
        # 获取角度离散表示
        idx_angle = int((angle + self.MAX_angle) / (2 * self.MAX_angle) * self.num_angle)
        # 获取角速度离散表示
        idx_angle_velocity = int(
            (angle_velocity + self.MAX_angle_velocity) / (2 * self.MAX_angle_velocity) * self.num_angle_velocity)
        # 注意，对于角速度空间，最后一个点归到前一个index下
        if idx_angle_velocity == self.num_angle_velocity:
            idx_angle_velocity -= 1
        return np.array([idx_angle, idx_angle_velocity])

    def idx2value(self, idx_angle, idx_angle_velocity):
        """
        给定离散空间下状态，返回在连续空间中的表示
        :param idx_angle: 离散空间下角度
        :param idx_angle_velocity: 离散空间下角速度
        :return: 连续空间下状态表示 [角度中点表示，角速度中点表示]
        """
        # 离散空间中角度中点表示为其连续空间中的角度
        angle = (idx_angle + 0.5) / self.num_angle * 2 * self.MAX_angle - self.MAX_angle
        # 离散空间中角速度中点表示为其连续空间中的角速度
        angle_velocity = (idx_angle_velocity + 0.5) / self.num_angle_velocity * \
                         2 * self.MAX_angle_velocity - self.MAX_angle_velocity
        # 角度规范化
        angle = (angle + self.MAX_angle) % (2 * self.MAX_angle) - self.MAX_angle
        return np.array([angle, angle_velocity])
