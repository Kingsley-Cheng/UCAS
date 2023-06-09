"""
倒立摆环境仿真
"""
# imports
import time
from os import path

import gym
import numpy as np
from gym import spaces
from gym.utils import seeding

import rendering


# 定义倒立摆环境（继承gym的Env类）
class InvertedPendulum(gym.Env):
    """
    需要重写step，reset，render，close函数
    """

    metadata = {
        'render.modes': ['human', 'rgb_array'],
        'video.frames_per_second': 30
    }

    def __init__(self):
        #  定义系统参数

        self._m = 0.055  # 重量
        self._g = 9.81  # 重力加速度
        self._l = 0.042  # 重心到转子的距离
        self._J = 1.91e-4  # 转动惯量
        self._b = 3e-6  # 粘滞阻尼
        self._K = 0.0536  # 转距常数
        self._R = 9.5  # 转子电阻

        self.Ts = 0.005  # 采样时间
        self.MAX_angle = np.pi  # 最大角度
        self.MAX_angle_velocity = 15 * np.pi  # 最大角速度
        self.MAX_voltage = 3  # 最大电压

        # 初始化渲染环境所需要的参数
        self.viewer = None
        self.pole_transform = None
        self.img = None
        self.imgtrans = None
        self.last_action = None

        self.state = None  # 环境当前状态

        high = np.array([self.MAX_angle, self.MAX_angle_velocity])
        # 定义动作空间
        self.action_space = spaces.Box(low=np.array([-self.MAX_voltage], dtype=np.float32),
                                       high=np.array([self.MAX_voltage], np.float32), dtype=np.float32)
        # 定义状态空间
        self.observation_space = spaces.Box(low=-np.float32(high), high=np.float32(high), dtype=np.float32)

        # 定义随机种子，方便复现实验
        self.seed()
        self.init_state = np.array([-np.pi, 0])  # 定义初始初始状态

    def seed(self, seed=None):
        """
        :param seed: 随机种子
        :return:
        """
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def angle_acceleration(self, state, action):
        """
        通过倒立摆系统连续时间动力学模型获得角加速度
        :param state: 状态，state[0]: 角度. state[1]: 角速度
        :param action: 动作，输入电压
        :return: 当前动作状态下决定的角加速度
        """
        return (self._m * self._g * self._l * np.sin(state[0]) - self._b * state[1] -
                (self._K ** 2) * state[1] / self._R + self._K * action / self._R) / self._J

    def transition(self, state, action):
        """
        通过欧拉法得到的离散时间动力学模型，获得下一个状态
        :param state: 当前状态，state[0]: 角度. state[1]: 角速度
        :param action: 当前动作，输入电压
        :return: 下一时刻的状态
        """
        next_angle = state[0] + self.Ts * state[1]  # 下一时刻角度
        next_angle_velocity = state[1] + self.Ts * self.angle_acceleration(state, action)  # 下一时刻角加速度

        # 判断获得的角度与角速度是否超过最大范围
        # 对获得角度进行规范化
        next_angle = (next_angle + self.MAX_angle) % (2 * self.MAX_angle) - self.MAX_angle
        # 对获得角速度进行截断
        next_angle_velocity = np.clip(next_angle_velocity, -self.MAX_angle_velocity, self.MAX_angle_velocity)
        return np.array([next_angle, next_angle_velocity])

    def step(self, action):
        """
        :param action: 输入动作
        :return: 返回下一个状态，观测到的收益，是否达到目标
        """
        # 对动作进行截断
        action = np.clip(action, -self.MAX_voltage, self.MAX_voltage)[0]
        self.last_action = action
        # 该动作产生的收益
        reward = self.reward(self.state, action)
        # 状态空间进行更新
        self.state = self.transition(self.state, action)
        # 判断是否到达终止状态
        done = np.equal(self.state, np.array([0, 0])).all()
        return self.state, reward, done, {}

    def reset(self):
        """
        重置环境
        :return: 返回初始状态
        """
        self.state = np.array([-np.pi, 0])
        self.last_action = None
        return self.state

    def render(self, mode="human"):
        """
        动画渲染
        :param mode:
        :return:
        """
        if self.viewer is None:
            self.viewer = rendering.Viewer(500, 500)
            self.viewer.set_bounds(-2.2, 2.2, -2.2, 2.2)
            rod = rendering.make_capsule(1, .2)
            rod.set_color(.8, .3, .3)
            self.pole_transform = rendering.Transform()
            rod.add_attr(self.pole_transform)
            self.viewer.add_geom(rod)
            axle = rendering.make_circle(.05)
            axle.set_color(0, 0, 0)
            self.viewer.add_geom(axle)
            fname = path.join(path.dirname(__file__), "clockwise.png")
            self.img = rendering.Image(fname, 1., 1.)
            self.imgtrans = rendering.Transform()
            self.img.add_attr(self.imgtrans)

        self.viewer.add_onetime(self.img)
        self.pole_transform.set_rotation(self.state[0] + np.pi / 2)
        if self.last_action:
            self.imgtrans.scale = (-self.last_action / 2, np.abs(self.last_action) / 2)

        return self.viewer.render(return_rgb_array=mode == 'rgb_array')

    def close(self):
        """
        关闭渲染环境
        :return: None
        """
        if self.viewer:
            self.viewer.close()
            self.viewer = None

    @staticmethod
    def reward(state, action):
        """
        奖励函数
        :param state:  当前状态，state[0]: 角度. state[1]: 角速度
        :param action: 当前动作，输入电压
        :return: 当前动作状态下获得的奖励
        """
        # 奖励函数中两个参数
        Q_reward = np.matrix([[5, 0], [0, 0.1]])
        R_reward = 1.
        # 将状态转变为矩阵，纬度 2*1
        return float(-np.matmul(np.matmul(state, Q_reward), state) - R_reward * (action ** 2))


if __name__ == "__main__":
    # 定义一个环境
    env = InvertedPendulum()
    # 重复3次实验
    for i in range(3):
        # 初始化环境
        obs = env.reset()
        # 渲染环境
        env.render()
        # 停 1 s
        time.sleep(1)
        for t in range(300):
            env.render()
            # 输出观测
            print(obs)
            # 采样动作
            a = env.action_space.sample()
            # 根据动作更新状态
            obs, r, done, info = env.step(a)
            # 如果达到目标状态，停止此次实验
            if done:
                print("Finished!")
                break
    # 关闭环境
    env.close()
