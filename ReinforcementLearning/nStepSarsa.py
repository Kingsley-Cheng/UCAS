import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm # tqdm 是显示循环进度条的库

class CliffWalkingEnv:
    def __init__(self, ncol, nrow):
        self.nrow = nrow
        self.ncol = ncol
        self.x = 0 # 记录当前智能体位置的横坐标
        self.y = self.nrow - 1 # 记录当前智能体位置的纵坐标

    def step(self, action): # 外部调用这个函数来改变当前位置
        # 4 种动作, change[0]:上, change[1]:下, change[2]:左, change[3]:右。坐标系原点(0, 0)
        # 定义在左上角
        change = [[0, -1], [0, 1], [-1, 0], [1, 0]]
        self.x = min(self.ncol - 1, max(0, self.x + change[action][0]))
        self.y = min(self.nrow - 1, max(0, self.y + change[action][1]))
        next_state = self.y * self.ncol + self.x
        reward = -1
        done = False
        if self.y == self.nrow - 1 and self.x > 0: # 下一个位置在悬崖或者是目标
            done = True
            if self.x != self.ncol - 1:
                reward = -100
        return next_state, reward, done
    
    def reset(self): # 回归初始状态，坐标轴原点在左上角
        self.x = 0
        self.y = self.nrow - 1
        return self.y * self.ncol + self.x


class nstep_Sarsa:
    """ n 步 Sarsa 算法"""
    def __init__(self, n, ncol, nrow, epsilon, alpha, gamma, n_action=4):
        self.Q_table = np.zeros([ncol*nrow, n_action])
        self.n_action = n_action
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.n = n # 采用 n 步 Sarsa 算法
        self.state_list = [] # 保存之前的状态
        self.action_list = [] # 保存之前的动作
        self.reward_list = [] # 保存之前的奖励

    def take_action(self, state):
        if np.random.random() < self.epsilon:
            action = np.random.randint(self.n_action)
        else:
            action = np.argmax(self.Q_table[state])
        return action
    
    def best_action(self, state): # 用于打印策略
        Q_max = np.max(self.Q_table[state])
        a = [0 for _ in range(self.n_action)]
        for i in range(self.n_action):
            if self.Q_table[state, i] == Q_max:
                a[i] = 1
        return a
    
    def update(self, s0, a0, r, s1, a1, done):
        self.state_list.append(s0)
        self.action_list.append(a0)
        self.reward_list.append(r)
        if len(self.state_list) == self.n: # 若保存的数据可以进行 n 步更新
            G = self.Q_table[s1, a1] # 得到 Q(s_{t+n},a_{t+n})
            for i in reversed(range(self.n)):
                G = self.gamma * G + self.reward_list[i] # 不断向前计算每一步的回报
                # 如果到达终止状态，最后几步虽然长度不够 n 步，也将其进行更新
                if done and i > 0:
                    s = self.state_list[i]
                    a = self.action_list[i]
                    self.Q_table[s, a] += self.alpha * (G - self.Q_table[s, a])
            s = self.state_list.pop(0)
            a = self.action_list.pop(0)
            self.reward_list.pop(0)
            # n 步 Sarsa 的主要更新步骤
            self.Q_table[s, a] += self.alpha * (G - self.Q_table[s, a])
        if done: # 如果到达终止状态，即将开始下一条序列，则将列表清空
            self.state_list = []
            self.action_list = []
            self.reward_list = []

def print_agent(agent, env, action_meaning, disaster=[], end=[]):
    for i in range(env.nrow):
        for j in range(env.ncol):
            if (i * env.ncol + j) in disaster:
                print('****', end=" ")
            elif (i * env.ncol + j) in end:
                print('EEEE', end=" ")
            else:
                a = agent.best_action(i * env.ncol + j)
                pi_str = ''
                for k in range(len(action_meaning)):
                    pi_str += action_meaning[k] if a[k] > 0 else 'o'
                print(pi_str, end=" ")
        print()

if __name__ == "__main__":
    ncol = 12
    nrow = 4
    env = CliffWalkingEnv(ncol, nrow)
    np.random.seed(0)
    n_step = 5 # 5 步 Sarsa 算法
    alpha = 0.1
    epsilon = 0.1
    gamma = 0.9
    agent = nstep_Sarsa(n_step, ncol, nrow, epsilon, alpha, gamma)
    num_episodes = 500 # 智能体在环境中运行的序列数量

    return_list = [] # 记录每一条序列的回报
    for i in range(10): # 显示 10 个进度条
        # tqdm 的进度条功能
        with tqdm(total=int(num_episodes/10), desc='Iteration %d' % i) as pbar:
            for i_episode in range(int(num_episodes/10)): # 每个进度条的序列数
                episode_return = 0
                state = env.reset()
                action = agent.take_action(state)
                done = False
                while not done:
                    next_state, reward, done = env.step(action)
                    next_action = agent.take_action(next_state)
                    episode_return += reward # 这里回报的计算不进行折扣因子衰减
                    agent.update(state, action, reward, next_state, next_action)
                    state = next_state
                    action = next_action
                return_list.append(episode_return)
                if (i_episode+1) % 10 == 0: # 每 10 条序列打印一下这10条序列的平均回报
                    pbar.set_postfix({'episode':'%d' % (num_episodes / 10 * i + 
                                                        i_episode + 1), 'return': '%.3f' % np.mean(return_list[-10:])})
                pbar.update(1)


    action_meaning = ["^", 'v', "<", ">"]
    print("Sarsa 算法最终收敛得到的策略为: ")
    print_agent(agent, env, action_meaning, list(range(37, 47)), [47])
    episode_list = list(range(len(return_list)))
    plt.plot(episode_list, return_list)
    plt.xlabel('Episodes')
    plt.ylabel('Returns')
    plt.title('Sara on {}'.format('Cliff Walking'))
    plt.show()
