from matplotlib import pyplot as plt
import numpy as np
from scipy.stats import poisson
import seaborn as sns
import copy

MAX_PARK_NUM = 20 # 每个停车场的最大停车数目
lambda1_rent = 3 # 租车点 1 的租车参数
lambda2_rent = 4 # 租车点 2 的租车参数
lambda1_return = 3 # 租车点 1 的还车参数
lambda2_return = 2 # 粗车点 2 的还车参数
MAX_MOVE = 5 # 最多移动车数
DISCOUNT = 0.9 # 折扣因子
MOVE_COST = 2 # 移车代价
RENT_INCOME = 10 # 租车收益
UPPER_BOUND = 6 # n 超过上限认为概率为 0
poisson_prob_dict = dict()
actions = np.arange(-MAX_MOVE, MAX_MOVE + 1)


def poisson_cal(n,lam):
    global poisson_prob_dict
    key = n * 10 + lam
    if key not in poisson_prob_dict:
        poisson_prob_dict[key] = poisson.pmf(n,lam)
    return poisson_prob_dict[key]

def CalculateReward(state,action,state_value):
    returns = 0
    returns -= MOVE_COST * abs(action)

    NUM_LOC1 = min(state[0] - action, MAX_PARK_NUM)
    NUM_LOC2 = min(state[1] + action,MAX_PARK_NUM)

    for rent1 in range(UPPER_BOUND):
        for rent2 in range(UPPER_BOUND):

            rent_prob = poisson_cal(rent1,lambda1_rent)* poisson_cal(rent2,lambda2_rent)

            num_loc1 = NUM_LOC1
            num_loc2 = NUM_LOC2

            valid_rent1 = min(num_loc1, rent1)
            valid_rent2 = min(num_loc2, rent2)

            reward = RENT_INCOME*(valid_rent1+valid_rent2)

            num_loc1 -= valid_rent1
            num_loc2 -= valid_rent2

            for retrun1 in range(UPPER_BOUND):
                for return2 in range(UPPER_BOUND):
                    return_prob = poisson_cal(retrun1,lambda1_return) * poisson_cal(return2,lambda2_return)

                    num_loc1_ = min(num_loc1+retrun1,MAX_PARK_NUM)
                    num_loc2_ = min(num_loc2+return2, MAX_PARK_NUM)

                    prob = return_prob * rent_prob
                    returns+= prob * (reward + DISCOUNT * state_value[num_loc1_,num_loc2_])
    return returns
                        
                     

class PolicyIteration:
    """
    Algorithm: Policy Iteration
    """
    def __init__(self, theta):
        self.v = np.zeros((MAX_PARK_NUM+1, MAX_PARK_NUM+1))
        self.pi = np.zeros(self.v.shape,dtype=np.int8)
        self.theta = theta # 策略评估收敛阈值

    def policy_evaluation(self): # 策略评估
        cnt = 1 # 计数器
        while 1:
            old_v = self.v.copy()
            for i in range(MAX_PARK_NUM+1):
                for j in range(MAX_PARK_NUM+1):
                    new_state_value = CalculateReward([i,j],self.pi[i,j], self.v)
                    self.v[i,j] = new_state_value 
            max_diff = np.max(abs(old_v - self.v))
            print(max_diff)
            if max_diff < self.theta:
                break
            cnt += 1
        print("策论评估进行 %d 轮后完成"% cnt)
        print(self.v)
    
    def policy_improvement(self): # 策略提升
        for i in range(MAX_PARK_NUM+1):
            for j in range(MAX_PARK_NUM+1):
                qsa_list = []
                for a in actions:
                    if (0<=a<=i) or (-j<=a<=0):
                        qsa_list.append(CalculateReward([i, j], a, self.v))
                    else:
                        qsa_list.append(-np.inf)
                new_action = actions[np.argmax(qsa_list)]
                self.pi[i,j] = new_action 
    
    def policy_iteration(self): # 策略迭代
        while 1:
            self.policy_evaluation()
            old_pi = copy.deepcopy(self.pi) # 将列表进行深拷贝，方便接下来进行比较
            self.policy_improvement()
            if (old_pi == self.pi).all():
                break


if __name__ == "__main__":
    agent = PolicyIteration(0.0001)
    agent.policy_iteration()
    print(agent.pi)
    # sns.heatmap(agent.pi)
    # plt.show()
