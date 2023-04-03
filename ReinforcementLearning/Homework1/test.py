import numpy as np
import itertools

WIDTH = 100
HEIGHT = 100
MASS = 1
SPEED_LIMIT = 35
NUM_FEATURE = 5
ACTION_LIST = [[-5,0], [0,-5], [5, 0], [0,5]]

T = 30
T_ACTION = 0.1
T_ENV = 0.01

class Curling:
    def __init__(self):
        self._width = WIDTH
        self._height = HEIGHT
        self._radius = 1
        self._mass = MASS
        self._Ts = 0.01
        self.alpha = 0.9

        self.target = None
        self.speed = None
        self.position = None
        self.f = None
        self.accelerate = None
        self.distance = None
        self.r = None
        self.reset()

    def reset(self, target=None, speed=None, position=None):
        self.target = target if target is not None else np.random.random(2) * (np.array([self._width, self._height])-2*self._radius)+1

        self.speed = speed if speed is not None else np.random.random(2)*20.0-10.0

        self.position = position if position is not None else np.random.random(2) * (np.array([self._width, self._height])-2*self._radius)+1
       
        self.f = 0.005 * np.power(self.speed, 2) * -np.sign(self.speed) if self.speed.all() else np.zeros(2)

        self.accelerate = self.f / self._mass
        self.distance = self.position - self.target
        self.r = self.reward()
    
    def step(self, action):
        self.f = 0.005 * np.power(self.speed, 2) * -np.sign(self.speed) + action if self.speed.all() else action

        self.accelerate = self.f / self._mass
        self.position += self.speed * self._Ts + self.accelerate * (self._Ts**2)/2
        self.speed += self.accelerate * self._Ts
        self.distance = self.position - self.target
        self.is_bound()
        self.r = self.reward()

    def is_bound(self):
        x = self.position[0]
        y = self.position[1]
        if (x<self._radius) or (x>self._width - self._radius):
            self.speed[0] *= -self.alpha
            x = 2*self._radius - x if x <self._radius else 2*(self._width-self._radius)-x
        
        if(y<self._radius) or (y>self._height-self._radius):
            self.speed[1] *= -self.alpha
            y = 2*self._radius - y if y <self._radius else 2*(self._height-self._radius) - y
        self.position = np.array([x,y])
    
    def reward(self):
        return -float(np.sqrt(sum(self.distance * self.distance)))

    
    def print(self):
        print(self.target)
        print(self.speed) 
        print(self.position)
        print(self.f)
        print(self.accelerate)
        print(self.distance)
        print(self.reward())
    
class RBF:
    def __init__(self,  num_features=NUM_FEATURE, action_list = ACTION_LIST):
        # 位置
        p_center = np.array([np.linspace(-WIDTH,WIDTH,num_features), np.linspace(-HEIGHT,HEIGHT, num_features)])

        speed_center = np.array([np.linspace(-SPEED_LIMIT, SPEED_LIMIT, num_features),np.linspace(-SPEED_LIMIT, SPEED_LIMIT, num_features)])

        self.sigma = np.array([2*WIDTH/(num_features-1), 2*HEIGHT/(num_features-1), 2*SPEED_LIMIT/(num_features-1), 2 * SPEED_LIMIT/(num_features-1)])
        self.feature = self.generate_feature(p_center, speed_center)
        self.action = action_list

    def generate_feature(self, p_center, speed_center):
        p_x = p_center[0]
        p_y = p_center[1]
        v_x = speed_center[0]
        v_y = speed_center[1]
        product = itertools.product(itertools.product(itertools.product(p_x,p_y),v_x), v_y)
        result = []
        for elem in product:
            result.append([elem[0][0][0], elem[0][0][1], elem[0][1], elem[1]])
        return np.array(result)
    
    def encode(self,state, action):
        state = np.sum((state - self.feature)**2 / self.sigma**2,axis=1)
        state = np.exp(-state/2)
        state = state/np.sum(state, axis=0)

        idx_action = []
        for i in self.action:
            idx_action.append(1 if i == action else 0)
        product = itertools.product(state, idx_action)
        result = [[],[],[],[]]
        i = 0
        for elem in product:
            if elem[1] == 1:
                result[i%4].append(elem[0])
            else:
                result[i%4].append(0)
            i += 1
        result = np.array(result)
        return result.ravel().reshape(2500,1)

class Sampling:
    def __init__(self, rbf, action_list =ACTION_LIST,is_train = False):
        self.rbf = rbf
        self.action_list = action_list
        self.is_train = is_train

    def epsilon_greedy(self, state, weight, epsilon=0.1):
        if self.is_train and np.random.random()<=epsilon:
            return self.action_list[np.random.choice([0,1,2,3])]
        else:
            q = []
            for i in self.action_list:
                q.append(np.matmul(self.rbf.encode(state, i).T, weight))
            q = np.array(q)
        return self.action_list[np.argmax(q)]
    
class Sarsa:
    def __init__(self, env, alpha, lam,gamma,rbf, sampler, path="./pw.npy"):
        self.alpha = alpha
        self.path = path
        self.gamma = gamma
        self.lam = lam
        self.env = env
        self.rbf = rbf
        self.sampler = sampler
        try:
            self.weight = np.load(path)
        except:
            # self.weight = np.zeros((2500,1))
            print("error")

    
    def update(self, episodes):
        step = T_ACTION/T_ENV
        episode = 0
        while True:
            episode += 1
            state = [self.env.distance[0], self.env.distance[1], self.env.speed[0], self.env.speed[1]]
            et = np.zeros_like(self.weight)
            self.env.reset()
            a0 = self.sampler.epsilon_greedy(state, self.weight)
            feature = self.rbf.encode(state, a0)
            self.env.step(a0)
            w_hist = []
            r_hist = []
            for t in range(int(T/T_ENV)):
                if (t%step == 0):
                    new_state = [self.env.distance[0], self.env.distance[1], self.env.speed[0], self.env.speed[1]]
                    a1 = self.sampler.epsilon_greedy(new_state, self.weight)
                    feature_new = self.rbf.encode(new_state, a1)
                    delta = self.env.r + self.gamma * np.matmul(feature_new.T, self.weight) - np.matmul(feature.T, self.weight)
                    et = self.gamma*self.lam*et + feature

                    self.weight += self.alpha*delta*et
                    a0 = a1
                    feature = feature_new
                    state = new_state
                    w_hist.append(self.weight)
                    r_hist.append(self.env.r)
                    # print(a0)
                self.env.step(a0)
                # self.env.print()
            w_hist = np.array(np.sum(np.abs(delta)))
            r_hist = np.array(r_hist)
            print(f"episode {episode+1}: state: {state}, reward:{np.sum(r_hist)}")
            print(f"weight:{r_hist[-1]}")
            if(episode+1)%10 == 0:
                print("save weight")
                np.save(self.path, self.weight)


if __name__ == "__main__":
    curling = Curling()
    rbf = RBF()
    sampler = Sampling(rbf, is_train=True)
    agent = Sarsa(curling, 0.3,0.9,0.9,rbf,sampler)
    agent.update(10000)

