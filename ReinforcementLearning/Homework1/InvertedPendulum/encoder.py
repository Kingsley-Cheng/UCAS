"""
径向基函数编码器
"""
import numpy as np
import itertools

class Encoder():
    def __init__(self,  num_angle = 10, num_angle_velocity= 10):

        MAX_angle = np.pi
        MAX_angle_velocity = 15*np.pi
        # 位置
        angle_center = np.linspace(-MAX_angle,-MAX_angle,num_angle)
        angle_velocity_center = np.linspace(-MAX_angle_velocity, MAX_angle_velocity, num_angle_velocity)
        self.num_angle = num_angle
        self.num_angle_velocity = num_angle_velocity

        self.sigma = np.matrix([[2*MAX_angle/(num_angle-1),0], [0, 2* MAX_angle_velocity/(num_angle_velocity-1)]])
        self.feature = self.generate_feature(angle_center, angle_velocity_center)
        self.action = np.array([-3,0,3])

    def generate_feature(self, angle_center, angle_velocity_center):
        product = itertools.product(angle_center,angle_velocity_center)
        result = []
        for elem in product:
            result.append([elem[0], elem[1]])
        return np.array(result)
    
    def encode(self,state, action):
        dim = self.feature.shape[0]
        result1 = np.zeros((dim,1))
        for i in range(dim):
            tmp = np.matrix(state - self.feature[i,:])
            tmp = -1/2 * (tmp*np.linalg.inv(self.sigma)*tmp.T)
            tmp = np.exp(tmp)
            result1[i] = tmp
        state = result1/np.sum(result1, axis=0)
        state = state[:,0]

        idx_action = []
        for i in self.action:
            idx_action.append(1 if i == action else 0)
        product = itertools.product(state, idx_action)
        result = [[],[],[]]
        i = 0
        for elem in product:
            if elem[1] == 1:
                result[i%3].append(elem[0])
            else:
                result[i%3].append(0)
            i += 1
        result = np.array(result).ravel()
        return result

if __name__ == "__main__":
    encoder = Encoder()
    rep = encoder.encode([np.pi, 0], 3)
    print(rep.shape)