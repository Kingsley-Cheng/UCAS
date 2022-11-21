# imports
import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets


class DimensionReduction:
    def __init__(self, k):
        # k 为该方法维数缩减后的维度
        self.k = k


class MDS(DimensionReduction):
    def __init__(self, k, p):
        super().__init__(k)
        self.p = p

    def distance_matrix(self, data):
        num_data = data.shape[0]
        distance_matrix = np.zeros((num_data, num_data))
        for i in range(num_data):
            for j in range(i):
                distance_matrix[i][j] = np.linalg.norm(data[i] - data[j], ord=self.p)
                distance_matrix[j][i] = distance_matrix[i][j]
        return distance_matrix

    @staticmethod
    def calculate_disti_(distance_matrix, i):
        num = distance_matrix.shape[0]
        dist2_i = np.sum(distance_matrix[i, :]) / num
        return dist2_i

    @staticmethod
    def calculate_dist_j(distance_matrix, j):
        num = distance_matrix.shape[0]
        dist2_j = np.sum(distance_matrix[:, j]) / num
        return dist2_j

    @staticmethod
    def calculate_dist(distance_matrix):
        num = distance_matrix.shape[0]
        dist2_j = np.sum(distance_matrix) / (num ** 2)
        return dist2_j

    def train(self, data):
        n_samples = data.shape[0]
        n_features = data.shape[1]
        assert n_features > self.k
        # 中心化
        data = data - np.mean(data, axis=0)
        dist_matrix = self.distance_matrix(data)
        B = np.zeros((n_samples, n_samples))
        for i in range(n_samples):
            for j in range(n_samples):
                B[i][j] = -1 / 2 * (dist_matrix[i][j] ** 2 - self.calculate_disti_(dist_matrix, i) ** 2
                                    - self.calculate_dist_j(dist_matrix, j) ** 2 + self.calculate_dist(
                            dist_matrix) ** 2)
        eigval, eigvec = np.linalg.eig(B)
        gamma = np.diag(np.sqrt(eigval[:self.k]))
        V = eigvec[:, :self.k]
        return np.transpose(np.dot(gamma, np.transpose(V)))


class PCA(DimensionReduction):
    def train(self, data):
        n_features = data.shape[1]
        assert n_features > self.k
        # 中心化
        data = data - np.mean(data, axis=0)
        corr_matrix = np.dot(data, np.transpose(data))
        eigvalue, eigvec = np.linalg.eig(corr_matrix)
        W = eigvec[:, :self.k]
        return W


def sample_test(sample_num, random_state=17):
    data = datasets.make_swiss_roll(sample_num, random_state=random_state)
    X, t = data
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(121, projection='3d')
    ax.scatter(X[:, 0], X[:, 1], X[:, 2], c=t, cmap=plt.cm.hot)
    pca = PCA(k=2)
    X_new = pca.train(X)
    ax = fig.add_subplot(122)
    print(X_new)
    ax.scatter(X_new[:, 0], X_new[:, 1], c=t, cmap=plt.cm.hot)
    plt.show()


if __name__ == "__main__":
    n = 500
    sample_test(n)
