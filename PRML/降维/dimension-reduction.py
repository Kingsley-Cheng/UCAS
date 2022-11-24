# imports
import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets
from torch import nn
from matplotlib.ticker import NullFormatter


class DimensionReduction:
    def __init__(self, d):
        # k 为该方法维数缩减后的维度
        self.d = d

    @staticmethod
    def distance_matrix(data):
        num_data = data.shape[0]
        distance_matrix = np.zeros((num_data, num_data))
        for i in range(num_data):
            for j in range(i):
                distance_matrix[i][j] = np.linalg.norm(data[i] - data[j], ord=2)
                distance_matrix[j][i] = distance_matrix[i][j]
        return distance_matrix


class MDS(DimensionReduction):

    def train(self, dist_matrix):
        n_samples = dist_matrix.shape[0]
        # 中心化
        B = np.zeros((n_samples, n_samples))
        for i in range(n_samples):
            for j in range(n_samples):
                disti_ = np.sum(dist_matrix[i, :]) / n_samples
                dist_j = np.sum(dist_matrix[:, j]) / n_samples
                dist = np.sum(dist_matrix) / (n_samples ** 2)
                B[i][j] = -1 / 2 * (dist_matrix[i][j] ** 2 - disti_ ** 2 - dist_j ** 2 + dist ** 2)
                B[j][i] = B[i][j]
        eigval, eigvec = np.linalg.eig(B)
        idx = np.argsort(-eigval)[:self.d]
        gamma = np.diag(np.sqrt(eigval[idx]))
        V = eigvec[:, idx]
        return np.transpose(np.dot(gamma, np.transpose(V)))


class PCA(DimensionReduction):

    @staticmethod
    def gaussian(data, sigma):
        n_samples = data.shape[0]
        K = np.zeros((n_samples, n_samples))
        for i in range(n_samples):
            for j in range(i):
                K[i][j] = np.exp(-(np.linalg.norm(data[i] - data[j], ord=2) ** 2) / (2 * sigma ** 2))
                K[j][i] = K[i][j]
        return K

    @staticmethod
    def laplacian(data, gamma):
        n_samples = data.shape[0]
        K = np.zeros((n_samples, n_samples))
        for i in range(n_samples):
            for j in range(i):
                K[i][j] = np.exp(-np.linalg.norm(data[i] - data[j], ord=2) ** 2 / gamma)
                K[j][i] = K[i][j]
        return K

    @staticmethod
    def linear(data):
        K = np.dot(data, np.transpose(data))
        return K

    def train(self, data, kernel="linear", coef=None):
        n_features = data.shape[1]
        assert n_features > self.d
        if kernel == 'laplacian' and coef is not None:
            K = self.laplacian(data, coef)
        elif kernel == 'gaussian' and coef is not None:
            K = self.gaussian(data, coef)
        else:
            K = self.linear(data)

        K = K - np.mean(K, axis=0)
        eigvalue, eigvec = np.linalg.eig(K)
        W = eigvec[:, :self.d]
        return W


class ISOMAP(DimensionReduction):
    def __init__(self, d, k=10):
        super().__init__(d)
        self.k = k

    @staticmethod
    def Dijkstra(distance_matrix, i):
        n_samples = distance_matrix.shape[0]
        passed = [i]
        nopass = [x for x in range(n_samples) if x != i]
        dist = distance_matrix[i]
        while len(nopass):
            idx = nopass[0]
            # 寻找 nopass 指标集下 dist 最小的指标
            for i in nopass:
                if dist[i] < dist[idx]:
                    idx = i
            # 更新指标集
            passed.append(idx)
            nopass.remove(idx)

            for i in nopass:
                length = dist[idx] + distance_matrix[idx][i]
                if length < dist[i]:
                    dist[i] = length
        return dist

    def train(self, distance_matrix, data):
        n_samples = data.shape[0]
        assert self.k < n_samples
        KNmatrix = np.ones((n_samples, n_samples)) * np.inf
        W = np.ones((n_samples, n_samples)) * np.inf
        for i in range(n_samples):
            k_neighbours = np.argpartition(distance_matrix[i], self.k)[:self.k + 1]
            KNmatrix[i][k_neighbours] = distance_matrix[i][k_neighbours]
        for i in range(n_samples):
            W[i] = self.Dijkstra(KNmatrix, i)

        assert np.sum(np.isinf(W)) == 0, "exist inf, adjust K-neighbours"
        mds = MDS(self.d)
        return mds.train(W)


class LLE(DimensionReduction):
    def __init__(self, d, k):
        super().__init__(d)
        self.k = k

    def calculate_weight(self, data, index_i, k_nears):
        D = data.shape[1]
        if self.k > D:
            tol = 1e-3
        else:
            tol = 0
        xi = data[index_i]
        xj = data[k_nears]
        A = xi - xj
        Zi = np.dot(A, A.T)
        Zi = Zi + np.eye(self.k) * tol * np.trace(Zi)
        INVZ = np.linalg.pinv(Zi)
        a = np.dot(INVZ, np.ones((self.k, 1)))
        b = np.dot(np.ones((1, self.k)), a)[0, 0]
        return a / b

    def train(self, distance_matrix, data):
        n_samples = distance_matrix.shape[0]
        W = np.zeros((n_samples, n_samples))
        for i in range(n_samples):
            k_nears = np.argsort(distance_matrix[i])[1:self.k + 1]
            WI = self.calculate_weight(data, i, k_nears)
            print(WI)
            for j in range(self.k):
                W[i][k_nears[j]] = WI[j]
        M = np.dot(np.transpose(np.eye(n_samples) - W), (np.eye(n_samples) - W))
        eigval, eigvec = np.linalg.eig(M)
        index = np.argsort(np.abs(eigval))[1:self.d + 1]
        return eigvec[:, index]


class LE(DimensionReduction):
    def __init__(self, d, k):
        super().__init__(d)
        self.k = k

    def train(self, distance_matrix, data, method="Heat kernel", gamma=None):
        n_samples = data.shape[0]
        W = np.zeros((n_samples, n_samples))
        D = np.zeros((n_samples, n_samples))
        for i in range(n_samples):
            k_nears = np.argsort(distance_matrix[i])[1:self.k + 1]
            for j in k_nears:
                if i != j:
                    if method == "Heat kernel" and gamma is not None:
                        W[i][j] = np.exp(-np.linalg.norm(data[i] - data[j], ord=2) / gamma ** 2)
                    else:
                        W[i][j] = 1
            D[i][i] = sum(W[i])
        L = D - W
        eigval, eigvec = np.linalg.eig(L)
        index = np.argsort(np.abs(eigval))[1:self.d + 1]
        return eigvec[:, index]


class AutoEncoder(nn.Module):
    def __init__(self):
        super(AutoEncoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(28 * 28, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 12),
            nn.ReLU(),
            nn.Linear(12, 3)
        )
        self.decoder = nn.Sequential(
            nn.Linear(3, 12),
            nn.ReLU(),
            nn.Linear(12, 64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, 28 * 28),
            nn.Tanh()
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x


def sample_test(sample_num, random_state=0):
    X, t = datasets.make_swiss_roll(sample_num, random_state=random_state)
    fig = plt.figure()
    ax = fig.add_subplot(121, projection='3d')
    ax.scatter(X[:, 0], X[:, 1], X[:, 2], c=t)
    iso = ISOMAP(2, 10)
    dist_matrix = iso.distance_matrix(X)
    X_new = iso.train(dist_matrix, X)
    # le = LE(2, 20)
    # dist_matrix = le.distance_matrix(X)
    # X_new = le.train(dist_matrix, X, gamma=4)
    ax = fig.add_subplot(122)
    ax.scatter(X_new[:, 0], X_new[:, 1], c=t)
    ax.xaxis.set_major_formatter(NullFormatter())
    ax.yaxis.set_major_formatter(NullFormatter())
    plt.axis('tight')
    plt.show()

    # data = datasets.make_s_curve(sample_num, random_state=random_state)
    # X, t = data
    # fig = plt.figure(figsize=(12, 8))
    # ax = fig.add_subplot(121, projection='3d')
    # ax.scatter(X[:, 0], X[:, 1], X[:, 2], c=t)
    # mds = MDS(2)
    # dist_matrix = mds.distance_matrix(X)
    # X_new = mds.train(dist_matrix)
    # isomap = ISOMAP(2, 9)
    # dist_matrix = isomap.distance_matrix(X)
    # X_new = isomap.train(dist_matrix, X)
    # lle = LLE(2, 10)
    # dist_matrix = lle.distance_matrix(X)
    # X_new = lle.train(dist_matrix, X)
    # ax = fig.add_subplot(122)
    # ax.scatter(X_new[:, 0], X_new[:, 1], c=t)
    # plt.show()


if __name__ == "__main__":
    n = 1000
    sample_test(n)
