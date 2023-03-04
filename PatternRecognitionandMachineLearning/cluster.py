import random
import numpy as np
from sklearn import datasets
import matplotlib.pyplot as plt
import scipy.stats as stats
import math


class Cluster:
    def __init__(self, n_clusters=None):
        self.n_clusters = n_clusters

    @staticmethod
    def external_evaluate_index(predict, label):
        num_labels = label.shape[0]
        num_predict = predict.shape[0]
        a, b, c, d = 0, 0, 0, 0
        if num_predict == num_labels:
            for i in range(num_labels):
                for j in range(i + 1, num_labels):
                    if predict[i] == predict[j]:
                        if label[i] == label[j]:
                            a += 1
                        else:
                            b += 1
                    else:
                        if label[i] == label[j]:
                            c += 1
                        else:
                            d += 1
        return a, b, c, d

    def JC(self, predict, label):
        a, b, c, _ = self.external_evaluate_index(predict, label)
        index = a / (a + b + c)
        print("Jaccard Coefficient：", index)

    def FMI(self, predict, label):
        a, b, c, _ = self.external_evaluate_index(predict, label)
        index = math.sqrt((a / (a + b)) * (a / (a + c)))
        print("Fowlkes and Mallows Index：", index)

    def RD(self, predict, label):
        num_predict = predict.shape[0]
        a, _, _, d = self.external_evaluate_index(predict, label)
        index = 2 * (a + d) / (num_predict * (num_predict - 1))
        print("Rand Index：", index)

    def get_clusters(self, data, predict):
        clusters = []
        for i in range(self.n_clusters):
            clusters.append(data[predict == i])
        return cluster

    @staticmethod
    def avg(clusters, p=2):
        num = len(cluster)
        dist = 0
        for i in range(num):
            for j in range(i + 1, num):
                dist += np.linalg.norm(clusters[i] - clusters[j], ord=p)
        return 2 / (num * (num - 1)) * dist

    @staticmethod
    def diam(clusters, p=2):
        num = len(clusters)
        max_dist = 0
        for i in range(num):
            for j in range(i + 1, num):
                dist = np.linalg.norm(clusters[i] - clusters[j], ord=p)
                if dist > max_dist:
                    max_dist = dist
        return max_dist

    @staticmethod
    def dmin(cluster_i, cluster_j, p=2):
        num_i = len(cluster_i)
        num_j = len(cluster_j)
        min_value = np.Inf
        for i in range(num_i):
            for j in range(num_j):
                dist = np.linalg.norm(cluster_i[i] - cluster_j[j], ord=p)
                if dist < min_value:
                    min_value = dist
        return min_value

    @staticmethod
    def dcen(cluster_i, cluster_j, p=2):
        mu_i = sum(cluster_i) / len(cluster_i)
        mu_j = sum(cluster_j) / len(cluster_j)
        return np.linalg.norm(mu_i - mu_j, ord=p)

    def DBI(self, clusters):
        DBI = 0
        for i in range(self.n_clusters):
            max_val = 0
            for j in range(self.n_clusters):
                if i != j:
                    dist = (self.avg(clusters[i]) + self.avg(clusters[j])) / self.dcen(clusters[i], clusters[j])
                    if dist > max_val:
                        max_val = dist
                DBI += max_val
        DBI = DBI / self.n_clusters
        print("Davies-Bouldin Index: ", DBI)

    def DI(self, clusters):
        DI = np.Inf
        max_diam = 0
        for i in range(self.n_clusters):
            diam = self.diam(clusters[i])
            if max_diam < diam:
                max_diam = diam
        for i in range(self.n_clusters):
            min_value = np.Inf
            for j in range(self.n_clusters):
                if i != j:
                    val = self.dmin(clusters[i], clusters[j]) / max_diam
                    if val < min_value:
                        min_value = val
            if min_value < DI:
                DI = min_value
        print("Dunn Index: ", DI)

    def evaluate_index(self, x, y, prediction, clusters):
        self.JC(pred, y)
        self.FMI(pred, y)
        self.RD(pred, y)
        self.DBI(clusters)
        self.DI(clusters)
        plt.scatter(x[:, 0], x[:, 1], c=prediction)


class KMeans(Cluster):
    """ 实现KMeans 算法 """

    def train(self, data, p=2, max_iter=10000):
        n_samples = data.shape[0]
        n_features = data.shape[1]
        centers = np.zeros((self.n_clusters, n_features))
        label = np.zeros(n_samples, dtype=int)
        # 初始化聚类中心
        for i in range(self.n_clusters):
            centers[i] = data[np.random.choice(n_samples)]

        for _ in range(max_iter):
            for j in range(n_samples):
                i = np.argmin(np.linalg.norm(centers - data[j], ord=p, axis=1, keepdims=True))
                label[j] = i

            center_old = centers.copy()
            for j in range(self.n_clusters):
                if len(np.where(label == j)) != 0:
                    centers[j] = np.mean(data[np.where(label == j)])
            if np.mean(np.abs(center_old - centers + 1e-10)) <= 1e-4:
                return label


class LVQ(Cluster):
    def __init__(self, n_clusters=None):
        super().__init__(n_clusters)
        self.p = None

    def train(self, data, label, max_iter=10000, lr=0.01):
        n_samples = data.shape[0]
        n_features = data.shape[1]
        tag = np.unique(label)
        n_labels = len(tag)
        self.p = np.zeros((self.n_clusters, n_features))
        label_p = np.zeros(self.n_clusters, dtype=int)

        # 初始化原型向量
        for i in range(self.n_clusters):
            self.p[i] = data[np.random.choice(np.where(label == labels[i % n_labels])[0])]
            label_p[i] = labels[i % n_labels]

        # 训练样本
        for _ in range(max_iter):
            p = self.p.copy()
            for j in range(n_samples):
                i = np.argmin(np.linalg.norm(self.p - data[j], ord=2, axis=1, keepdims=True))
                if label[j] == label_p[i]:
                    self.p[i] += lr * (data[j] - self.p[i])
                else:
                    self.p[i] -= lr * (data[j] - self.p[i])

            if np.mean(np.abs(p - self.p) < 1e-4):
                break

            return self.predict(data)

    def predict(self, x):
        n_test = x.shape[0]
        prediction = np.zeros(n_test, dtype=int)

        for j in range(n_test):
            pred[j] = np.argmin(np.linalg.norm(x[j] - self.p, ord=2, axis=1, keepdims=True))
        return prediction


# Gauss 混合模型
class GMM(Cluster):

    def train(self, data, max_iter=100):
        n_samples = data.shape[0]
        n_features = data.shape[1]
        alpha = np.zeros(self.n_clusters)
        mu = np.zeros((self.n_clusters, n_features))
        sigma = np.zeros((self.n_clusters, n_features, n_features))
        for i in range(self.n_clusters):
            alpha[i] = np.random.random(1) + 0.1
            mu[i] = np.random.random(n_features) * np.mean(data)
            sigma[i] = np.diag([1, 1])
        alpha = alpha / np.sum(alpha)
        for _ in range(max_iter):
            gamma = np.zeros((self.n_clusters, n_samples))
            for j in range(n_samples):
                for i in range(self.n_clusters):
                    gamma[i, j] = stats.multivariate_normal.pdf(data[j], mu[i], sigma[i])
                gamma[:, j] = gamma[:, j] / np.sum(gamma[:, j])
            alpha_old = alpha.copy()
            for i in range(self.n_clusters):
                mu[i] = np.dot(gamma[i], data) / np.sum(gamma[i])
                for j in range(n_samples):
                    sigma[i] += gamma[i, j] * np.dot(data[j] - mu[i], (data[j] - mu[i]).T)
                alpha[i] = np.sum(gamma[i]) / n_samples
            if np.mean(np.abs(alpha - alpha_old)) < 1e-4:
                prediction = np.argmax(gamma, axis=0)
                return prediction


class DBSCAN(Cluster):
    def __init__(self, MinPts=5, radius=1, n_clusters=None):
        super().__init__(n_clusters)
        self.MinPts = MinPts  # 核心对象最小样本数
        self.radius = radius  # 核心点领域大小

    def is_reachable(self, x, y):
        dist = math.sqrt(sum((x[i] - y[i]) ** 2 for i in range(len(x))))
        return dist <= self.radius

    # 判断 point 是否为核心点，并寻找一个点的领域中的点
    def seek_neighbours(self, point, data):
        neighbour = set()
        for index, pn in enumerate(data):
            if self.is_reachable(point, pn):
                neighbour.add(index)
        is_keypoint = len(neighbour) >= self.MinPts
        return is_keypoint, neighbour

    def train(self, data):
        un_visit = set([i for i in range(data.shape[0])])
        clusters = -1 * np.ones(data.shape[0])
        k = 0
        while len(un_visit):
            p = random.choice(list(un_visit))
            un_visit.remove(p)
            is_keypoint, neighbours = self.seek_neighbours(data[p], data)
            # 如果是核心点，则添加一个类簇
            if is_keypoint:
                clusters[p] = k
                # 核心点需要找其所有密度相连的点
                while neighbours:
                    index = neighbours.pop()
                    if index in un_visit:
                        un_visit.remove(index)
                        is_keypoint, neigh_of_neigh = self.seek_neighbours(data[index], data)
                        if is_keypoint:
                            neighbours |= neigh_of_neigh
                    clusters[index] = k
                k += 1
        self.n_clusters = k
        return clusters


class AGENS(Cluster):
    @staticmethod
    def dist_min(cluster_x, cluster_y):
        return min(np.linalg.norm(s1 - s2) for s1 in cluster_x for s2 in cluster_y)

    @staticmethod
    def dist_max(cluster_x, cluster_y):
        return max(np.linalg.norm(s1 - s2) for s1 in cluster_x for s2 in cluster_y)

    @staticmethod
    def dist_avg(cluster_x, cluster_y):
        return sum(np.linalg.norm(s1 - s2) for s1 in cluster_x for s2 in cluster_y) / (len(cluster_y) * len(cluster_x))

    @staticmethod
    def create_distance_matrix(data, clusters, distance_method):
        num = len(clusters)
        distance_matrix = np.zeros((num, num))
        for i in range(num):
            for j in range(i):
                distance_matrix[i][j] = distance_method(data[clusters[i]], data[clusters[j]])
                distance_matrix[j][i] = distance_matrix[i][j]
        return distance_matrix

    @staticmethod
    def find_min(distance_matrix):
        num = distance_matrix.shape[0]
        min_distance = distance_matrix[0][1]
        x = 0
        y = 1
        for i in range(num):
            for j in range(num):
                if i != j and distance_matrix[i][j] < min_distance:
                    min_distance = distance_matrix[i][j]
                    x = i
                    y = j
        return x, y, min_distance

    def train(self, data, distance_method="min"):
        method = {"min": self.dist_min, "max": self.dist_max, "avg": self.dist_avg}
        distance_method = method.get(distance_method)
        clusters = []
        n_samples = data.shape[0]
        for i in range(n_samples):
            clusters.append([i])
        distance_matrix = self.create_distance_matrix(data, clusters, distance_method)
        index = distance_matrix.shape[0]
        while index > self.n_clusters:
            id_x, id_y, min_distance = self.find_min(distance_matrix)
            clusters[id_x].extend(clusters[id_y])
            clusters.pop(id_y)
            distance_matrix = self.create_distance_matrix(data, clusters, distance_method)
            index -= 1
        prediction = np.zeros(n_samples)
        for k in range(self.n_clusters):
            pred[clusters[k]] = k
        return prediction


if __name__ == '__main__':
    # Create datasets
    sample_clusters = 5
    samples_num = 500
    random_state = 20
    X, labels = datasets.make_blobs(centers=sample_clusters, n_samples=samples_num, random_state=random_state)
    plt.scatter(X[:, 0], X[:, 1], alpha=0.8)
    # KMeans
    kmeans = KMeans(5)
    pred = kmeans.train(X)
    cluster = kmeans.get_clusters(labels, pred)
    kmeans.evaluate_index(X, labels, pred, cluster)

    # LVQ+KMeans
    # kmeans
    kmeans = KMeans(3)
    pred_k = kmeans.train(X)
    # lvq
    lvq = LVQ(5)
    pred_l = lvq.train(X, pred_k)
    clusters_l = lvq.get_clusters(labels, pred_l)
    lvq.evaluate_index(X, labels, pred_l, clusters_l)

    # GMM
    gmm = GMM(5)
    pred = gmm.train(X)
    cluster = gmm.get_clusters(X, pred)
    gmm.evaluate_index(X, labels, pred, cluster)

    # DBSCAN
    dbscan = DBSCAN()
    pred = dbscan.train(X)
    cluster = dbscan.get_clusters(X, pred)
    dbscan.evaluate_index(X, labels, pred, cluster)

    # AGENS
    agens = AGENS(5)
    pred = agens.train(X)
    cluster = agens.get_clusters(X, pred)
    agens.evaluate_index(X, labels, pred, cluster)
