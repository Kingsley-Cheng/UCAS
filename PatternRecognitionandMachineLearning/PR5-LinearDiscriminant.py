import numpy as np
from sklearn import model_selection
import matplotlib.pyplot as plt


def load_Gaussian_data(mu_1, mu_2, sigma_1, sigma_2, n):
    def generate_gaussian(mu, sigma, num):
        x = []
        for i in range(num):
            a = np.random.multivariate_normal(mu, sigma) + np.random.randn(1)
            x.append(a)
        return np.array(x)

    X1 = generate_gaussian(mu_1, sigma_1, int(n / 2))
    y1 = np.ones(X1.shape[0])
    X2 = generate_gaussian(mu_2, sigma_2, int(n / 2))
    y2 = np.zeros(X2.shape[0])
    X = np.vstack((X1, X2))
    y = np.hstack((y1, y2))
    X_1, X_2, y_1, y_2 = model_selection.train_test_split(X, y, test_size=0.3, random_state=42)
    return X_1, X_2, y_1, y_2


class Linear:
    def __init__(self, X, y):
        self.weight = None
        self.X = X
        self.y = y

    def test(self, X):
        X = np.hstack((X, np.ones((X.shape[0], 1))))
        output = np.dot(X, self.weight)
        w2_idx = np.where(output <= 0)
        prediction = np.ones(X.shape[0])
        prediction[w2_idx] = 0
        return prediction

    @staticmethod
    def accuracy(prediction, y):
        correct = 0.0
        total = y.shape[0]
        correct += np.sum(prediction == y)
        print("Accuracy: ", correct / total)

    def Normalize(self):
        pass


class Perceptron(Linear):
    def __init__(self, X, y):
        super().__init__(X, y)

        self.Normalize()
        self.weight = np.zeros(self.X.shape[1])

    def Normalize(self):
        n = self.X.shape[0]
        w2_idx = np.where(self.y == 0)
        self.X = np.hstack((self.X, np.ones((n, 1))))
        self.X[w2_idx, :] *= -1

    def Batch_train(self, lr=0.5, margin=0., iteration_times=1000, epsilon=1e-2):
        k = 0
        while True:
            k += 1
            lr_k = lr / k
            misclassified_idx = np.where(np.dot(self.X, self.weight) <= margin)[0]
            increments = lr_k * np.sum(self.X[misclassified_idx], axis=0)
            self.weight += increments
            if np.linalg.norm(increments, ord=2) / self.X.shape[0] < epsilon or k >= iteration_times:
                print("iterations: ", k)
                break

    def train(self, lr=0.5, margin=0., iteration_times=1000, epsilon=1e-3):
        k = 0
        while True:
            update = 0
            k += 1
            lr_k = lr / k
            misclassified_idx = np.where(np.dot(self.X, self.weight) <= margin)[0]
            for idx in misclassified_idx:
                increments = self.X[idx]
                update += increments
                self.weight += lr_k * increments
            if np.linalg.norm(update, ord=2) / self.X.shape[0] < epsilon or k >= iteration_times:
                print("iterations: ", k)
                break


class MSE(Linear):
    def __init__(self, X, y):
        super().__init__(X, y)
        self.Normalize()
        self.weight = np.zeros(self.X.shape[1])

    def Normalize(self):
        n = self.X.shape[0]
        w1_idx = np.where(self.y == 1)[0]
        w2_idx = np.where(self.y == 0)[0]
        self.X = np.hstack((self.X, np.ones((n, 1))))
        self.X[w2_idx, :] *= -1
        self.y[w2_idx] = w2_idx.shape[0] / n
        self.y[w1_idx] = w1_idx.shape[0] / n

    def train(self, lr=1., iteration_times=1000, epsilon=1e-3):
        k = 0
        while True:
            k += 1
            lr_k = lr / (k * self.X.shape[0])
            increments = np.dot(self.X.T, (self.y - np.dot(self.X, self.weight)).T)
            self.weight += lr_k * increments
            if np.linalg.norm(increments, ord=2) < epsilon or k >= iteration_times:
                print("iterations: ", k)
                break

    def Ho_Kashyap(self, lr=1, iteration_times=1000, epsilon=0.05):
        k = 0
        while True:
            k += 1
            lr_k = lr / k
            e = np.dot(self.X, self.weight) - self.y
            e_positive = (e + abs(e)) / 2
            self.y += 2 * lr_k * e_positive
            self.weight = np.dot(np.dot(np.linalg.pinv(np.dot(self.X.T, self.X)), self.X.T), self.y)
            if np.linalg.norm(lr_k * abs(e), ord=2) < epsilon or k >= iteration_times:
                print("iterations: ", k)
                break


def plot_image(X, y, y_pred):
    fig = plt.figure()
    ax = fig.add_subplot(121)
    ax.scatter(X[:, 0], X[:, 1], c=y, alpha=0.6)
    ax = fig.add_subplot(122)
    ax.scatter(X[:, 0], X[:, 1], c=y_pred, alpha=0.6)
    plt.show()


if __name__ == "__main__":
    mu1 = [-1, -2]
    mu2 = [1, 2]
    sigma1 = [[5, 0.5], [0.5, 1]]
    sigma2 = [[1, -0.5], [-0.5, 5]]
    n_samples = 2000

    X_train, X_test, y_train, y_test = load_Gaussian_data(mu1, mu2, sigma1, sigma2, n_samples)

    # Perceptron

    perceptron = Perceptron(X_train, y_train)
    perceptron.Batch_train(margin=0.3, lr=1, iteration_times=1000, epsilon=1e-2)
    # perceptron.train(margin=0.3, lr=1, iteration_times=1000, epsilon=1e-2)
    predicts = perceptron.test(X_test)
    plot_image(X_test, y_test, predicts)
    perceptron.accuracy(predicts, y_test)
    print("\n")

    # MSE

    mse = MSE(X_train, y_train)
    # mse.train()
    mse.Ho_Kashyap()
    predicts = mse.test(X_test)
    plot_image(X_test, y_test, predicts)
    mse.accuracy(predicts, y_test)
