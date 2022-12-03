"""
Author: Kingsley-Cheng
Time: 2022/12/03
Description: [贝叶斯分类算法实现]
"""

import logging
import math

# imports
import numpy as np
from sklearn import datasets
from sklearn import model_selection
from tqdm import tqdm

logging.basicConfig(format='%(asctime)s - %(levelname)s: %(message)s', level=logging.DEBUG)


# 判别函数采用 Gauss
class BayesClassifier:
    def __init__(self):
        self.param = dict()
        pass

    def train(self, X, y, labels):
        logging.info("Calculate prior probability")
        n_sample = y.shape[0]
        prior_prob = dict()
        for i in labels:
            prob = np.sum(y == i) / n_sample
            if prob == 0:
                prior_prob[i] = 0.00001
            else:
                prior_prob[i] = prob
        self.param["prior_prob"] = prior_prob

        logging.info("Calculate class mu and sigma")
        mu = dict()
        sigma = dict()

        for i in labels:
            X_type = X[y == i]
            mu[i] = np.mean(X_type, axis=0)
            sigma[i] = np.cov(X_type, rowvar=False)
        self.param["class_mu"] = mu
        self.param["class_sigma"] = sigma

    @staticmethod
    def g(x, mu, sigma, prior_prob):
        return -0.5 * (x - mu).T @ np.linalg.pinv(sigma) @ (x - mu) - 0.5 * math.log(
            np.linalg.det(sigma) + 0.00001) + math.log(
            prior_prob)

    def predict(self, X, labels):
        n_test = X.shape[0]
        pred_prob = np.zeros((labels.shape[0], n_test))

        for idx in tqdm(range(n_test)):
            for i in range(labels.shape[0]):
                pred_prob[i][idx] = self.g(X[idx], self.param["class_mu"][labels[i]],
                                           self.param["class_sigma"][labels[i]],
                                           self.param["prior_prob"][labels[i]])
        preds = np.argmax(pred_prob, axis=0)
        preds_label = labels[preds]
        return preds, preds_label

    @staticmethod
    def accuracy(y, preds):
        total = y.shape[0]
        correct = np.sum(y == preds)
        return correct / total


def loading_data():
    data = datasets.load_digits()
    X = data.data
    y = data.target
    labels = np.unique(y)
    return X, y, labels


def sample_test():
    logging.info("load data and split into train and test sets")
    X, y, labels = loading_data()
    X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, test_size=0.3, random_state=0)

    classifier = BayesClassifier()
    logging.info("Training")
    classifier.train(X_train, y_train, labels)
    logging.info("Predict")
    preds, preds_labels = classifier.predict(X_test, labels)
    print(f"test_accuracy: {classifier.accuracy(y_test, preds) * 100} %")


if __name__ == "__main__":
    sample_test()
