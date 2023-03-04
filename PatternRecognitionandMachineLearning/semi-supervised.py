# imports
import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm
from sklearn import datasets


def train_test_split(data, labels, rate=0.8):
    n_samples = data.shape[0]
    idx = int(n_samples * rate)
    label_idx = int(idx * 0.1)
    train_labeldata = data[:label_idx, :]
    train_unlabeldata = data[label_idx:idx, :]
    train_unlabels = labels[label_idx:idx]
    test_data = data[idx:, :]
    train_labels = labels[:label_idx]
    test_labels = labels[idx:]
    return (train_labeldata, train_labels), (train_unlabeldata, train_unlabels), (test_data, test_labels)


X, y = datasets.make_moons(300, random_state=0)
X = np.array(X)
y = np.array(y)
train_labeled, train_unlabeled, test = train_test_split(X, y)
X_l, y_l = train_labeled
X_u, y_u = train_unlabeled

svm = svm.SVC(C=1, kernel="linear")
svm.fit(X_l, y_l)
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(121)
ax.scatter(X_l[:, 0], X_l[:, 1], c=y_l)

Cu = 0.0001
Cl = 1
n_l = y_l.shape[0]
n_u = y_u.shape[0]
n_samples = n_l + n_u

weight = np.ones(n_samples)
weight[n_u:] = Cu
X_train = np.append(X_l, X_u, axis=0)
y_pred = svm.predict(X_u)
while Cu < Cl:
    y_train = np.concatenate(y_l, y_pred)
    svm.fit(X_train, y_train, sample_weight=weight)
    while True:
        y_pred = svm.predict(X_u)
        u_dist = svm.decision_function(X_u)
        coef = np.linalg.norm(svm.coef_)
        xi = 1 - u_dist * coef * y_pred
        positive_set, positive_idx = xi[y_pred>0], np.where(y_pred>0)
        negative_set, negative_idx = xi[y_pred<0], np.where(y_pred<0)
        positive_max_idx,negative_max_idx =
                if y_pred[i] * y_pred[j] < 0 < coef[i] and coef[j] > 0 and coef[i] + coef[j] > 2:
                    y_pred[i] = -1 * y_pred[i]
                    y_pred[j] = -1 * y_pred[j]
                    y_train = np.concatenate(y_l, y_pred)
                    svm.fit(X_train, y_train, sample_weight=weight)
                else:
                    break

    Cu = min(2 * Cu, Cl)

y_train = np.append(y_l, y_pred)
ax = fig.add_subplot(122)
ax.scatter(X_train[:, 0], X_train[:, 1], c=y_train)
plt.show()

X_t, y_t = test
print(np.sum(y_t == svm.predict(X_t)) / y_t.shape[0])

y_train = np.append(y_l, y_u)
svm.fit(X_train, y_train)
print(np.sum(y_t == svm.predict(X_t)) / y_t.shape[0])
