import numpy as np
import os
from tqdm import tqdm
import math

File_PATH = "./data"
Train_PATH = File_PATH + "/training_features.txt"
Test_True_PATH = File_PATH + "/test_true.txt"

Corpus_PATH = "./Corpus/MSRA"
Corpus_Train_PATH = Corpus_PATH + "/msra_train.txt"
Corpus_Test_PATH = Corpus_PATH + "/msra_test.txt"

state_to_idx = {"O": 0, "PER": 1, "LOC": 2, "ORG": 3}
idx_to_state = ["O", "PER", "LOC", "ORG"]


def sentence_to_features(sentence, states):
    sentence_filled = "00" + sentence + "00"
    sentence_length = len(sentence_filled)
    feature = ""
    for i in range(2, sentence_length - 2):
        feature += idx_to_state[states[i - 2]] + " "
        feature += sentence_filled[i] + " " + sentence_filled[i - 2] + " " + sentence_filled[i - 1] + " " + \
                   sentence_filled[i + 1] + " " + sentence_filled[i + 2]
        if i == 2:
            feature += " 0"
        else:
            feature += f" {sentence_filled[i - 1]}/{sentence_filled[i]}"

        if i == sentence_length - 3:
            feature += " 0"
        else:
            feature += f" {sentence_filled[i]}/{sentence_filled[i + 1]}"

        if i <= 3:
            feature += " 0"
        else:
            feature += f" {sentence_filled[i - 2]}/{sentence_filled[i - 1]}/{sentence_filled[i]}"
        if i >= sentence_length - 4:
            feature += " 0"
        else:
            feature += f" {sentence_filled[i]}/{sentence_filled[i + 1]}/{sentence_filled[i + 2]}"
        if i <= 3 or i >= sentence_length - 4:
            feature += " 0"
        else:
            feature += f" {sentence_filled[i - 2]}/{sentence_filled[i - 1]}/{sentence_filled[i]}/{sentence_filled[i + 1]}/{sentence_filled[i + 2]} "
        feature += "\n"
    return feature


def Corpus_to_features(corpus_path, feature_path):
    corpus = open(corpus_path, "r", encoding="utf-8").read().split("\n")
    f = open(feature_path, "w", encoding="utf-8")
    for sentence in tqdm(corpus):
        # 转化为字典格式
        if sentence:
            sentence = eval(sentence)
            text = sentence["text"]
            entity_list = sentence["entity_list"]
            state_idx = np.zeros(len(text), dtype=int)
            for entity in entity_list:
                start = entity["entity_index"]["begin"]
                end = entity["entity_index"]["end"]
                kind = entity["entity_type"]
                state_idx[start:end - 1] += state_to_idx[kind]
            state = sentence_to_features(text, state_idx)
            f.write(state)
    f.close()


class MaxEnt:
    def __init__(self):
        self._samples = []  # 样本集，包含了样本与特征
        self._y = set([])  # 标签集合
        self._numXY = {}  # key是(xi,yi), value 是 (xi,yi) 的数量
        self._xyID = {}
        self._N = 0  # 样本数量
        self._n = 0  # 特征对（xi,yi）的数量
        self._C = 0
        self._ep = []
        self._ep_ = []
        self._w = []
        self._last_w = []
        self._EPS = 0.01

    def load_data(self, feature_path):
        features = open(feature_path, "r", encoding="utf-8").read().split("\n")
        for feature in tqdm(features):
            feature = feature.split(" ")
            y = feature[0]
            X = feature[1:]
            self._samples.append(feature)
            self._y.add(y)
            for x in set(X):
                self._numXY[(x, y)] = self._numXY.get((x, y), 0) + 1

    def init_param(self):
        self._N = len(self._samples)
        self._n = len(self._numXY)
        self._C = max([len(sample) - 1 for sample in self._samples])
        self._w = np.zeros(self._n)
        self._last_w = np.zeros(self._n)
        self._ep_ = np.zeros(self._n)
        for i, xy in tqdm(enumerate(self._numXY)):
            self._ep_[i] = self._numXY[xy] / self._N
            self._xyID[xy] = i

    def _Zx(self, X):
        ZX = 0
        for y in self._y:
            sum_y = 0
            for x in X:
                if (x, y) in self._numXY:
                    sum_y += self._w[self._xyID[(x, y)]]
            ZX += math.exp(sum_y)
        return ZX


if __name__ == "__main__":
    os.makedirs(File_PATH, exist_ok=True)
    Corpus_to_features(Corpus_Train_PATH, Train_PATH)
    maxent = MaxEnt()
    maxent.load_data(Train_PATH)
    maxent.init_param()
    # Corpus_to_datasets(Corpus_Test_PATH, Test_True_PATH)
