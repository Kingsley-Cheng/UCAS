import numpy as np
import os

File_PATH = "./data"
Train_PATH = File_PATH + "/training.txt"
Test_True_PATH = File_PATH + "/test_true.txt"

Corpus_PATH = "./Corpus/MSRA"
Corpus_Train_PATH = Corpus_PATH + "/msra_train.txt"
Corpus_Test_PATH = Corpus_PATH + "/msra_test.txt"


def Corpus_to_datasets(corpus_path, file_path):
    index_to_state = np.array(["S", "PER", "LOC", "ORG"])
    state_to_index = {"S": 0, "PER": 1, "LOC": 2, "ORG": 3}
    corpus = open(corpus_path, "r", encoding="utf-8").read().split("\n")
    f = open(file_path, "w", encoding="utf-8")
    for sentence in corpus:
        # 转化为字典格式
        if sentence:
            sentence = eval(sentence)
            text = sentence["text"]
            entity_list = sentence["entity_list"]
            state_idx = np.zeros(len(text), dtype=int)
            for entity in entity_list:
                start = entity["entity_index"]["begin"]
                end = entity["entity_index"]["end"]
                kind = state_to_index[entity["entity_type"]]
                state_idx[start:end + 1] += kind
            state = "".join(str(index_to_state[state_idx])).replace("[", "").replace("]", "").replace("\n", "")
            f.write(state + "\n")
    f.close()


def sentence_to_feature(sentence, i):
    sentence_length = len(sentence)
    feature_length = 10
    """
    w: word
    w-1:pre_word
    w-2:pre_pre_word
    w+1:next_word
    w+2:next_next_word
    w-1:w:pre+w
    w:w+1:w+next
    """
    features = np.zeros(feature_length)
    features[0] = sentence[i]
    if i == 0:
        features[1] = "S"
    else:
        features[1] = sentence[i]
    if i == 1 or i == 0:
        features[1] = 1
        pass


def data_to_features(file_path, feature_path):
    data = open(file_path, "r", encoding="utf-8").read().split("\n")
    pass


class MaxEnt:
    def __init__(self):
        self._samples = []  # 样本集，包含了样本与特征
        self._y = set([])  # 标签集合
        self._numXY = defaultdict(int) # key是(xi,yi), value 是 (xi,yi) 的数量
        self._xyID = {}
        self._N = 0 # 样本数量
        self._n = 0 # 特征对（xi,yi）的数量
        self._ep = [] #
        self._ep_ = []
        self._w = []
        self._last_w = []
        self._EPS = 0.01

if __name__ == "__main__":
    os.makedirs(File_PATH, exist_ok=True)
    Corpus_to_datasets(Corpus_Train_PATH, Train_PATH)
    Corpus_to_datasets(Corpus_Test_PATH, Test_True_PATH)
