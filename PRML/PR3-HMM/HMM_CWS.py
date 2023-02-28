"""
author: Kingsley-Cheng
Date: 2022-12-10
tag: HMM+CWS
"""

# imports
import numpy as np
from tqdm import tqdm
import logging

logging.basicConfig(format='%(asctime)s - %(levelname)s: %(message)s', level=logging.DEBUG)

dataset = "mrs"
PATH = f"./Corpus/{dataset}"
Corpus_Training_PATH = PATH + f"/{dataset}_training.utf8"
Corpus_Test_PATH = PATH + f"/{dataset}_test.utf8"
Corpus_Gold_PATH = PATH + f"/{dataset}_test_gold.utf8"
State_Training_PATH = PATH + f"/{dataset}_training_state.utf8"


def Sentence_to_State(sentence):
    """
    Convert sentence to state
    "S": single word
    "B": begin
    "M": middle
    "E": end
    :param sentence
    :return: state
    """
    state = ""
    for word in sentence.split(" "):
        if word:
            word_len = len(word)
            if word_len == 1:
                state = state + "S" + " "
            else:
                state = state + "B" + "M" * (word_len - 2) + "E" + " "
    return state


def Corpus_to_State(corpus_path, State_path):
    """
    Convert whole Corpus to State file
    :param corpus_path
    :param State_path
    :return: State_file
    """
    corpus = open(corpus_path, "r", encoding="utf-8").read().split("\n")
    corpus_len = len(corpus)
    f = open(State_path, "w", encoding="utf-8")
    for idx, sentence in tqdm(enumerate(corpus)):
        if sentence:
            state = Sentence_to_State(sentence)
            if idx != corpus_len - 1:
                f.write(state + "\n")
    f.close()


class HMM:
    def __init__(self):
        self._state_to_idx = {"S": 0, "B": 1, "M": 2, "E": 3}
        self._idx_to_state = ["S", "B", "M", "E"]
        state_num = len(self._idx_to_state)
        self._init_matrix = np.zeros(state_num)
        self._transit_matrix = np.zeros((state_num, state_num))
        self._emit_matrix = {"S": {"total": 0}, "B": {"total": 0}, "M": {"total": 0}, "E": {"total": 0}}

    def count_init_matrix(self, states):
        if states:
            idx = self._state_to_idx[states[0]]
            self._init_matrix[idx] += 1

    def count_transit_matrix(self, states):
        for state1, state2 in zip(states[:-1], states[1:]):
            if state1 and state2:
                idx1 = self._state_to_idx[state1]
                idx2 = self._state_to_idx[state2]
                self._transit_matrix[idx1][idx2] += 1

    def count_emit_matrix(self, states, sentence):
        for state, word in zip(states, sentence):
            if word:
                self._emit_matrix[state][word] = self._emit_matrix[state].get(word, 0) + 1
                self._emit_matrix[state]["total"] += 1

    def normalize(self):
        self._init_matrix = self._init_matrix / np.sum(self._init_matrix)
        self._transit_matrix = self._transit_matrix / np.sum(self._transit_matrix, axis=1, keepdims=True)
        for state in self._emit_matrix.keys():
            for word in self._emit_matrix[state].keys():
                if word != "total":
                    self._emit_matrix[state][word] = self._emit_matrix[state][word] / self._emit_matrix[state][
                        "total"] * 100

    def train(self, corpus_path, state_path):
        corpus = open(corpus_path, "r", encoding="utf-8").read().split("\n")
        states = open(state_path, "r", encoding="utf-8").read().split("\n")
        for sentence, state in zip(corpus, states):
            if sentence:
                sentence = sentence.replace(" ", "")
                state = state.replace(" ", "")
                self.count_init_matrix(state)
                self.count_transit_matrix(state)
                self.count_emit_matrix(state, sentence)
        self.normalize()

    def Viterbi(self, sentence):
        V = [{}]
        path = {}
        route = []
        for state in self._state_to_idx:
            V[0][state] = self._init_matrix[self._state_to_idx[state]] + self._emit_matrix[state].get(sentence[0], 1e-6)
            path[state] = state
        route.append(path)
        for i in range(1, len(sentence)):
            V.append({})
            path = {}
            for state1 in self._state_to_idx:
                prob = []
                for state0 in self._state_to_idx:
                    idx1 = self._state_to_idx[state0]
                    idx2 = self._state_to_idx[state1]
                    prob.append(V[i - 1][state0] + self._transit_matrix[idx1][idx2])
                idx = np.argmax(prob)
                state = self._idx_to_state[idx]
                prob = np.max(prob)
                V[i][state1] = prob + self._emit_matrix[state1].get(sentence[i], 1e-6)
                path[state1] = state
            route.append(path)
        (prob, state) = max([(V[len(sentence) - 1][y], y) for y in self._idx_to_state])
        best_path = [state]
        for i in reversed(route):
            state = i[state]
            best_path.append(state)
        best_path.pop()
        best_path.reverse()
        return best_path

    def sentence_to_cut(self, sentence):
        path = self.Viterbi(sentence)
        result = ""
        for i in range(len(sentence) - 1):
            if path[i] == "S" or path[i] == "E":
                result = result + sentence[i] + " "
            else:
                result += sentence[i]
        result += sentence[len(sentence) - 1]
        return result

    def test(self, test_path, test_gold_path):
        test_corpus = open(test_path, "r", encoding="utf-8").read().split("\n")
        test_corpus_gold = open(test_gold_path, "r", encoding="utf-8").read().split("\n")
        total_test = 0
        total_test_gold = 0
        correct = 0
        for sentence1, sentence2 in tqdm(zip(test_corpus, test_corpus_gold)):
            if sentence1:
                sentence2 = sentence2.split("  ")
                sentence1 = self.sentence_to_cut(sentence1)
                sentence1 = sentence1.split(" ")
                total_test_gold += len(sentence2)
                total_test += len(sentence1)
                correct += len([i for i in sentence1 if i in sentence2])
        precision = correct / total_test
        recall = correct / total_test_gold
        F_score = (2 * precision * recall) / (precision + recall)
        print("Precision: ", precision)
        print("Recall: ", recall)
        print("FScore: ", F_score)


if __name__ == "__main__":
    logging.info("Convert Corpus to State")
    Corpus_to_State(Corpus_Training_PATH, State_Training_PATH)
    logging.info("Construct HMM Model")
    hmm = HMM()
    logging.info("Start to Train")
    hmm.train(Corpus_Training_PATH, State_Training_PATH)
    logging.info("Test")
    hmm.test(Corpus_Test_PATH, Corpus_Gold_PATH)
    print(hmm.sentence_to_cut("今天天气真好！"))
