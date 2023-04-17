import joblib

import matplotlib.pyplot as plt
import numpy as np

import tools
import re
import pandas as pd
import torch
import model

from sklearn.metrics import adjusted_rand_score

FILE_PATH = "./data/"
MODEL_PATH = './model/'


def read_agnews(file_path):
    index_name = ['label', 'title', 'content']
    agnews_train = pd.read_csv(file_path + "train.csv", names=index_name)
    agnews_test = pd.read_csv(file_path + "test.csv", names=index_name)
    all_news = pd.concat([agnews_train, agnews_test])
    data = [[], [], []]
    for label, title, content in all_news.values:
        data[0].append(label)
        data[1].append(re.sub('[^A-Za-z]+', ' ', title).strip().lower())
        data[2].append(re.sub('[^A-Za-z]+', ' ', content).strip().lower())
    return data[0], data[1], data[2]


def load_data_agnews(sentence, batch_size, max_window_size, num_noise_words):
    vocab = tools.Vocab(sentence)
    subsampled, counter = tools.subsample(sentence, vocab)
    corpus = [vocab[line] for line in subsampled]
    all_centers, all_contexts = tools.get_centers_and_contexts(
        corpus, max_window_size)
    all_negatives = tools.get_negatives(
        all_contexts, vocab, counter, num_noise_words)

    class AGNewsDataset(torch.utils.data.Dataset):
        def __init__(self, centers, contexts, negatives):
            assert len(centers) == len(contexts) == len(negatives)
            self.centers = centers
            self.contexts = contexts
            self.negatives = negatives

        def __getitem__(self, index):
            return (self.centers[index], self.contexts[index],
                    self.negatives[index])

        def __len__(self):
            return len(self.centers)

    dataset = AGNewsDataset(all_centers, all_contexts, all_negatives)

    data_iter = torch.utils.data.DataLoader(
        dataset, batch_size, shuffle=True,
        collate_fn=tools.batchify)
    return data_iter, vocab


def get_similar_tokens(query_token, vocab, k, embed):
    W = embed.weight.data
    x = W[vocab[query_token]]
    # 计算余弦相似性。增加1e-9以获得数值稳定性
    cos = torch.mv(W, x) / torch.sqrt(torch.sum(W * W, dim=1) *
                                      torch.sum(x * x) + 1e-9)
    topk = torch.topk(cos, k=k + 1)[1].cpu().numpy().astype('int32')
    for i in topk[1:]:  # 删除输入词
        print(f'cosine sim={float(cos[i]):.3f}: {vocab.to_tokens(i)}')


def download_20news(path=FILE_PATH, subset='train', shuffle=False, remove=('headers', 'footers', 'quotes')):
    import ssl
    from sklearn.datasets import fetch_20newsgroups
    ssl._create_default_https_context = ssl._create_unverified_context
    fetch_20newsgroups(data_home=path, subset=subset, shuffle=shuffle, remove=remove, download_if_missing=True)


def load_20news(category, path=FILE_PATH):
    with open(path + "20news-bydate_py3.pkz", 'rb') as f:
        news20 = joblib.load(f)
        data = [[], []]
        print(news20['train']['target_names'])
        for X, y in zip(news20['train']['data'], news20['train']['target']):
            if y in category:
                data[0].append(re.sub('[^A-Za-z]+', ' ', X).strip().lower())
                data[1].append(y)
        for X, y in zip(news20['test']['data'], news20['test']['target']):
            if y in category:
                data[0].append(re.sub('[^A-Za-z]+', ' ', X).strip().lower())
                data[1].append(y)
        f.close()
    return data[0], data[1]


if __name__ == "__main__":
    text, label = load_20news(category=[10, 11, 12, 13])
    # feature = model.tfidf(text)
    # np.save("tf_idf.npy",feature)
    # feature = np.load("tf_idf.npy")

    # text = tools.tokenize(text)
    # data_iter, vocab = load_data_agnews(text, 512, 5, 5)
    #
    # embbed_size = 100
    # lr, num_epochs = 0.001, 5
    # net = model.word2vec_net(vocab, embbed_size)
    # epoch_loss = model.train(net, data_iter, vocab, lr, num_epochs,init='none')
    # state = {'net': net.state_dict(), 'epoch': num_epochs}
    # torch.save(state, MODEL_PATH + "wor2vec")
    # net.load_state_dict(torch.load(MODEL_PATH + "wor2vec")['net'])
    # W = net[0].weight.data
    # features = []
    # for i in range(len(text)):
    #     x = W[vocab[text[i]]]
    #     features.append(torch.mean(x,dim=0).numpy())
    # features = np.array(features)
    # np.save("word2vec.npy", features)
    # feature = np.load("word2vec_glove.npy")
    # ypred = tools.kmeans(feature, 5)
    # print(adjusted_rand_score(label, ypred))
    # pca = PCA(3)
    # new_X = pca.fit_transform(feature)
    # kmeans = KMeans(4,random_state=0,n_init='auto').fit(new_X)
    # # dbscan = DBSCAN(eps=0.3,min_samples=5).fit(feature)
    # y_labels = kmeans.labels_
    # # y_labels = dbscan.labels_
    # result = homogeneity_completeness_v_measure(label,y_labels)
    # print(result)
    # # print(label)
    # # right = sum(y_labels==label)
    # # print(right/len(label))
    # plt.scatter(new_X[:,0],new_X[:,1],c=y_labels)
    # plt.show()
    # content = content[:10000]
    # label = content[:10000]
    # transformer = TfidfVectorizer()
    # tfidf = transformer.fit_transform(content)
    # tfidf = tfidf.todense()
    # feature = np.array(tfidf)
    # kmeans = KMeans(4, random_state=0, n_init='auto').fit(feature)
    # # dbscan = DBSCAN(eps=0.3,min_samples=5).fit(feature)
    # y_labels = kmeans.labels_
    # # y_labels = dbscan.labels_
    # result = homogeneity_completeness_v_measure(label,y_labels)
    # print(result)
