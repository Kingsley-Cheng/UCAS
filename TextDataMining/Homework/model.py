import math

import numpy as np
import torch
from torch import nn
import tools
from sklearn.feature_extraction.text import TfidfVectorizer

def skip_gram(center, contexts_and_negatives, embed_v, embed_u):
    v = embed_v(center)
    u = embed_u(contexts_and_negatives)
    pred = torch.bmm(v, u.permute(0, 2, 1))
    return pred


class SigmoidBCELoss(nn.Module):
    # 带掩码的二元交叉熵损失
    def __init__(self):
        super().__init__()

    def forward(self, inputs, target, mask=None):
        out = nn.functional.binary_cross_entropy_with_logits(
            inputs, target, weight=mask, reduction="none")
        return out.mean(dim=1)


def word2vec_net(vocab, embed_size):
    net = nn.Sequential(nn.Embedding(num_embeddings=len(vocab),
                                     embedding_dim=embed_size),
                        nn.Embedding(num_embeddings=len(vocab),
                                     embedding_dim=embed_size))
    return net


def train(net, data_iter, vocab, lr, num_epochs, device="cpu", init="glove"):
    loss = SigmoidBCELoss()

    def init_weights(m):
        if type(m) == nn.Embedding and init != 'glove':
            nn.init.xavier_uniform_(m.weight)
        elif type(m) == nn.Embedding and init == 'glove':
            glove100 = tools.TokenEmbedding("./data/glove.6B.100d")
            for i in range(1,len(m.weight)):
                char = str(vocab.to_tokens(i))
                m.weight.data[i] = torch.tensor(glove100[[char]][0])

    epoch_loss = [[], []]
    net.apply(init_weights)
    net = net.to(device)
    optimizer = torch.optim.Adam(net.parameters(), lr=lr)
    # 规范化的损失之和，规范化的损失数
    metric = tools.Accumulator(2)
    for epoch in range(num_epochs):
        num_batches = len(data_iter)
        for i, batch in enumerate(data_iter):
            optimizer.zero_grad()
            center, context_negative, mask, label = [
                data.to(device) for data in batch]

            pred = skip_gram(center, context_negative, net[0], net[1])
            l = (loss(pred.reshape(label.shape).float(), label.float(), mask)
                 / mask.sum(axis=1) * mask.shape[1])
            l.sum().backward()
            optimizer.step()
            metric.add(l.sum(), l.numel())
            if (i + 1) % (num_batches // 5) == 0 or i == num_batches - 1:
                epoch_loss[0].append(epoch + (i + 1) / num_batches)
                epoch_loss[1].append(metric[0] / metric[1])
                print(metric[0] / metric[1])
    print(f'loss {metric[0] / metric[1]:.3f}')
    return epoch_loss


def tfidf(text):
    vectorizer = TfidfVectorizer()
    train_v = vectorizer.fit_transform(text)
    return np.array(train_v.todense())