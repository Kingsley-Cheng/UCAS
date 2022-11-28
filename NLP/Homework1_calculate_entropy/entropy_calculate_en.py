import re
from collections import Counter

import matplotlib.pyplot as plt
import numpy as np


# 读取文件内容
def read_text(i):
    with open(f"BOOK_EN/book{i}", mode="r") as f:
        text = f.read()
        f.close()
        return text


# 文本清洗
def text_clean(text):
    obj = re.compile("[^A-Z^a-z]")  # 去除除字母以外的所有符号
    text = obj.sub("", text)
    text = text.lower()  # 全部转化为小写
    return text


# 统计词频
def count_alpha(text, num):
    text = text[0:num - 1]
    count = Counter(text)
    num = count.most_common()
    return num


# 计算熵的函数
def entropy_cal(alpha_num):
    # 读取字母列表
    alpha_num_list = np.array(alpha_num)
    # 读取各个字母的词频
    num = np.array(alpha_num_list[:, 1], dtype="int64")
    alpha = np.array(alpha_num_list[:, 0])
    # 计算总词频
    total_num = np.sum(num)
    # 计算各个词的频率
    num_prob = np.divide(num, total_num)
    # 计算频率对数
    log2_num_prob = np.log2(num_prob)
    # 各个字母的熵
    en_alpha = -np.multiply(num_prob, log2_num_prob)
    # 计算英语语言熵
    en_total = -np.dot(num_prob, log2_num_prob)
    # 拼接字典，返回结果
    result = dict(np.c_[alpha, en_alpha])
    result['total'] = en_total
    return result


def main():
    text = read_text(1)
    text = text_clean(text)
    # 以5000个字母为统计单位，依次加入计算熵，观察趋势
    length = len(text)
    num = length // 5000
    alpha_num = count_alpha(text, length)
    entropy_cal(alpha_num)
    # 绘制各个英语字母的熵的趋势图
    for alpha in [chr(i) for i in range(97, 123)]:
        total = []
        y = []
        for i in range(num):
            if i == num - 1:
                alpha_num = count_alpha(text, length)
                total.append(round(float(entropy_cal(alpha_num)[f"{alpha}"]), 3))
                y.append(length)
            else:
                alpha_num = count_alpha(text, (i + 1) * 5000)
                total.append(round(float(entropy_cal(alpha_num)[f'{alpha}']), 3))
                y.append((i + 1) * 5000)
        plt.plot(y, total, 'r-')
        plt.xlabel("Number of Words")
        plt.ylabel("Entropy")
        plt.title(f'Word "{alpha}" Entropy Tendency')
        plt.savefig(f"Entropy_EN/{alpha}.jpg")

    # 绘制英语语言熵的趋势图
    total = []
    y = []
    for i in range(num):
        if i == num - 1:
            alpha_num = count_alpha(text, length)
            total.append(round(float(entropy_cal(alpha_num)["total"]), 3))
            y.append(length)
        else:
            alpha_num = count_alpha(text, (i + 1) * 5000)
            total.append(round(float(entropy_cal(alpha_num)["total"]), 3))
            y.append((i + 1) * 5000)
    plt.plot(y, total, 'r-')
    plt.xlabel("Number of Words")
    plt.ylabel("Entropy")
    plt.title('Total Entropy Tendency')
    plt.savefig("Entropy_EN/total.jpg")
    plt.close()


if __name__ == "__main__":
    main()
