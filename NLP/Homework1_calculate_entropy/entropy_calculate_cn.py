from collections import Counter
import re
import numpy as np
import matplotlib.pyplot as plt


# 读取文件内容
def read_text(i):
    with open(f"BOOK_CN/book{i}", mode="r") as f:
        text = f.read()
        f.close()
        return text


# 文本清洗
def text_clean(text):
    obj = re.compile("[^\u4e00-\u9fa5]")  # 去除除汉字以外的所有符号
    text = obj.sub("", text)
    return text


# 统计词频
def count_alpha(text, num):
    text = text[0:num-1]
    count = Counter(text)
    num = count.most_common()
    return num


# 计算熵的函数
def entropy_cal(alpha_num):
    # 读取汉字词列表
    alpha_num_list = np.array(alpha_num)
    # 读取各个汉字的词频
    num = np.array(alpha_num_list[:, 1], dtype="int64")
    alpha = np.array(alpha_num_list[:, 0])
    # 计算总词频
    total_num = np.sum(num)
    # 计算各个词的频率
    num_prob = np.divide(num, total_num)
    # 计算频率对数
    log2_num_prob = np.log2(num_prob)
    # 各个汉字的熵
    en_alpha = -np.multiply(num_prob, log2_num_prob)
    # 计算汉语语言熵
    en_total = -np.dot(num_prob, log2_num_prob)
    # 拼接字典，返回结果
    result = dict(np.c_[alpha, en_alpha])
    result['total'] = en_total
    return result


def main():
    text = read_text(12)  # 读取第 i 本书的内容
    text = text_clean(text)
    length = len(text)
    num = length // 5000
    # 绘制汉语语言熵的趋势图
    total_en = []
    y = []
    for i in range(num):
        if i == num - 1:
            alpha_num = count_alpha(text, length)
            total_en.append(round(float(entropy_cal(alpha_num)["total"]), 3))
            y.append(length)
        else:
            alpha_num = count_alpha(text, (i + 1) * 5000)
            total_en.append(round(float(entropy_cal(alpha_num)["total"]), 3))
            y.append((i + 1) * 5000)
    plt.plot(y, total_en, 'r-')
    plt.xlabel("Number of Words")
    plt.ylabel("Entropy")
    plt.title('Total Entropy Tendency')
    plt.savefig("Entropy_CN/total.jpg")
    plt.close()


if __name__ == "__main__":
    main()