import jieba
from collections import Counter
import matplotlib.pyplot as plt
import os
import re


def rid_of_ad(content):
    ad = ['本书来自www.cr173.com免费txt小说下载站', '更多更新免费电子书请关注www.cr173.com', '新语丝电子文库']
    for ads in ad:
        content = content.replace(ads, '')
    return content


def preprocess(root_dir):
    corpus = []
    for file in os.listdir(root_dir):
        if file [-3:] == 'txt':
            path = os.path.join(root_dir, file)
            with open(path, 'r', encoding='ANSI') as f:
                text = [line.strip("\n").replace("\u3000", "").replace("\t", "") for line in f][3:]
                corpus += text
    pattern = r'[^\u4E00-\u9FA5]'
    regex = re.compile(pattern)
    replacements = ["\t", "\n", "\u3000", "\u0020", "\u00A0", " "]
    for j in range(len(corpus)):
        corpus[j] = rid_of_ad(corpus[j])  # 去除广告
        corpus[j] = re.sub(regex, "", corpus[j])  # 只保留中文
        for replacement in replacements:
            corpus[j] = corpus[j].replace(replacement, "") # 去除换行符、分页符等符号
    corpus = [x for x in corpus if x != ""] # 去除空字符串
    return corpus


def calculate_word_frequency(text):
    all_words = []
    # 使用jieba分词进行分词并统计词频
    for line in text:
        words = list(jieba.cut(line))
        all_words.extend(words)
    word_counts = Counter(all_words).most_common()
    freq = [count for _, count in word_counts]  # 获取词频列表
    ranks = list(range(1, len(word_counts) + 1))  # 获取排名列表
    return ranks, freq

def plot_zipfs_law(ranks, frequencies):
    # 绘制Log-log图
    plt.scatter(ranks, frequencies)
    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel('log_rank')
    plt.ylabel('log_frequency')
    plt.title("Zipf's Law")
    plt.show()


root = './dataset'
context = preprocess(root)
rank, fre = calculate_word_frequency(context)
print(rank, fre)
plot_zipfs_law(rank, fre)