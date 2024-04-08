import jieba
import re
import numpy as np
import os


def replace_digits_with_chinese(number):
    digits_mapping = {1: '一', 2: '二', 3: '三'}
    str_number = str(number)
    result = ''
    for char in str_number:
        if char.isdigit():
            digit = int(char)
            result += digits_mapping[digit]
        else:
            result += char
    return result


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
            corpus[j] = corpus[j].replace(replacement, "")
    corpus = [x for x in corpus if x != ""] # 去除空字符串
    return corpus


def cha_fre(file,n):
    adict = {}
    if n == 1:
        for line in file:
            for i in line:
                if i in adict:
                    adict[i] += 1
                else:adict[i] = 1
    elif n == 2:
        for line in file:
            for i in range(len(line)-1):
                if (line[i]+line[i+1]) in adict:
                    adict[line[i]+line[i+1]] += 1
                else:adict[line[i]+line[i+1]] = 1
    else:
        for line in file:
            for i in range(len(line)-2):
                if (line[i]+line[i+1]+line[i+2]) in adict:
                    adict[line[i]+line[i+1]+line[i+2]] += 1
                else:adict[line[i]+line[i+1]+line[i+2]] = 1
    return adict


def word_fre(file,n):
    adict = {}
    if n == 1:
        for line in file:
            words = list(jieba.cut(line))
            for i in range(len(words)):
                if tuple(words[i:i+1]) in adict:
                    adict[tuple(words[i:i+1])] += 1
                else:adict[tuple(words[i:i+1])] = 1
    elif n == 2:
        for line in file:
            words = list(jieba.cut(line))
            for i in range(len(words)-1):
                if tuple(words[i:i+2]) in adict:
                    adict[tuple(words[i:i+2])] += 1
                else:adict[tuple(words[i:i+2])] = 1
    else:
        for line in file:
            words = list(jieba.cut(line))
            for i in range(len(words)-2):
                if tuple(words[i:i+3]) in adict:
                    adict[tuple(words[i:i+3])] += 1
                else:adict[tuple(words[i:i+3])] = 1
    return adict


def cal_cha_entropy(file,n):
    if n == 1:
        frequency = cha_fre(file,1)
        sums = np.sum(list(frequency.values()))
        entropy = -np.sum([i*np.log2(i/sums) for i in  frequency.values()])/sums
    elif n == 2:
        frequency1 = cha_fre(file,1)
        frequency2 = cha_fre(file,2)
        sums = np.sum(list(frequency2.values()))
        entropy = -np.sum([v*np.log2(v/frequency1[k[:n-1]]) for k,v in  frequency2.items()])/sums
    else:
        frequency2 = cha_fre(file,2)
        frequency3 = cha_fre(file,3)
        sums = np.sum(list(frequency3.values()))
        entropy = -np.sum([v*np.log2(v/frequency2[k[:n-1]]) for k,v in  frequency3.items()])/sums
    return entropy


def cal_word_entropy(file,n):
    if n == 1:
        frequency = word_fre(file,1)
        sums = np.sum(list(frequency.values()))
        entropy = -np.sum([i*np.log2(i/sums) for i in  frequency.values()])/sums
    elif n == 2:
        frequency1 = word_fre(file,1)
        frequency2 = word_fre(file,2)
        sums = np.sum(list(frequency2.values()))
        entropy = -np.sum([v*np.log2(v/frequency1[k[:n-1]]) for k,v in  frequency2.items()])/sums
    else:
        frequency2 = word_fre(file,2)
        frequency3 = word_fre(file,3)
        sums = np.sum(list(frequency3.values()))
        entropy = -np.sum([v*np.log2(v/frequency2[k[:n-1]]) for k,v in  frequency3.items()])/sums
    return entropy


root = './dataset'
context = preprocess(root)
for t in range(1, 4):
    cha = cal_cha_entropy(context, t)
    word = cal_word_entropy(context, t)
    print("基于字的" + replace_digits_with_chinese(t) + "元模型的平均信息熵为:", cha)
    print("基于词的" + replace_digits_with_chinese(t) + "元模型的平均信息熵为:", word)