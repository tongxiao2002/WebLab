import math
import nltk
import pickle
from nltk.util import invert_dict
import numpy as np
from nltk.stem.porter import PorterStemmer


def load_dict(filename: str):
    return pickle.load(open(filename, "rb"))


def semantic_search(invert_indices: dict, keywords: list):
    N = 306242
    tfidf = np.zeros(N, dtype=np.float32)
    for word in keywords:
        for id, value in invert_indices[word].items():
            tfidf[id] += value
    return np.argsort(tfidf)[-10:].tolist()

N = 306242

# 思路：

class TextInfo():
    def __init__(self, d: float, k: float):
        self.dot = d        #
        self.norm = k


def get_searchwords(filename: str, indice: dir):
    searchtfidf = {}  # searchwords 对应 tfidf
    Porter_stemmer = nltk.stem.PorterStemmer()
    norm = 0

    with open(filename, "r") as fin:
        for line in fin:
            word = Porter_stemmer.stem(line.strip())  # 标准化 词汇

            if word in indice.keys():
                idf = math.log(N / len(indice[word]) + 1)
                searchtfidf[word] = idf
                norm += idf * idf
            else:
                searchtfidf[word] = 0

    return searchtfidf, math.sqrt(norm)


def compute_vsm(searchwordsfile: str, indicefile: str):
    picklefile = open(indicefile, 'rb')
    inverse_indice = pickle.load(picklefile)   

    searchtfidf, searchnorm = get_searchwords(searchwordsfile, inverse_indice)
    textvsm = [TextInfo(0, 0) for _ in range(N)]  # record Textinfo 记录cosine 累加的结果·
    textcosine = []  # 由vsm累加的结果得到cos值，最后得到最小的10个数对应的id和uuid
   # searchwords norm
    closestid = []

    for word, q in searchtfidf.items():
        if word in inverse_indice.keys():
            for id, d in inverse_indice[word].items():
                a = textvsm[id].dot
                b = textvsm[id].norm
                textvsm[id] = TextInfo(a + d * q, b + d * d)

    for info in textvsm:
        textnorm = math.sqrt(info.norm)
        dot = info.dot
        try:
            cosine = dot / (textnorm * searchnorm)
        except ZeroDivisionError:
            cosine = 0
        textcosine.append(cosine)  # id是否从0开始

    origincosine = textcosine[:]
    textcosine.sort(reverse=True)

    print(textcosine[:1000])

    for cosine in textcosine[:10]:
        closestid.append(origincosine.index(cosine))

    return closestid


if __name__ == "__main__":
    # invert_indices = "lab1/output/invert_indices.dict"
    # invert_indices = load_dict(invert_indices)
    # doc_ids = semantic_search(invert_indices, keywords)
    # print(doc_ids)
    searchwordsfile = "lab1/data/searchwords.txt"
    indicefile = "lab1/data/output/invert_indices.dict"

    closestid = compute_vsm(searchwordsfile, indicefile)
    print(closestid)

