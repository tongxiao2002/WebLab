import math
from re import S
import nltk
from nltk.stem.porter import PorterStemmer
import numpy
import pickle

N = 306242

#思路：

class TextInfo():
    def __init__(self, d:float , k:float ):
        self.dot = d        #
        self.norm = k


def get_searchwords(filename:str, indice:dir):
    searchtfidf ={}     #searchwords 对应 tfidf
    Porter_stemmer = nltk.stem.PorterStemmer()
    norm = 0 

    with open(filename, "r") as fin:
        for line in fin:
            word = Porter_stemmer.stem(line.strip())#标准化 词汇

            if word in indice.keys():
                idf = math.log(N/len(indice[word])+1)
                searchtfidf[word] = idf
                norm += idf*idf
            else:
                searchtfidf[word] = 0

    return searchtfidf, math.sqrt(norm)

def compute_vsm(searchwordsfile:str, indicefile:str):
    picklefile = open(indicefile, 'rb')
    inverse_indice = pickle.load(picklefile)

    searchtfidf, searchnorm = get_searchwords(searchwordsfile, inverse_indice)
    textvsm = []        #recode Textinfo 记录cosine 累加的结果·
    textcosine = []     #由vsm累加的结果得到cos值，最后得到最小的10个数对应的id和uuid
   # searchwords norm 
    closestid = []

    for word,q in searchtfidf.keys():
        if word in inverse_indice.keys():
            for id, d in inverse_indice[word]:
                a = textvsm[id].TextInfo.dot
                b = textvsm[id].TextInfo.norm
                textvsm[id] = TextInfo(a+ d*q, b+d*d)

    for info in textvsm:
        textnorm = math.sqrt(info.TextInfo.norm)
        dot = info.TextInfo.dot
        cosine = dot/(textnorm*searchnorm)
        textcosine.append(cosine)    #id是否从0开始

    origincosine = textcosine[:]
    textcosine.sort()

    for cosine in textcosine[0:9]:
        closestid.append(origincosine.index(cosine))

    return closestid


if __name__ == "__main__":
    searchwordsfile = "D://WorkPlace/data/searchwords.txt"
    indicefile =  "D://WorkPlace/data/output/invert_indices.dict"

    closestid = compute_vsm(searchwordsfile, indicefile)

    