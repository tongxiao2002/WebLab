import math
from re import S
import nltk
import numpy
import pickle

N = 306242

#思路：

class TextInfo():
    def __init__(self, d:float , k:float ):
        self.dot = d
        self.norm = k


def get_searchwords(filename:str, indicefile:str):
    searchtfidf ={}
    with open(filename, "r") as fin:
        for line in fin:
            pass
    return searchtfidf

def compute_vsm(filename:str, indicefile:str):
    inverse_indice = {}
    searchtfidf = {}
    textvsm = []
    textcosine = []
    searchnorm = 1      # searchwords norm 

    for word,q in searchtfidf.keys():
        if word in inverse_indice.keys():
            for id, d in inverse_indice[word]:
                a = textvsm[id].TextInfo.dot
                b = textvsm[id].TextInfo.norm
                textvsm[id] = TextInfo(a+ d*q, b+d*d)

    for info in textvsm:
        textnorm = math.sqrt(info.TextInfo.norm)
        dot = info.TextInfo.dot
        textcosine.append(dot/(textnorm*searchnorm))




