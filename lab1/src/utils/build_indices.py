import os
import math
import json
import nltk
import tqdm
import pickle
import gensim
import multiprocessing


def yield_text(filename: str):
    with open(filename, "r") as fin:
        for line in fin:
            json_data = json.loads(line)
            yield json_data["id"], json_data["title"] + " " + json_data["text"]


def build_one_file(filename: str, stopwords: list, offset_idx: int):
    invert_indices = {}
    porter_stemmer = nltk.stem.PorterStemmer()
    for id, text in tqdm.tqdm(yield_text(filename), position=offset_idx):
        word_list = list(gensim.utils.tokenize(text, lowercase=True, deacc=True))   # tokenize
        del_idx = []
        for idx, word in enumerate(word_list):  # delete stopwords and normalize
            if word in stopwords:
                del_idx.append(idx)
            else:
                word_list[idx] = porter_stemmer.stem(word)
        for idx in reversed(del_idx):
            word_list.pop(idx)

        for word in word_list:  # build indices
            if word not in invert_indices.keys():
                # invert_indices[word] = {id: {"c": 1, "t": len(text)}}     # tf-idf 值先存下 text 长度
                invert_indices[word] = {id: [1, len(text)]}
            else:
                # if id not in invert_indices[word].keys():
                    # invert_indices[word][id] = {"c": 0, "t": len(text)}
                # invert_indices[word][id]["c"] += 1
                if id not in invert_indices[word].keys():
                    invert_indices[word][id] = [0, len(text)]
                invert_indices[word][id][0] += 1
    # q.put(invert_indices)
    return invert_indices


def build(infiles: list, stopfile: str, outfile: str):
    N = 306242      # 新闻总条数
    stopwords = []
    with open(stopfile, "r") as fin:
        for line in fin:
            stopwords.append(line.strip())

    invert_indices = {}
    pool = multiprocessing.Pool(processes=len(infiles))
    process_list = []
    for idx, filename in enumerate(infiles):
        process_list.append(pool.apply_async(func=build_one_file, args=(filename, stopwords, idx)))
    pool.close()
    pool.join()

    for p in process_list:
        indices = p.get()
        for k, v in indices.items():
            if k not in invert_indices.keys():
                invert_indices[k] = v
            else:
                for id, count in v.items():
                    invert_indices[k][id] = count

    for word, id_list in invert_indices.items():      # compute tf-idf
        for id, v in id_list.items():
            invert_indices[word][id] = float(v[0] / v[1]) *\
                                       math.log(float(N / (len(id_list) + 1)))

    # save to outfile
    if not os.path.isdir(os.path.dirname(outfile)):
        os.makedirs(os.path.dirname(outfile))
    pickle.dump(invert_indices, open(outfile, "wb"))

    
if __name__ == "__main__":
    files = ["data/" + "2018_0" + str(idx) + ".json" for idx in range(1, 6)]
    stopfile = "data/stopwords.txt"
    outfile = "output/invert_indices.dict"
    build(files, stopfile, outfile)