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
            yield json_data["uuid"], json_data["id"], json_data["title"] + " " + json_data["text"]


def build_one_file(filename: str, stopwords: list, offset_idx: int):
    invert_indices = {}
    info_indice = {}    #recode the len of text 
    uuid_indice = {}   #recode relationship of id and uuid 

    porter_stemmer = nltk.stem.PorterStemmer()

    for uuid, id, text in tqdm.tqdm(yield_text(filename), position=offset_idx):
        word_list = list(gensim.utils.tokenize(text, lowercase=True, deacc=True))   # tokenize
        del_idx = []

        for idx, word in enumerate(word_list):  # delete stopwords and normalize
            if word in stopwords:
                del_idx.append(idx)
            else:
                word_list[idx] = porter_stemmer.stem(word)
        for idx in reversed(del_idx):
            word_list.pop(idx)

        info_indice[id] = len(word_list)    #for compute tfidf 
        uuid_indice[id] = uuid
        for word in word_list:              # build indices
            if word not in invert_indices.keys():
                invert_indices[word] = {id:1}
            else:
                if id not in invert_indices[word].keys():
                    invert_indices[word][id] = 0
                invert_indices[word][id] += 1

    return invert_indices, info_indice, uuid_indice


def build(infiles: list, stopfile: str, outfile: str, id2uuidfile:str):
    N = 306242          # 新闻总条数
    stopwords = []
    with open(stopfile, "r") as fin:        ##get stopwords 
        for line in fin:
            stopwords.append(line.strip())

    invert_indices = {}
    info_indice = {}
    uuid_indice = {}
    pool = multiprocessing.Pool(processes=len(infiles))
    process_list = []
    for idx, filename in enumerate(infiles):
        process_list.append(pool.apply_async(func=build_one_file, args=(filename, stopwords, idx)))
    pool.close()
    pool.join()

    for p in process_list:
        indices, info, uuid = p.get()
        for k, v in indices.items():
            if k not in invert_indices.keys():
                invert_indices[k] = v
            else:
                for id, count in v.items():
                    invert_indices[k][id] = count
        info_indice.update(info)
        uuid_indice.update(uuid)

    for word, id_list in invert_indices.items():      # compute tf-idf
        for id, v in id_list.items():
            word_len = info_indice[id]

            invert_indices[word][id] = float(v / word_len) *\
                                       math.log(float(N / (len(id_list) + 1)))
    # save to outfile
    if not os.path.isdir(os.path.dirname(outfile)):
        os.makedirs(os.path.dirname(outfile))
    pickle.dump(invert_indices, open(outfile, "wb"))
    pickle.dump(uuid_indice, open(id2uuidfile, "wb"))


    
if __name__ == "__main__":
    files = ["lab1/data/" + "2018_0" + str(idx) + ".json" for idx in range(1, 6)]
    #files = ["lab1/data/2018.json"]
    stopfile = "lab1/data/stopwords.txt"
    outfile = "lab1/data/output/invert_indices.dict"
    id2uuidfile = "lab1/data/output/id2uuid.dict"
    build(files, stopfile, outfile, id2uuidfile)