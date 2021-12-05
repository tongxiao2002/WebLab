import os
import pickle
import logging
import numpy as np
from datetime import datetime
from gensim.models import Word2Vec
import torch


def get_logger(log_dir) -> logging.Logger:
    logger = logging.getLogger(name="TransE")
    logger.setLevel(logging.INFO)
    now = datetime.now()
    filehandler = logging.FileHandler(filename=os.path.join(log_dir, "TransE_{}.log".format(now.strftime("%Y-%m-%d_%H:%M:%S"))), encoding="utf-8", mode="w")
    filehandler.setLevel(logging.INFO)
    formatter = logging.Formatter(fmt="%(asctime)s - %(filename)s - %(levelname)s - %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
    filehandler.setFormatter(formatter)
    consolehandler = logging.StreamHandler()
    consolehandler.setLevel(logging.INFO)
    consolehandler.setFormatter(formatter)
    logger.addHandler(filehandler)
    logger.addHandler(consolehandler)   # log to file and print to console
    return logger


def build_word2vec(entity_file: str, relation_file: str, vector_size: int=200):
    entity_ids = []
    entity_sentences = []
    relation_ids = []
    relation_sentences = []
    with open(entity_file, "r", encoding="utf-8") as fin:
        for line in fin:
            id, sentence = line.strip().split("\t")
            entity_ids.append(int(id))
            entity_sentences.append(sentence.split())
    with open(relation_file, "r", encoding="utf-8") as fin:
        for line in fin:
            id, sentence = line.strip().split("\t")
            relation_ids.append(int(id))
            relation_sentences.append(sentence.split())
    sentence = entity_sentences
    sentence.extend(relation_sentences)
    model = Word2Vec(sentences=sentence, size=vector_size, min_count=1)
    # word2idx = {"_UNK": 0}
    word2idx = {}
    # vectors = [np.zeros(vector_size, dtype=np.float32)]
    vectors = []
    for idx, (id, sentence) in enumerate(zip(entity_ids, entity_sentences)):
        word2idx[id] = idx
        mean_vector = np.mean([model.wv[word] for word in sentence], axis=0)
        vectors.append(mean_vector)
    vectors = np.stack(vectors)
    entity_embedding = {"word2idx": word2idx, "embedding": vectors}
    pickle.dump(entity_embedding, open("lab2/output/entity_emb.bin", "wb"))

    word2idx = {"_UNK": 0}
    vectors = [np.zeros(vector_size, dtype=np.float32)]
    for idx, (id, sentence) in enumerate(zip(relation_ids, relation_sentences)):
        word2idx[id] = idx + 1
        mean_vector = np.mean([model.wv[word] for word in sentence], axis=0)
        vectors.append(mean_vector)
    vectors = np.stack(vectors)
    entity_embedding = {"word2idx": word2idx, "embedding": vectors}
    pickle.dump(entity_embedding, open("lab2/output/relation_emb.bin", "wb"))


def collate_fn(data_list):
    '''
    暂时不支持 TestTripletDataset, 因为少了尾实体
    '''
    batch = {"h": [], "r": [], "t": []}
    for data in data_list:
        batch["h"].append(data["h"])
        batch["r"].append(data["r"])
        batch["t"].append(data["t"])
    batch["h"] = torch.LongTensor(batch["h"])
    batch["r"] = torch.LongTensor(batch["r"])
    batch["t"] = torch.LongTensor(batch["t"])
    return batch
