import pickle
import numpy as np


def load_dict(filename: str):
    return pickle.load(open(filename, "rb"))


def semantic_search(invert_indices: dict, keywords: list):
    N = 306242
    tfidf = np.zeros(N, dtype=np.float32)
    for word in keywords:
        for id, value in invert_indices[word].items():
            tfidf[id] += value
    return np.argsort(tfidf)[-10:].tolist()


if __name__ == "__main__":
    keywords = "best worst delta"
    keywords = keywords.split()
    # normalization
    invert_indices = "lab1/output/invert_indices.dict"
    invert_indices = load_dict(invert_indices)
    doc_ids = semantic_search(invert_indices, keywords)
    print(doc_ids)