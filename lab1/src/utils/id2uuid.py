# coding=utf-8

import json
import tqdm
import pickle


def build_mapdict(infile: str) -> dict:
    map_dict = {}
    with open(infile, "r", encoding="utf-8") as fin:
        for line in tqdm.tqdm(fin):
            json_data = json.loads(line.strip())
            map_dict[json_data["id"]] = json_data["uuid"]
    return map_dict


if __name__ == "__main__":
    infiles = ["lab1/data/2018_0" + str(idx) + ".json" for idx in range(1, 6)]
    map_dict = {}
    for infile in infiles:
        map_dict.update(build_mapdict(infile))
    pickle.dump(map_dict, open("lab1/output/id2uuid.pkl", "wb"))