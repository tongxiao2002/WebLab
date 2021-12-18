import os
import yaml
import tqdm
import torch
import torch.utils.data
from model import TransH, TransE
from dataloader import TestTripletDataset
from utils.utils import *


MODELS = {
          "TransE": TransE,
          "TransH": TransH
         }


def predict(config: dict, testfile: str, outputfile: str):
    device = config["parameter"]["device"]
    entity_pre_emb = pickle.load(open(config["embedding"]["entity_embedding_path"], "rb"))
    relation_pre_emb = pickle.load(open(config["embedding"]["relation_embedding_path"], "rb"))

    model = MODELS[config["model"]["model_name"]](entity_pre_emb["word2idx"],
                                                  relation_pre_emb["word2idx"],
                                                  emb_dim=config["embedding"]["emb_dim"])

    checkpoint = torch.load(os.path.join(config["model"]["model_save_dir"], config["model"]["model_name"] + ".bin"), map_location="cpu")
    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(device)

    test_dataset = TestTripletDataset(testfile)

    dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=config["parameter"]["batch_size"], shuffle=False)

    with open(outputfile, "w", encoding="utf-8") as fout:
        for test_batch in tqdm.tqdm(dataloader):
            h = test_batch["h"].to(device)
            r = test_batch["r"].to(device)
            pred_results = model.predict((h, r))
            for result in pred_results["hit@5"]:
                str_result = ''
                for item in result:
                    str_result += str(item) + ","
                str_result = str_result[:-1]
                fout.write(str_result + '\n')


if __name__ == "__main__":
    config = yaml.load(open("lab2/src/config.yaml", "r", encoding="utf-8"), Loader=yaml.FullLoader)
    predict(config, config["data"]["test_file"], "lab2/submit/result.txt")
