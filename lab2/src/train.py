import os
import yaml
import tqdm
import torch
import pickle
import torch.utils.data
from model import TransH, TransE, MarginLoss
from dataloader import FullTripletDataset
from utils.utils import *


MODELS = {
          "TransE": TransE,
          "TransH": TransH
         }


def train(config: dict, logger: logging.Logger):
    device = config["parameter"]["device"]
    if not os.path.isdir(config["model"]["model_save_dir"]):
        os.mkdir(config["model"]["model_save_dir"])
    if not os.path.isfile(config["embedding"]["entity_embedding_path"]) or not os.path.isfile(config["embedding"]["relation_embedding_path"]):
        build_word2vec(config["data"]["entity_file"],
                       config["data"]["relation_file"],
                       train_file=config["data"]["train_file"],
                       test_file=config["data"]["test_file"],
                       vector_size=config["embedding"]["emb_dim"])
    entity_pre_emb = pickle.load(open(config["embedding"]["entity_embedding_path"], "rb"))
    relation_pre_emb = pickle.load(open(config["embedding"]["relation_embedding_path"], "rb"))
    best_syn_score = 0.0
    if not os.path.isfile(os.path.join(config["model"]["model_save_dir"], config["model"]["model_name"] + ".bin")):
        model = MODELS[config["model"]["model_name"]](entity_pre_emb["word2idx"],
                                                      relation_pre_emb["word2idx"],
                                                      emb_dim=config["embedding"]["emb_dim"],
                                                      entity_pre_embedding=entity_pre_emb["embedding"],
                                                      relation_pre_embedding=relation_pre_emb["embedding"]).to(device)
    else:
        checkpoint = torch.load(os.path.join(config["model"]["model_save_dir"], config["model"]["model_name"] + ".bin"), map_location="cpu")
        entity_word2idx = checkpoint["entity_word2idx"]
        relation_word2idx = checkpoint["relation_word2idx"]
        model = MODELS[config["model"]["model_name"]](entity_word2idx,
                                                      relation_word2idx,
                                                      emb_dim=config["embedding"]["emb_dim"])
        model.load_state_dict(checkpoint["model_state_dict"])
        model = model.to(device)
        best_syn_score = checkpoint["best_syn_score"]
    loss_fn = MarginLoss().to(device)

    train_dataset = FullTripletDataset(config["data"]["train_file"])
    eval_dataset = FullTripletDataset(config["data"]["eval_file"])

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=config["parameter"]["batch_size"], shuffle=True)
    eval_loader = torch.utils.data.DataLoader(eval_dataset, batch_size=config["parameter"]["batch_size"], shuffle=False)

    optimizer = torch.optim.SGD(model.parameters(), lr=config["parameter"]["lr"], weight_decay=config["parameter"]["decay"])
    for epoch in range(config["parameter"]["epochs"]):
        tmp_step = 0
        total_loss = 0
        model.train()
        model.normalize_embedding()
        for batch in tqdm.tqdm(train_loader):
            h = batch["h"].to(device)
            r = batch["r"].to(device)
            t = batch["t"].to(device)
            if config["model"]["model_name"] == "TransE":
                score, neg_score = model((h, r, t))
                optimizer.zero_grad()
                loss = loss_fn(score, neg_score)
                loss.backward()
                total_loss += loss.item()
                optimizer.step()
            elif config["model"]["model_name"] == "TransH":
                loss = model((h, r, t))
                optimizer.zero_grad()
                loss.backward()
                total_loss += loss.item()
                optimizer.step()
            model.normalize_embedding()

            tmp_step += 1
            if tmp_step % 100 == 0:
                logger.info("Epoch: {}, Step: {}, Loss: {:.4f}".format(epoch + 1, tmp_step, total_loss / 100))
                total_loss = 0

        model.eval()
        hit_1_num = 0
        hit_5_num = 0
        hit_10_num = 0
        eval_num = 0
        for eval_batch in tqdm.tqdm(eval_loader):
            h = eval_batch["h"].to(device)
            r = eval_batch["r"].to(device)
            pred_results = model.predict((h, r))
            true_labels = eval_batch["t"].flatten().cpu().tolist()
            eval_num += len(true_labels)
            for idx in range(len(true_labels)):
                hit_1 = pred_results["hit@1"][idx]
                hit_5 = pred_results["hit@5"][idx]
                hit_10 = pred_results["hit@10"][idx]
                if true_labels[idx] in hit_1:
                    hit_1_num += 1
                if true_labels[idx] in hit_5:
                    hit_5_num += 1
                if true_labels[idx] in hit_10:
                    hit_10_num += 1
        hit_1_acc = hit_1_num / eval_num
        hit_5_acc = hit_5_num / eval_num
        hit_10_acc = hit_10_num / eval_num
        syn_score = 0.3 * hit_1_acc + 0.7 * hit_5_acc    # 加权平均, hit@5 权值最高
        if syn_score > best_syn_score:
            best_syn_score = syn_score
            torch.save({"entity_word2idx": entity_pre_emb["word2idx"],
                        "relation_word2idx": relation_pre_emb["word2idx"],
                        "model_state_dict": model.state_dict(),
                        "hit@1": hit_1_acc,
                        "hit@5": hit_5_acc,
                        "best_syn_score": best_syn_score}, os.path.join(config["model"]["model_save_dir"], config["model"]["model_name"] + ".bin"))
        logger.info("Epoch: {}, Hit@1: {:.4f}, Hit@5: {:.4f}, Hit@10: {:.4f}".format(epoch + 1, hit_1_acc, hit_5_acc, hit_10_acc))


if __name__ == "__main__":
    config_file = "lab2/src/config.yaml"
    config = yaml.load(open(config_file, "r", encoding="utf-8"), Loader=yaml.FullLoader)
    logger = get_logger(config)
    set_seed(config["parameter"]["seed"])
    train(config, logger)
    # build_word2vec(entity_file, relation_file)
