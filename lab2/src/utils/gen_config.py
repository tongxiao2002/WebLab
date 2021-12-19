import yaml


if __name__ == "__main__":
    config = {}
    config["data"] = {}
    config["data"]["entity_file"] = "lab2/data/entity_with_text.txt"
    config["data"]["relation_file"] = "lab2/data/relation_with_text.txt"
    config["data"]["train_file"] = "lab2/data/train.txt"
    config["data"]["eval_file"] = "lab2/data/dev.txt"
    config["data"]["test_file"] = "lab2/data/test.txt"

    config["embedding"] = {}
    config["embedding"]["emb_dim"] = 200
    config["embedding"]["entity_embedding_path"] = "lab2/output/entity_emb.bin"
    config["embedding"]["relation_embedding_path"] = "lab2/output/relation_emb.bin"

    config["model"] = {}
    config["model"]["model_name"] = "TransH"
    config["model"]["model_save_dir"] = "lab2/output"

    config["parameter"] = {}
    config["parameter"]["lr"] = 1
    config["parameter"]["epochs"] = 10
    config["parameter"]["decay"] = 0
    config["parameter"]["batch_size"] = 256
    config["parameter"]["device"] = "cuda:0"
    config["parameter"]["seed"] = 114514

    config["log"] = {}
    config["log"]["log_dir"] = "lab2/logs"

    yaml.dump(config, open("lab2/src/config.yaml", "w", encoding="utf-8"))
