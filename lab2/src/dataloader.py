import torch
import torch.utils.data


class FullTripletDataset(torch.utils.data.Dataset):
    def __init__(self, datafile: str) -> None:
        super(FullTripletDataset, self).__init__()
        self.triplet = {"h": [], "r": [], "t": []}
        self.triplet_num = 0
        with open(datafile, "r", encoding="utf-8") as fin:
            for line in fin:
                h, r, t = line.strip().split("\t")
                self.triplet["h"].append(int(h))
                self.triplet["r"].append(int(r))
                self.triplet["t"].append(int(t))
                self.triplet_num += 1
        self.triplet["h"] = torch.LongTensor(self.triplet["h"])
        self.triplet["r"] = torch.LongTensor(self.triplet["r"])
        self.triplet["t"] = torch.LongTensor(self.triplet["t"])

    def __getitem__(self, idx):
        triplet = {key: torch.LongTensor([val[idx]]) for key, val in self.triplet.items()}
        return triplet

    def __len__(self):
        return self.triplet_num


class TestTripletDataset(torch.utils.data.Dataset):
    def __init__(self, datafile: str) -> None:
        super(TestTripletDataset, self).__init__()
        self.twosome = {"h": [], "r": [], "t": []}
        self.twosome_num = 0
        with open(datafile, "r", encoding="utf-8") as fin:
            for line in fin:
                h, r, t = line.strip().split("\t")
                self.twosome["h"].append(int(h))
                self.twosome["r"].append(int(r))
                self.twosome_num += 1
        self.twosome["h"] = torch.LongTensor(self.twosome["h"])
        self.twosome["r"] = torch.LongTensor(self.twosome["r"])

    def __getitem__(self, idx):
        twosome = {key: torch.LongTensor([val[idx]]) for key, val in self.twosome.items()}
        return twosome

    def __len__(self):
        return self.twosome_num
