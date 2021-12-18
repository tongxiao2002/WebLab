import torch
import random
import numpy as np
import torch.nn.functional as F


class TransE(torch.nn.Module):
    def __init__(self, entity_word2idx: dict,
                       relation_word2idx: dict,
                       emb_dim: int=200,
                       entity_pre_embedding: np.ndarray=None,
                       relation_pre_embedding: np.ndarray=None):
        super(TransE, self).__init__()
        self.entity_word2idx = entity_word2idx
        self.relation_word2idx = relation_word2idx
        entity_word_num = len(self.entity_word2idx)
        relation_word_num = len(self.relation_word2idx)
        self.entity_embedding = torch.nn.Embedding(entity_word_num, emb_dim)
        self.relation_embedding = torch.nn.Embedding(relation_word_num, emb_dim)
        if entity_pre_embedding is not None and relation_pre_embedding is not None:
            # self.entity_embedding.weight.data.copy_(torch.from_numpy(entity_pre_embedding))
            # self.relation_embedding.weight.data.copy_(torch.from_numpy(relation_pre_embedding))
            torch.nn.init.xavier_normal_(self.entity_embedding.weight.data)
            torch.nn.init.xavier_normal_(self.relation_embedding.weight.data)
        else:
            torch.nn.init.xavier_normal_(self.entity_embedding.weight.data)
            torch.nn.init.xavier_normal_(self.relation_embedding.weight.data)
        self.relation_embedding.weight.data = F.normalize(self.relation_embedding.weight.data, 2, -1)

        self.entity_idx2word = {}
        for k, v in self.entity_word2idx.items():
            self.entity_idx2word[v] = k

    def forward(self, data):
        h, r, t = data
        h = h.flatten()
        h = torch.LongTensor([[self.entity_word2idx[word.item()]] for word in h]).to(h.device)
        r = r.flatten()
        r = torch.LongTensor([[self.relation_word2idx[word.item()]] for word in r]).to(h.device)
        t = t.flatten()
        t = torch.LongTensor([[self.entity_word2idx[word.item()]] for word in t]).to(h.device)
        h = self.entity_embedding(h)
        t = self.entity_embedding(t)
        r = self.relation_embedding(r)
        # neg_h_idx = torch.LongTensor([[random.randint(0, len(self.entity_word2idx) - 1)] for _ in range(len(h))]).to(h.device)
        neg_t_idx = torch.LongTensor([[random.randint(0, len(self.entity_word2idx) - 1)] for _ in range(len(t))]).to(h.device)   # random tail entity
        # neg_t = torch.LongTensor([[self.entity_idx2word[t_idx]] for t_idx in neg_t_idx]).to(h.device)
        # neg_h = self.entity_embedding(neg_h_idx)
        neg_t = self.entity_embedding(neg_t_idx)
        # h = F.normalize(h, 2, -1)
        # t = F.normalize(t, 2, -1)
        # r = F.normalize(r, 2, -1)
        # neg_h = F.normalize(neg_h, 2, -1)
        # neg_t = F.normalize(neg_t, 2, -1)
        score = (h + r) - t
        neg_score = (h + r) - neg_t
        return score, neg_score

    def predict(self, data):
        h, r = data
        h = h.flatten()
        h = torch.LongTensor([[self.entity_word2idx[word.item()]] for word in h]).to(h.device)
        r = r.flatten()
        r = torch.LongTensor([[self.relation_word2idx[word.item()]] for word in r]).to(h.device)
        h = self.entity_embedding(h)
        r = self.relation_embedding(r)
        predict = h + r
        results = {"hit@1": [], "hit@5": [], "hit@10": []}
        # for vec in predict:
        #     predict_idx = torch.argmax(torch.cosine_similarity(vec, self.entity_embedding.weight.data, dim=1))
        #     results.append(self.entity_idx2word[predict_idx.item()])
        # normed_predict = F.normalize(predict, dim=-1)
        # normed_entity_embedding = F.normalize(self.entity_embedding.weight.data, dim=-1)

        # cosine_matrix: (batch_size, entity_num)
        # cosine_matrix = torch.matmul(normed_predict, normed_entity_embedding.T)
        # results = torch.argmax(cosine_matrix, dim=-1)       # 取最大预测
        # results = torch.LongTensor([[self.entity_idx2word[entity.item()]] for entity in results])
        for tail_vec in predict:
            scoreMat = tail_vec - self.entity_embedding.weight.data
            scorelist = torch.norm(scoreMat, p=2, dim=-1, keepdim=False)
            result_idx = torch.topk(scorelist, k=10, dim=-1, largest=False)[1].tolist()  # 只要 indices
            result_idx = [self.entity_idx2word[idx] for idx in result_idx]
            results["hit@1"].append(result_idx[:1])
            results["hit@5"].append(result_idx[:5])
            results["hit@10"].append(result_idx[:10])
        return results

    def normalize_embedding(self):
        self.entity_embedding.weight.data = F.normalize(self.entity_embedding.weight.data, 2, -1)


class TransH(torch.nn.Module):
    def __init__(self, entity_word2idx: dict,
                       relation_word2idx: dict,
                       emb_dim: int=200,
                       margin: float=1.0,
                       C: float=1.0,
                       eps: float=0.001,
                       entity_pre_embedding: np.ndarray=None,
                       relation_pre_embedding: np.ndarray=None):
        super(TransH, self).__init__()
        self.entity_word2idx = entity_word2idx
        self.relation_word2idx = relation_word2idx
        entity_word_num = len(self.entity_word2idx)
        relation_word_num = len(self.relation_word2idx)
        self.entity_embedding = torch.nn.Embedding(entity_word_num, emb_dim)
        self.relation_embedding = torch.nn.Embedding(relation_word_num, emb_dim)
        self.norm_embedding = torch.nn.Embedding(relation_word_num, emb_dim)
        self.margin = margin
        self.C = C
        self.eps = eps
        torch.nn.init.xavier_normal_(self.norm_embedding.weight.data)       # 初始化参数
        if entity_pre_embedding is not None and relation_pre_embedding is not None:
            self.entity_embedding.weight.data.copy_(torch.from_numpy(entity_pre_embedding))
            self.relation_embedding.weight.data.copy_(torch.from_numpy(relation_pre_embedding))
        else:
            torch.nn.init.xavier_normal_(self.entity_embedding.weight.data)
            torch.nn.init.xavier_normal_(self.relation_embedding.weight.data)
        self.entity_idx2word = {}
        for k, v in self.entity_word2idx.items():
            self.entity_idx2word[v] = k

    def _transfer(self, vector, norm):
        norm = F.normalize(norm, 2, dim=-1)
        return vector - torch.sum(norm * vector, dim=-1, keepdim=True) * norm

    def forward(self, data):
        h, batch_r, t = data
        h = h.flatten()
        h = torch.LongTensor([[self.entity_word2idx[word.item()]] for word in h]).to(h.device)
        batch_r = batch_r.flatten()
        batch_r = torch.LongTensor([[self.relation_word2idx[word.item()]] for word in batch_r]).to(h.device)
        t = t.flatten()
        t = torch.LongTensor([[self.entity_word2idx[word.item()]] for word in t]).to(h.device)
        h = self.entity_embedding(h)
        t = self.entity_embedding(t)
        r = self.relation_embedding(batch_r)
        r_norm = self.norm_embedding(batch_r)
        h_v = self._transfer(h, r_norm)
        t_v = self._transfer(t, r_norm)
        # h, t, r, r_norm: (batch_size, emb_dim)
        neg_t_idx = torch.LongTensor([[random.randint(1, len(self.entity_word2idx) - 1)] for _ in range(len(t))]).to(h.device)   # random tail entity except _UNK
        # neg_t = torch.LongTensor([[self.entity_idx2word[t_idx]] for t_idx in neg_t_idx]).to(h.device)
        neg_t = self.entity_embedding(neg_t_idx)
        neg_t_v = self._transfer(neg_t, r_norm)
        # h_v = F.normalize(h_v, 2, -1)
        # t_v = F.normalize(t_v, 2, -1)
        # r = F.normalize(r, 2, -1)
        # neg_t_v = F.normalize(neg_t_v, 2, -1)
        score = torch.norm((h_v + r) - t_v, p=2, dim=-1).flatten()
        neg_score = torch.norm((h_v + r) - neg_t_v, p=2, dim=-1).flatten()

        margin_loss = F.relu(score - neg_score + self.margin).mean()
        entity_loss = F.relu(torch.norm(self.entity_embedding.weight.data, p=2, dim=-1) - 1).mean()
        orth_loss = F.relu(torch.sum(self.relation_embedding.weight.data * self.norm_embedding.weight.data, dim=-1) / torch.norm(self.relation_embedding.weight.data, p=2, dim=-1) - self.eps ** 2).mean()
        return margin_loss + self.C * (entity_loss + orth_loss)

    def predict(self, data):
        h, batch_r = data
        h = h.flatten()
        h = torch.LongTensor([[self.entity_word2idx[word.item()]] for word in h]).to(h.device)
        batch_r = batch_r.flatten()
        batch_r = torch.LongTensor([[self.relation_word2idx[word.item()]] for word in batch_r]).to(h.device)
        h = self.entity_embedding(h)
        r = self.relation_embedding(batch_r)
        r_norm = self.norm_embedding(batch_r)
        h_v = self._transfer(h, r_norm)
        predict = h_v + r
        # results = []
        # for vec in predict:
        #     predict_idx = torch.argmax(torch.cosine_similarity(vec, self.entity_embedding.weight.data, dim=1))
        #     results.append(self.entity_idx2word[predict_idx.item()])
        # r_norm = F.normalize(r_norm, 2, dim=-1).squeeze(dim=1)
        results = {"hit@1": [], "hit@5": [], "hit@10": []}
        for tail_predict, norm_Hyper in zip(predict, r_norm):
            # compute entity_embedding in Hyperplane
            entity_hyper_embedding = self.entity_embedding.weight.data - torch.matmul(torch.sum(self.entity_embedding.weight.data * norm_Hyper,
                                                                                                dim=-1,
                                                                                                keepdim=True), norm_Hyper)

            scoreMat = tail_predict - entity_hyper_embedding
            scorelist = torch.norm(scoreMat, p=2, dim=-1, keepdim=False)
            result_idx = torch.topk(scorelist, k=10, dim=-1, largest=False)[1].tolist()  # 只要 indices
            result_idx = [self.entity_idx2word[idx] for idx in result_idx]
            results["hit@1"].append(result_idx[:1])
            results["hit@5"].append(result_idx[:5])
            results["hit@10"].append(result_idx[:10])
        return results

    def normalize_embedding(self):
        self.entity_embedding.weight.data = F.normalize(self.entity_embedding.weight.data, 2, -1)
        self.norm_embedding.weight.data = F.normalize(self.norm_embedding.weight.data, 2, -1)


class MarginLoss(torch.nn.Module):
    def __init__(self, margin: float=1.0):
        super(MarginLoss, self).__init__()
        self.margin = margin

    def forward(self, score, neg_score):
        score = torch.norm(score, 2, -1).flatten()
        neg_score = torch.norm(neg_score, 2, -1).flatten()
        loss = score - neg_score + self.margin
        return F.relu(loss).mean()
