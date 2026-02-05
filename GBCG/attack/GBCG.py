import numpy as np
import random
import torch
import torch.nn as nn
import scipy.sparse as sp
from matplotlib import pyplot as plt
from scipy.sparse import vstack, csr_matrix
from collections import defaultdict
from util.tool import targetItemSelect

class GBCG():
    def __init__(self, LightGCN_model, arg, data):
        self.model = LightGCN_model
        self.interact = data.matrix()
        self.userNum = data.user_num
        self.itemNum = data.item_num

        self.G_losses = []
        self.D_losses = []

        with open("","r") as f:
            raw = f.read().strip().split(", ")
        raw = [x.strip("'") for x in raw]
        self.targetItem = [data.item[x] for x in raw if x in data.item]
        print("targetItem:", self.targetItem)
        if not self.targetItem:
            raise ValueError("No valid target items")

        self.item_embeddings = LightGCN_model.get_item_embedding()
        self.item_popularity = np.array(self.interact.sum(axis=0)).flatten()

        self.filtered_items = self._filter_by_cosine(0.65)
        self.item_clusters = self._cluster_items(self.filtered_items)

        self.maliciousFeedbackSize = arg.maliciousFeedbackSize
        self.maliciousUserSize = arg.maliciousUserSize
        self._init_parameters()

        self.select_len = len(self.targetItem) + self.maliciousFeedbackNum

        self.G = None
        self.D = None
        self.BiLevelOptimizationEpoch = 100
        self.template_nz_counts = []
        self.G_lr = arg.lr_G
        self.D_lr = arg.lr_D

    def _init_parameters(self):
        self.itemP = np.array((self.interact.sum(0) / self.interact.sum()))[0]
        self.itemP[self.targetItem] = 0

        if self.maliciousFeedbackSize == 0:
            self.maliciousFeedbackNum = int(self.interact.sum() / self.userNum)
            print("maliciousFeedbackNum:", self.maliciousFeedbackNum)
        elif self.maliciousFeedbackSize >= 1:
            self.maliciousFeedbackNum = self.maliciousFeedbackSize
        else:
            self.maliciousFeedbackNum = int(self.maliciousFeedbackSize * self.itemNum)

        if self.maliciousUserSize < 1:
            self.fakeUserNum = int(self.userNum * self.maliciousUserSize)
        else:
            self.fakeUserNum = int(self.maliciousUserSize)

        self.attackForm = "dataAttack"
        self.recommenderGradientRequired = False
        self.recommenderModelRequired = False

    def _filter_by_cosine(self, threshold):
        E = self.item_embeddings
        normE = E / np.linalg.norm(E, axis=1, keepdims=True)
        T = normE[self.targetItem]
        cos = normE.dot(T.T)
        ms = cos.max(axis=1)
        filt = np.where(ms > threshold)[0]
        return filt if len(filt) > 0 else np.arange(self.itemNum)

    def _cluster_items(self, items, purity_threshold: float = 0.8):
        emb = self.item_embeddings[items]
        target_emb = self.item_embeddings[self.targetItem]
        diff = emb[:, None, :] - target_emb[None, :, :]
        dists = np.linalg.norm(diff, axis=2)
        pseudo_labels = dists.argmin(axis=1)

        clusters = defaultdict(list)
        for idx, lbl in zip(items, pseudo_labels):
            clusters[int(lbl)].append(int(idx))

        cluster_dict = {}
        for new_id, old_id in enumerate(clusters.keys()):
            cluster_dict[new_id] = clusters[old_id]
        return cluster_dict

    def _full_predict(self, X):
        emb = torch.FloatTensor(self.item_embeddings).cuda()
        U = X @ emb
        return U @ emb.t()

    def _counterfactual_gain(self, sel_items):
        users = np.random.choice(self.userNum, self.fakeUserNum, replace=False)
        Xb = self.interact[users].toarray()
        Xb_t = torch.FloatTensor(Xb).cuda()

        with torch.no_grad():
            base = self._full_predict(Xb_t)
            s_base = base[:, self.targetItem].sum().item()

        Xcf = Xb.copy()
        Xcf[:, sel_items] = 1
        Xcf_t = torch.FloatTensor(Xcf).cuda()
        with torch.no_grad():
            cf = self._full_predict(Xcf_t)
            s_cf = cf[:, self.targetItem].sum().item()

        return s_cf - s_base

    def posionDataAttack(self, epoch1=50, epoch2=50):
        candidates = set(self.filtered_items) - set(self.targetItem)
        print("candidates_len:", len(candidates))
        select = set(self.targetItem)
        print(f"=== Starting greedy selection, initial size={len(select)} ===")
        
        while len(select) < self.select_len:
            best, gain = None, -1e9
            for i in candidates:
                g = self._counterfactual_gain(list(select | {i}))
                if g > gain:
                    gain, best = g, i
            select.add(best)
            candidates.remove(best)
            print(f"Added {best}, gain {gain:.4f}, current size={len(select)}")
        
        self.selectItem = list(select)
        print("Final selectItem:", self.selectItem)

        if self.G is None:
            self._init_GD(epoch1, epoch2)

        self._plot_loss_curves()
        return self._generate_fake_data()

    def _init_GD(self, epoch1, epoch2):
        self.select_len = len(self.selectItem)
        self.G = Encoder(self.select_len).cuda()
        self.D = Decoder(self.select_len).cuda()
        optG = torch.optim.Adam(self.G.parameters(), lr=self.G_lr)
        optD = torch.optim.Adam(self.D.parameters(), lr=self.D_lr)

        for ep in range(self.BiLevelOptimizationEpoch):
            print(f"BiLevel Epoch {ep + 1}/{self.BiLevelOptimizationEpoch}")

            self.G.eval()
            self.D.train()
            epoch_D_loss = 0.0
            for _ in range(epoch1):
                real = self._sample_batch(self.fakeUserNum)
                Z = torch.randn_like(real)
                loss_D = (self.D(self.G(Z)) - self.D(real)).mean()
                epoch_D_loss += loss_D.item()
                optD.zero_grad(); loss_D.backward(); optD.step()
            
            avg_D_loss = epoch_D_loss / epoch1
            print(f"D_loss: {avg_D_loss}")
            self.D_losses.append(avg_D_loss)

            self.D.eval()
            self.G.train()
            epoch_G_loss = 0.0
            for _ in range(epoch2):
                real = self._sample_batch(self.fakeUserNum)
                Z = torch.randn_like(real)
                fake = self.G(Z)
                lossG = (-self.D(fake) + 0.01 * torch.norm(fake - real) / self.select_len).mean()
                epoch_G_loss += lossG.item()
                optG.zero_grad(); lossG.backward(); optG.step()
            
            avg_G_loss = epoch_G_loss / epoch2
            print("avg_G_loss:", avg_G_loss)
            self.G_losses.append(avg_G_loss)

    def _plot_loss_curves(self):
        plt.figure(figsize=(10, 5))
        plt.plot(self.G_losses, label='Generator Loss', color='blue')
        plt.plot(self.D_losses, label='Discriminator Loss', color='red')
        plt.title("Training Loss History")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.legend()
        plt.grid(True)
        plt.savefig('G_D_Loss_Curves.png')
        plt.close()
        print("Loss curves saved to G_D_Loss_Curves.png")

    def _sample_batch(self, B):
        out = []
        for _ in range(B):
            cid = random.choice(list(self.item_clusters.keys()))
            users, items = self.interact.nonzero()
            mask = np.isin(items, self.item_clusters[cid])
            candU = users[mask]
            if len(candU) == 0:
                vec = np.zeros(self.itemNum)
            else:
                u = np.random.choice(candU)
                vec = self.interact[u].toarray().flatten()
            out.append(vec[self.selectItem])
        return torch.FloatTensor(np.stack(out)).cuda()

    def _generate_fake_data(self):
        rows, cols, vals = [], [], []
        for u in range(self.fakeUserNum):
            Z = torch.randn((1, self.select_len)).cuda()
            rat = self.G(Z).detach().cpu().numpy().flatten()
            binr = self._project(rat, self.maliciousFeedbackNum)
            for idx in np.where(binr > 0)[0]:
                rows.append(u); cols.append(self.selectItem[idx]); vals.append(1)
            for t in self.targetItem:
                rows.append(u); cols.append(t); vals.append(1)

        fake = csr_matrix((vals, (rows, cols)), shape=(self.fakeUserNum, self.itemNum))
        return vstack([self.interact, fake])

    def _project(self, mat, n):
        t, idx = torch.topk(torch.tensor(mat), min(n, len(mat)))
        res = torch.zeros_like(torch.tensor(mat))
        res[idx] = 1
        return res.numpy()

    def _csr_to_tensor(self, data):
        c = sp.csr_matrix(data).tocoo()
        idx = torch.LongTensor([c.row, c.col])
        vals = torch.FloatTensor(c.data)
        return torch.sparse.FloatTensor(idx, vals, c.shape).cuda()

class MLP(nn.Module):
    def __init__(self, inputSize, hiddenSize, sigmoidFunc=False):
        super().__init__()
        layers = []
        for i, h in enumerate(hiddenSize):
            in_dim = inputSize if i == 0 else hiddenSize[i-1]
            layers.append(nn.Linear(in_dim, h))
            layers.append(nn.Sigmoid() if sigmoidFunc else nn.LeakyReLU(0.2))
        self.net = nn.Sequential(*layers)
    def forward(self, x):
        return self.net(x)

class Encoder(nn.Module):
    def __init__(self, k):
        super().__init__()
        self.k = k
        self.G_e = MLP(k, [64, 32, 16*k])
        self.G_l = MLP(k, [64, 32, 16])
        self.G_r = MLP(k*16, [k])
    def forward(self, x):
        L_t = self.G_l(x)
        H = self.G_e(x).view(x.size(0), self.k, 16)
        L = L_t.t() @ L_t
        R = torch.bmm(H, L.unsqueeze(0).repeat(x.size(0),1,1))
        return self.G_r(R.view(x.size(0), -1))

class Decoder(nn.Module):
    def __init__(self, k):
        super().__init__()
        self.D_r = MLP(k, [64, 32, 16, 1], sigmoidFunc=True)
    def forward(self, x):
        return self.D_r(x)