import numpy as np
from sklearn.cluster import KMeans
import math
import torch
import os
os.environ["OMP_NUM_THREADS"] = "1"


class GranularBall:
    def __init__(self, indices, center, radius, labels=None):
        self.indices = np.array(indices, dtype=int)
        self.center = np.array(center, dtype=float)
        self.radius = float(radius)
        self.labels = None if labels is None else np.array(labels)
        self.purity = None
        if labels is not None:
            self._compute_purity()

    def _compute_purity(self):
        if self.labels is None or len(self.labels) == 0:
            self.purity = 0.0
        else:
            vals, counts = np.unique(self.labels, return_counts=True)
            self.purity = float(np.max(counts)) / float(np.sum(counts))


def build_granular_balls(X, labels=None, purity_threshold=1.0, min_size=10, max_iter=20, verbose=False):
    n, d = X.shape
    idx_all = np.arange(n)

    def make_ball(indices):
        pts = X[indices]
        center = pts.mean(axis=0)
        dists = np.linalg.norm(pts - center[None, :], axis=1)
        radius = float(np.max(dists)) if len(dists) > 0 else 0.0
        lab = None
        if labels is not None:
            lab = labels[indices]
        return GranularBall(indices=indices, center=center, radius=radius, labels=lab)

    root = make_ball(idx_all)
    balls = [root]
    it = 0
    while it < max_iter:
        it += 1
        changed = False
        new_balls = []
        for ball in balls:
            if labels is None:
                if len(ball.indices) <= min_size:
                    new_balls.append(ball)
                    continue
                
                overall_range = np.max(X, axis=0) - np.min(X, axis=0)
                range_norm = np.linalg.norm(overall_range)
                if ball.radius < 1e-6 or ball.radius < 0.02 * (range_norm + 1e-8):
                    new_balls.append(ball)
                    continue
            else:
                if ball.purity is None:
                    ball._compute_purity()
                if ball.purity >= purity_threshold or len(ball.indices) <= min_size:
                    new_balls.append(ball)
                    continue

            pts = X[ball.indices]
            if len(pts) <= 2:
                new_balls.append(ball)
                continue
            try:
                km = KMeans(n_clusters=2, random_state=42, n_init=5).fit(pts)
            except Exception:
                new_balls.append(ball)
                continue
            
            labels_k = km.labels_
            idx0 = ball.indices[labels_k == 0]
            idx1 = ball.indices[labels_k == 1]
            
            if len(idx0) == 0 or len(idx1) == 0:
                new_balls.append(ball)
                continue
            
            child0 = make_ball(idx0)
            child1 = make_ball(idx1)

            if labels is not None:
                parent_quality = ball.purity * len(ball.indices)
                child_quality = (child0.purity * len(child0.indices) + child1.purity * len(child1.indices))
                if child_quality <= parent_quality + 1e-8:
                    new_balls.append(ball)
                    continue

            new_balls.append(child0)
            new_balls.append(child1)
            changed = True

        balls = new_balls
        if verbose:
            print(f"[GB] iter={it}, balls={len(balls)}")
        if not changed:
            break
    return balls


def smooth_item_embeddings_numpy(item_emb, granular_balls, alpha=0.6, mode='intra'):
    emb = item_emb.copy()
    new_emb = emb.copy()
    for gb in granular_balls:
        member_idx = gb.indices
        if len(member_idx) == 0:
            continue
        center = np.mean(emb[member_idx], axis=0)
        new_emb[member_idx] = (1.0 - alpha) * emb[member_idx] + alpha * center[None, :]
    return new_emb


def smooth_item_embeddings_torch(item_emb_tensor, granular_balls, alpha=0.6, device='cpu'):
    if isinstance(item_emb_tensor, np.ndarray):
        item_emb_tensor = torch.from_numpy(item_emb_tensor)
    item_emb = item_emb_tensor.detach().cpu()
    new = item_emb.clone()
    for gb in granular_balls:
        idx = torch.LongTensor(gb.indices)
        if idx.numel() == 0:
            continue
        center = torch.mean(item_emb[idx], dim=0, keepdim=True)
        new[idx] = (1.0 - alpha) * item_emb[idx] + alpha * center
    return new.to(device)


def gb_summary(balls, topk=10):
    s = []
    for i, gb in enumerate(sorted(balls, key=lambda x: -len(x.indices))[:topk]):
        s.append((i, len(gb.indices), gb.radius, gb.purity))
    return s