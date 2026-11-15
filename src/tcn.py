import numpy as np, torch, torch.nn as nn


class Block(nn.Module):
    def __init__(self, ch, dil):
        super().__init__()
        self.c1 = nn.Conv1d(ch, ch, 3, padding=dil, dilation=dil)
        self.c2 = nn.Conv1d(ch, ch, 3, padding=dil, dilation=dil)
        self.act = nn.ReLU()

    def forward(self, x):
        return self.act(x + self.c2(self.act(self.c1(x))))


class TCNMAE(nn.Module):
    def __init__(self, d, ch=64, dils=(1, 2, 4)):
        super().__init__()
        self.inp = nn.Conv1d(d + 2, ch, 1)
        self.blocks = nn.Sequential(*[Block(ch, k) for k in dils])
        self.out = nn.Conv1d(ch, d, 1)

    def forward(self, x, valid, masked):
        xin = x * (valid & ~masked).unsqueeze(-1)
        pad = (~valid).float().unsqueeze(-1)
        m = masked.float().unsqueeze(-1)
        h = torch.cat([xin, pad, m], -1).transpose(1, 2)
        return self.out(self.blocks(self.inp(h))).transpose(1, 2)


def random_mask(valid, p=0.25):
    r = torch.rand(valid.shape)
    r[~valid] = 2.0
    masked = (r < p) & valid
    force = r.argmin(1)
    masked[torch.arange(len(valid)), force] = True
    return masked & valid


def masked_loss(model, x, valid, masked):
    err = ((model(x, valid, masked) - x) ** 2).mean(-1)
    return (err * masked).sum() / masked.sum()


@torch.no_grad()
def window_scores(model, X, V, K=4, bs=2048):
    model.eval()
    X, V = torch.from_numpy(X), torch.from_numpy(V)
    L = X.shape[1]
    pos = torch.arange(L)
    out = []
    for i in range(0, len(X), bs):
        x, v = X[i:i + bs], V[i:i + bs]
        tot = torch.zeros(len(x))
        for k in range(K):
            masked = (pos % K == k).unsqueeze(0) & v
            tot += (((model(x, v, masked) - x) ** 2).mean(-1) * masked).sum(1)
        out.append((tot / v.sum(1)).numpy())
    return np.concatenate(out)