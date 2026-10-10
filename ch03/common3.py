import numpy as np, torch, torch.nn as nn, torch.nn.functional as F
from sklearn.datasets import load_digits
torch.manual_seed(0); np.random.seed(0)
d = load_digits()                                   # 1,797 real 8x8 handwritten digits, intensities 0..16
X_int = torch.tensor(d.data, dtype=torch.long); Y = torch.tensor(d.target)
perm = torch.randperm(len(X_int), generator=torch.Generator().manual_seed(0))
tr, te = perm[:1500], perm[1500:]
Xtr_int, Xte_int = X_int[tr], X_int[te]
Xtr, Xte = Xtr_int.float() / 16, Xte_int.float() / 16          # scaled to [0, 1]
Ytr, Yte = Y[tr], Y[te]
def batches(X, bs=128):
    while True:
        yield X[torch.randint(0, len(X), (bs,))]
