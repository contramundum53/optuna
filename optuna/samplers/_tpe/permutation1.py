#%%

import numpy as np
import math
import itertools
N=7

NN = math.factorial(N)

J = np.array(list(itertools.permutations(range(N))))

I = np.full((N,) * N, -1, int)
I[tuple(J[:, i] for i in range(N))] = np.arange(NN)

J2 = J[np.arange(NN)[:, None, None], J[np.arange(NN)[None, :]]]
G = I[tuple(J2[..., i] for i in range(N))]

Jinv = np.empty((NN, N), int)
Jinv[np.arange(NN)[:, None], J] = np.arange(N)[None, :]
Ginv = I[tuple(Jinv[..., i] for i in range(N))]

Iswap = np.array(list(itertools.combinations(range(N), 2)))
Jswap = np.array([np.arange(N)] * Iswap.shape[0])
Jswap[np.arange(Jswap.shape[0])[:, None], Iswap] = Iswap[:, ::-1]
Gswap = I[tuple(Jswap[..., i] for i in range(N))]



M0 = np.ones(Iswap.shape[0])
M = np.zeros((NN, NN), np.float64)
M[np.arange(NN)[None, :], G[Gswap]] = M0[:, None]
np.fill_diagonal(M, -np.sum(M, axis=1))
eigvals = np.linalg.eigvalsh(M)
from collections import Counter


counter = Counter((eigvals - 0.5).astype(int))
print(dict(counter))
P, Q = np.linalg.eigh(M)

#%%
Q[:, 0]
#%%


M0 = np.random.rand(N, N)
M0 = (M0 + M0.T) / 2
np.fill_diagonal(M0, 0)
#%%

E = np.zeros((NN, NN, NN), np.int64)
E[np.arange(NN)[:, None], np.arange(NN)[None, :], I[tu]]
M = np.zeros((NN, NN), np.float64)



