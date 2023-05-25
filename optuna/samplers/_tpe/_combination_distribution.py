import numpy as np

EPS = 1e-8
def batch_sample_combination(beta: np.ndarray, n: int, N: int, rng: np.random.RandomState) -> np.ndarray:
    beta = np.minimum(beta, 1 - EPS) # Avoid beta=1

    z = rng.uniform(size=(N, n))
    pns = beta[:, None] ** (n - np.arange(n))[None, :]
    z = (np.log(pns + z * (1-pns)) / np.log(beta)[:, None]).astype(int)
    w = z.copy()
    for i in range(1, n):
        w[:, i:] += (w[:, i:] >= z[:, :-i])
    return w

def batch_inversion_number(w: np.ndarray) -> np.ndarray:
    inversion_num = np.zeros(w.shape[:-1], dtype=int)
    for i in range(w.shape[-1]):
        inversion_num += np.sum(w[..., i + 1:] < w[..., i, None], axis=-1)
    return inversion_num

def batch_log_pdf(beta: np.ndarray, w: np.ndarray) -> np.ndarray:
    beta = np.minimum(beta, 1 - EPS) # Avoid beta=1

    n = w.shape[-1]
    inv_nums = batch_inversion_number(w)
    log_scale = np.sum(np.log((1 - beta[..., None] ** (n - np.arange(n))) / (1 - beta[..., None])), axis=-1)
    return inv_nums * np.maximum(np.log(beta), -1.0/EPS) - log_scale
