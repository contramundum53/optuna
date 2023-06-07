import numpy as np
import functools

@functools.lru_cache
def get_num_possible_swaps(n: int, k: int) -> np.ndarray:
    res = np.zeros((k+1,), dtype=np.float64)
    res[0] = 1.0
    for i in range(1, k + 1):
        res[i] = res[i-1] * (k - i + 1) * (n - k - i + 1) // (i ** 2)
    return res

def batch_sample_combination(beta: np.ndarray, origin: np.ndarray, k: int, rng: np.random.RandomState) -> np.ndarray:
    n = origin.shape[-1]
    N = beta.shape[-1]
    nps = get_num_possible_swaps(n, k)
    probs = np.cumsum(nps[None, :] * beta[:, None] ** np.arange(k+1), axis=-1)
    probs /= probs[:, -1, None]
    n_swaps = np.sum(rng.rand(N)[:, None] > probs, axis=-1)
    result = origin.copy()
    for i in range(N):
        ones_selected = rng.choice(np.argwhere(origin[i, :])[:, 0] , size=(n_swaps[i],), replace=False)
        zeros_selected = rng.choice(np.argwhere(~origin[i, :])[:, 0], size=(n_swaps[i],), replace=False)
        result[i, ones_selected] = False
        result[i, zeros_selected] = True
    return result

def batch_log_pdf(beta: np.ndarray, origin: np.ndarray, k: int, x: np.ndarray) -> np.ndarray:
    return np.log(beta) * 0.5 * np.count_nonzero(origin != x, axis=-1)
    