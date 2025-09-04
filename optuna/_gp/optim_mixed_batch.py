import typing

import greenlet
import numpy as np

from optuna._gp.acqf import BaseAcquisitionFunc
from optuna._gp.optim_mixed import local_search_mixed
from optuna.logging import get_logger


if typing.TYPE_CHECKING:
    import torch

_logger = get_logger(__name__)


class BatchEvaluatedAcquisitionFunc(BaseAcquisitionFunc):
    def __init__(self, underlying_acquisition_func: BaseAcquisitionFunc) -> None:
        super().__init__(
            underlying_acquisition_func.length_scales,
            underlying_acquisition_func.search_space,
        )
        self._underlying_acquisition_func = underlying_acquisition_func

    def eval_acqf(self, x: "torch.Tensor") -> "torch.Tensor":
        raise NotImplementedError(
            "BatchEvaluatedAcquisitionFunc.eval_acqf is not implemented. "
            "Please use eval_acqf_no_grad or eval_acqf_with_grad."
        )

    def eval_acqf_no_grad(self, x: np.ndarray) -> np.ndarray:
        return greenlet.getcurrent().parent.switch((False, x))

    def eval_acqf_with_grad(self, x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        return greenlet.getcurrent().parent.switch((True, x))


def local_search_mixed_batch(
    acqf: BaseAcquisitionFunc,
    initial_normalized_params: np.ndarray,  # shape: (batch, dim)
    *,
    tol: float = 1e-4,
    max_iter: int = 100,
) -> tuple[np.ndarray, np.ndarray]:
    batch_size, dim = initial_normalized_params.shape

    def optimize_task_factory(
        x0: np.ndarray,
    ) -> typing.Callable[[float], tuple[np.ndarray, float]]:
        def optimize_task(_y0: float) -> tuple[np.ndarray, float]:
            return local_search_mixed(
                BatchEvaluatedAcquisitionFunc(acqf),
                x0,
                tol=tol,
                max_iter=max_iter,
            )

        return optimize_task

    optimize_greenlets = [
        greenlet.greenlet(optimize_task_factory(initial_normalized_params[i]))
        for i in range(batch_size)
    ]

    active_tasks = list(range(batch_size))

    res_ys = np.full(batch_size, np.nan)
    res_xs = np.full((batch_size, dim), np.nan)
    temp_ys = np.full(batch_size, np.nan)
    temp_grads = np.full((batch_size, dim), np.nan)
    temp_xs = np.full((batch_size, dim), np.nan)
    requires_grad = np.zeros(batch_size, dtype=np.bool_)
    while True:
        for i in range(len(active_tasks) - 1, -1, -1):
            task_i = active_tasks[i]
            if requires_grad[i]:
                res = optimize_greenlets[task_i].switch(temp_ys[i], temp_grads[i])
            else:
                res = optimize_greenlets[task_i].switch(temp_ys[i])
            if optimize_greenlets[task_i].dead:
                res_xs[task_i], res_ys[task_i] = res
                active_tasks[i] = active_tasks[-1]
                active_tasks.pop()
                if len(active_tasks) == 0:
                    return (np.array(res_xs), np.array(res_ys))
            else:
                requires_grad[i], temp_xs[i] = res

        # TODO(contramundum53): Can be made smarter
        if np.any(requires_grad[: len(active_tasks)]):
            temp_ys, temp_grads = acqf.eval_acqf_with_grad(temp_xs[: len(active_tasks)])
        else:
            temp_ys = acqf.eval_acqf_no_grad(temp_xs[: len(active_tasks)])


def optimize_acqf_mixed_batch(
    acqf: BaseAcquisitionFunc,
    *,
    warmstart_normalized_params_array: np.ndarray | None = None,
    n_preliminary_samples: int = 2048,
    n_local_search: int = 10,
    tol: float = 1e-4,
    rng: np.random.RandomState | None = None,
) -> tuple[np.ndarray, float]:
    rng = rng or np.random.RandomState()

    if warmstart_normalized_params_array is None:
        warmstart_normalized_params_array = np.empty((0, acqf.search_space.dim))

    assert (
        len(warmstart_normalized_params_array) <= n_local_search - 1
    ), "We must choose at least 1 best sampled point + given_initial_xs as start points."

    sampled_xs = acqf.search_space.sample_normalized_params(n_preliminary_samples, rng=rng)

    # Evaluate all values at initial samples
    f_vals = acqf.eval_acqf_no_grad(sampled_xs)
    assert isinstance(f_vals, np.ndarray)

    max_i = np.argmax(f_vals)

    # TODO(nabenabe): Benchmark the BoTorch roulette selection as well.
    # https://github.com/pytorch/botorch/blob/v0.14.0/botorch/optim/initializers.py#L942
    # We use a modified roulette wheel selection to pick the initial param for each local search.
    probs = np.exp(f_vals - f_vals[max_i])
    probs[max_i] = 0.0  # We already picked the best param, so remove it from roulette.
    probs /= probs.sum()
    n_non_zero_probs_improvement = int(np.count_nonzero(probs > 0.0))
    # n_additional_warmstart becomes smaller when study starts to converge.
    n_additional_warmstart = min(
        n_local_search - len(warmstart_normalized_params_array) - 1,
        n_non_zero_probs_improvement,
    )
    if n_additional_warmstart == n_non_zero_probs_improvement:
        _logger.warning("Study already converged, so the number of local search is reduced.")
    chosen_idxs = np.array([max_i])
    if n_additional_warmstart > 0:
        additional_idxs = rng.choice(
            len(sampled_xs), size=n_additional_warmstart, replace=False, p=probs
        )
        chosen_idxs = np.append(chosen_idxs, additional_idxs)

    best_x = sampled_xs[max_i, :]
    best_f = float(f_vals[max_i])

    warmstart_xs = np.vstack([sampled_xs[chosen_idxs, :], warmstart_normalized_params_array])

    opt_xs, opt_fs = local_search_mixed_batch(
        acqf,
        warmstart_xs,
        tol=tol,
    )

    max_i = np.argmax(opt_fs)

    if best_f < float(opt_fs[max_i]):
        return opt_xs[max_i, :], float(opt_fs[max_i])
    else:
        return best_x, best_f
