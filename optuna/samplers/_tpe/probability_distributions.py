from typing import List
from typing import NamedTuple
from typing import Union

import numpy as np

from optuna.samplers._tpe import _truncnorm
from optuna.samplers._tpe import _permutation_distribution


class _BatchedCategoricalDistributions(NamedTuple):
    weights: np.ndarray


class _BatchedTruncNormDistributions(NamedTuple):
    mu: np.ndarray
    sigma: np.ndarray
    low: float  # Currently, low and high do not change per trial.
    high: float


class _BatchedDiscreteTruncNormDistributions(NamedTuple):
    mu: np.ndarray
    sigma: np.ndarray
    low: float  # Currently, low, high and step do not change per trial.
    high: float
    step: float

class _BatchedPermutationDistributions(NamedTuple):
    origin: np.ndarray
    beta: np.ndarray

_BatchedDistributions = Union[
    _BatchedCategoricalDistributions,
    _BatchedTruncNormDistributions,
    _BatchedDiscreteTruncNormDistributions,
    _BatchedPermutationDistributions,
]


def _get_sampled_type(batched_distribution: _BatchedDistributions) -> tuple[np.dtype, tuple[...]]:
    if isinstance(batched_distribution, (_BatchedCategoricalDistributions, _BatchedTruncNormDistributions, _BatchedDiscreteTruncNormDistributions)):
        return (np.float64, ())
    elif isinstance(batched_distribution, _BatchedPermutationDistributions):
        return (int, (batched_distribution.origin.shape[1],))
    else:
        raise NotImplementedError

class _MixtureOfProductDistribution(NamedTuple):
    weights: np.ndarray
    distributions: dict[str, _BatchedDistributions]


    def sample(self, rng: np.random.RandomState, batch_size: int) -> np.ndarray:
        active_indices = rng.choice(len(self.weights), p=self.weights, size=batch_size)

        dtype = [
            (param, *_get_sampled_type(d)) for param, d in self.distributions.items()
        ]
        ret = np.empty((batch_size,), dtype=dtype)
        for i, (param, d) in enumerate(self.distributions.items()):
            if isinstance(d, _BatchedCategoricalDistributions):
                active_weights = d.weights[active_indices, :]
                rnd_quantile = rng.rand(batch_size)
                cum_probs = np.cumsum(active_weights, axis=-1)
                assert np.isclose(cum_probs[:, -1], 1).all()
                cum_probs[:, -1] = 1  # Avoid numerical errors.
                ret[param] = np.sum(cum_probs < rnd_quantile[:, None], axis=-1)
            elif isinstance(d, _BatchedTruncNormDistributions):
                active_mus = d.mu[active_indices]
                active_sigmas = d.sigma[active_indices]
                ret[param] = _truncnorm.rvs(
                    a=(d.low - active_mus) / active_sigmas,
                    b=(d.high - active_mus) / active_sigmas,
                    loc=active_mus,
                    scale=active_sigmas,
                    random_state=rng,
                )
            elif isinstance(d, _BatchedDiscreteTruncNormDistributions):
                active_mus = d.mu[active_indices]
                active_sigmas = d.sigma[active_indices]
                samples = _truncnorm.rvs(
                    a=(d.low - d.step / 2 - active_mus) / active_sigmas,
                    b=(d.high + d.step / 2 - active_mus) / active_sigmas,
                    loc=active_mus,
                    scale=active_sigmas,
                    random_state=rng,
                )
                ret[param] = np.clip(
                    d.low + np.round((samples - d.low) / d.step) * d.step, d.low, d.high
                )
            elif isinstance(d, _BatchedPermutationDistributions):
                perm = _permutation_distribution.batch_sample_permutation(
                    d.beta[active_indices], d.origin.shape[1], batch_size, rng
                )
                ret[param] = d.origin[active_indices[:, None], perm]
            else:
                assert False

        return ret

    def log_pdf(self, x: np.ndarray) -> np.ndarray:
        batch_size = len(x)
        n_vars = len(self.distributions)

        log_pdfs = np.empty((batch_size, len(self.weights), n_vars), dtype=np.float64)
        for i, (param, d) in enumerate(self.distributions.items()):
            xi = x[param]
            if isinstance(d, _BatchedCategoricalDistributions):
                log_pdfs[:, :, i] = np.log(
                    np.take_along_axis(
                        d.weights[None, :, :], xi[:, None, None].astype(np.int64), axis=-1
                    )
                )[:, :, 0]
            elif isinstance(d, _BatchedTruncNormDistributions):
                log_pdfs[:, :, i] = _truncnorm.logpdf(
                    x=xi[:, None],
                    a=(d.low - d.mu[None, :]) / d.sigma[None, :],
                    b=(d.high - d.mu[None, :]) / d.sigma[None, :],
                    loc=d.mu[None, :],
                    scale=d.sigma[None, :],
                )
            elif isinstance(d, _BatchedDiscreteTruncNormDistributions):
                lower_limit = d.low - d.step / 2
                upper_limit = d.high + d.step / 2
                x_lower = np.maximum(xi - d.step / 2, lower_limit)
                x_upper = np.minimum(xi + d.step / 2, upper_limit)
                log_gauss_mass = _truncnorm._log_gauss_mass(
                    (x_lower[:, None] - d.mu[None, :]) / d.sigma[None, :],
                    (x_upper[:, None] - d.mu[None, :]) / d.sigma[None, :],
                )
                log_p_accept = _truncnorm._log_gauss_mass(
                    (d.low - d.step / 2 - d.mu[None, :]) / d.sigma[None, :],
                    (d.high + d.step / 2 - d.mu[None, :]) / d.sigma[None, :],
                )
                log_pdfs[:, :, i] = log_gauss_mass - log_p_accept
            elif isinstance(d, _BatchedPermutationDistributions):
                perm = np.empty((batch_size, d.origin.shape[0], d.origin.shape[1]), dtype=int)
                perm[
                    np.arange(batch_size)[:, None, None], 
                    np.arange(d.origin.shape[0])[None, :, None],
                    d.origin[None, :, :]
                ] = xi[:, None, :]
                log_pdfs[:, :, i] = _permutation_distribution.batch_log_pdf(
                    d.beta[None, :], perm
                )
            else:
                assert False
        weighted_log_pdf = np.sum(log_pdfs, axis=-1) + np.log(self.weights[None, :])
        max_ = weighted_log_pdf.max(axis=1)
        # We need to avoid (-inf) - (-inf) when the probability is zero.
        max_[np.isneginf(max_)] = 0
        with np.errstate(divide="ignore"):  # Suppress warning in log(0).
            return np.log(np.exp(weighted_log_pdf - max_[:, None]).sum(axis=1)) + max_
