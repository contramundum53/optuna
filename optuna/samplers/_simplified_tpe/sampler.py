import math
from typing import Any
from typing import Callable
from typing import Container
from typing import Dict
from typing import List
from typing import Optional
from typing import Sequence
from typing import Tuple
from typing import Union
import warnings

import numpy as np

from optuna._hypervolume import WFG
from optuna.distributions import BaseDistribution
from optuna.exceptions import ExperimentalWarning
from optuna.logging import get_logger
from optuna.samplers import _search_space
from optuna.samplers._base import _CONSTRAINTS_KEY
from optuna.samplers._base import _process_constraints_after_trial
from optuna.samplers._base import BaseSampler
from optuna.samplers._random import RandomSampler
from optuna.samplers._search_space import IntersectionSearchSpace
from optuna.samplers._search_space.group_decomposed import _GroupDecomposedSearchSpace
from optuna.samplers._search_space.group_decomposed import _SearchSpaceGroup
from optuna.samplers._simplified_tpe.parzen_estimator import _ParzenEstimator
from optuna.samplers._simplified_tpe.parzen_estimator import _ParzenEstimatorParameters
from optuna.study import Study
from optuna.study._study_direction import StudyDirection
from optuna.trial import FrozenTrial
from optuna.trial import TrialState


EPS = 1e-12
_logger = get_logger(__name__)


def default_gamma(x: int) -> int:

    return min(int(np.ceil(0.1 * x)), 25)


def hyperopt_default_gamma(x: int) -> int:

    return min(int(np.ceil(0.25 * np.sqrt(x))), 25)


def default_weights(x: int) -> np.ndarray:

    if x == 0:
        return np.asarray([])
    elif x < 25:
        return np.ones(x)
    else:
        ramp = np.linspace(1.0 / x, 1.0, num=x - 25)
        flat = np.ones(25)
        return np.concatenate([ramp, flat], axis=0)


class SimplifiedTPESampler(BaseSampler):
    """Sampler using TPE (Tree-structured Parzen Estimator) algorithm.

    This sampler is based on *independent sampling*.
    See also :class:`~optuna.samplers.BaseSampler` for more details of 'independent sampling'.

    On each trial, for each parameter, TPE fits one Gaussian Mixture Model (GMM) ``l(x)`` to
    the set of parameter values associated with the best objective values, and another GMM
    ``g(x)`` to the remaining parameter values. It chooses the parameter value ``x`` that
    maximizes the ratio ``l(x)/g(x)``.

    For further information about TPE algorithm, please refer to the following papers:

    - `Algorithms for Hyper-Parameter Optimization
      <https://papers.nips.cc/paper/4443-algorithms-for-hyper-parameter-optimization.pdf>`_
    - `Making a Science of Model Search: Hyperparameter Optimization in Hundreds of
      Dimensions for Vision Architectures <http://proceedings.mlr.press/v28/bergstra13.pdf>`_
    - `Multiobjective tree-structured parzen estimator for computationally expensive optimization
      problems <https://dl.acm.org/doi/10.1145/3377930.3389817>`_
    - `Multiobjective Tree-Structured Parzen Estimator <https://doi.org/10.1613/jair.1.13188>`_

    Example:

        .. testcode::

            import optuna
            from optuna.samplers import TPESampler


            def objective(trial):
                x = trial.suggest_float("x", -10, 10)
                return x**2


            study = optuna.create_study(sampler=TPESampler())
            study.optimize(objective, n_trials=10)

    Args:
        consider_prior:
            Enhance the stability of Parzen estimator by imposing a Gaussian prior when
            :obj:`True`. The prior is only effective if the sampling distribution is
            either :class:`~optuna.distributions.FloatDistribution`,
            or :class:`~optuna.distributions.IntDistribution`.
        prior_weight:
            The weight of the prior. This argument is used in
            :class:`~optuna.distributions.FloatDistribution`,
            :class:`~optuna.distributions.IntDistribution`, and
            :class:`~optuna.distributions.CategoricalDistribution`.
        consider_magic_clip:
            Enable a heuristic to limit the smallest variances of Gaussians used in
            the Parzen estimator.
        consider_endpoints:
            Take endpoints of domains into account when calculating variances of Gaussians
            in Parzen estimator. See the original paper for details on the heuristics
            to calculate the variances.
        n_startup_trials:
            The random sampling is used instead of the TPE algorithm until the given number
            of trials finish in the same study.
        n_ei_candidates:
            Number of candidate samples used to calculate the expected improvement.
        gamma:
            A function that takes the number of finished trials and returns the number
            of trials to form a density function for samples with low grains.
            See the original paper for more details.
        weights:
            A function that takes the number of finished trials and returns a weight for them.
            See `Making a Science of Model Search: Hyperparameter Optimization in Hundreds of
            Dimensions for Vision Architectures <http://proceedings.mlr.press/v28/bergstra13.pdf>`_
            for more details.

            .. note::
                In the multi-objective case, this argument is only used to compute the weights of
                bad trials, i.e., trials to construct `g(x)` in the `paper
                <https://papers.nips.cc/paper/4443-algorithms-for-hyper-parameter-optimization.pdf>`_
                ). The weights of good trials, i.e., trials to construct `l(x)`, are computed by a
                rule based on the hypervolume contribution proposed in the `paper of MOTPE
                <https://dl.acm.org/doi/10.1145/3377930.3389817>`_.
        seed:
            Seed for random number generator.
        multivariate:
            If this is :obj:`True`, the multivariate TPE is used when suggesting parameters.
            The multivariate TPE is reported to outperform the independent TPE. See `BOHB: Robust
            and Efficient Hyperparameter Optimization at Scale
            <http://proceedings.mlr.press/v80/falkner18a.html>`_ for more details.

            .. note::
                Added in v2.2.0 as an experimental feature. The interface may change in newer
                versions without prior notice. See
                https://github.com/optuna/optuna/releases/tag/v2.2.0.
        group:
            If this and ``multivariate`` are :obj:`True`, the multivariate TPE with the group
            decomposed search space is used when suggesting parameters.
            The sampling algorithm decomposes the search space based on past trials and samples
            from the joint distribution in each decomposed subspace.
            The decomposed subspaces are a partition of the whole search space. Each subspace
            is a maximal subset of the whole search space, which satisfies the following:
            for a trial in completed trials, the intersection of the subspace and the search space
            of the trial becomes subspace itself or an empty set.
            Sampling from the joint distribution on the subspace is realized by multivariate TPE.
            If ``group`` is :obj:`True`, ``multivariate`` must be :obj:`True` as well.

            .. note::
                Added in v2.8.0 as an experimental feature. The interface may change in newer
                versions without prior notice. See
                https://github.com/optuna/optuna/releases/tag/v2.8.0.

            Example:

            .. testcode::

                import optuna


                def objective(trial):
                    x = trial.suggest_categorical("x", ["A", "B"])
                    if x == "A":
                        return trial.suggest_float("y", -10, 10)
                    else:
                        return trial.suggest_int("z", -10, 10)


                sampler = optuna.samplers.TPESampler(multivariate=True, group=True)
                study = optuna.create_study(sampler=sampler)
                study.optimize(objective, n_trials=10)
        warn_independent_sampling:
            If this is :obj:`True` and ``multivariate=True``, a warning message is emitted when
            the value of a parameter is sampled by using an independent sampler.
            If ``multivariate=False``, this flag has no effect.
        constant_liar:
            If :obj:`True`, penalize running trials to avoid suggesting parameter configurations
            nearby.

            .. note::
                Abnormally terminated trials often leave behind a record with a state of
                `RUNNING` in the storage.
                Such "zombie" trial parameters will be avoided by the constant liar algorithm
                during subsequent sampling.
                When using an :class:`~optuna.storages.RDBStorage`, it is possible to enable the
                ``heartbeat_interval`` to change the records for abnormally terminated trials to
                `FAIL`.

            .. note::
                It is recommended to set this value to :obj:`True` during distributed
                optimization to avoid having multiple workers evaluating similar parameter
                configurations. In particular, if each objective function evaluation is costly
                and the durations of the running states are significant, and/or the number of
                workers is high.

            .. note::
                Added in v2.8.0 as an experimental feature. The interface may change in newer
                versions without prior notice. See
                https://github.com/optuna/optuna/releases/tag/v2.8.0.
        constraints_func:
            An optional function that computes the objective constraints. It must take a
            :class:`~optuna.trial.FrozenTrial` and return the constraints. The return value must
            be a sequence of :obj:`float` s. A value strictly larger than 0 means that a
            constraints is violated. A value equal to or smaller than 0 is considered feasible.
            If ``constraints_func`` returns more than one value for a trial, that trial is
            considered feasible if and only if all values are equal to 0 or smaller.

            The ``constraints_func`` will be evaluated after each successful trial.
            The function won't be called when trials fail or they are pruned, but this behavior is
            subject to change in the future releases.

            .. note::
                Added in v3.0.0 as an experimental feature. The interface may change in newer
                versions without prior notice.
                See https://github.com/optuna/optuna/releases/tag/v3.0.0.

    """

    def __init__(
        self,
        search_space: Dict[str, BaseDistribution],
        n_startup_trials: int = 10,
        n_ei_candidates: int = 24,
        gamma: Callable[[int], int] = default_gamma,
        weights: Callable[[int], np.ndarray] = default_weights,
        seed: Optional[int] = None,
    ) -> None:

        self._parzen_estimator_parameters = _ParzenEstimatorParameters(
            consider_prior=True,
            prior_weight=1.0,
            consider_magic_clip=True,
            consider_endpoints=False,
            weights=default_weights,
            multivariate=False,
        )
        self._prior_weight = self._parzen_estimator_parameters.prior_weight
        self._multivariate = self._parzen_estimator_parameters.multivariate
        self._n_startup_trials = n_startup_trials
        self._n_ei_candidates = n_ei_candidates
        self._gamma = gamma
        self._weights = weights

        self._rng = np.random.RandomState(seed)
        self._random_sampler = RandomSampler(seed=seed)

        self._search_space = search_space

    def reseed_rng(self) -> None:

        self._rng.seed()
        self._random_sampler.reseed_rng()

    def infer_relative_search_space(
        self, study: Study, trial: FrozenTrial
    ) -> Dict[str, BaseDistribution]:
        return self._search_space

    def sample_relative(
        self, study: Study, trial: FrozenTrial, search_space: Dict[str, BaseDistribution]
    ) -> Dict[str, Any]:
        if search_space == {}:
            return {}

        param_names = list(search_space.keys())
        values, scores = _get_observation_pairs(study, param_names)

        # If the number of samples is insufficient, we run random trial.
        n = len(scores)
        if n < self._n_startup_trials:
            return {param_name: self._random_sampler.sample_independent(
                            study, trial, param_name, dist
                        ) for (param_name, dist) in search_space.items()}

        # We divide data into below and above.
        indices_below, indices_above = _split_observation_pairs(scores, self._gamma(n))
        # `None` items are intentionally converted to `nan` and then filtered out.
        # For `nan` conversion, the dtype must be float.
        config_values = {k: np.asarray(v, dtype=float) for k, v in values.items()}
        below = _build_observation_dict(config_values, indices_below)
        above = _build_observation_dict(config_values, indices_above)

        # We then sample by maximizing log likelihood ratio.
        mpe_below = _ParzenEstimator(below, search_space, self._parzen_estimator_parameters)
        mpe_above = _ParzenEstimator(above, search_space, self._parzen_estimator_parameters)
        samples_below = mpe_below.sample(self._rng, self._n_ei_candidates)
        log_likelihoods_below = mpe_below.log_pdf(samples_below)
        log_likelihoods_above = mpe_above.log_pdf(samples_below)
        ret = SimplifiedTPESampler._compare(samples_below, log_likelihoods_below, log_likelihoods_above)

        for param_name, dist in search_space.items():
            ret[param_name] = dist.to_external_repr(ret[param_name])

        return ret

    def sample_independent(
        self,
        study: Study,
        trial: FrozenTrial,
        param_name: str,
        param_distribution: BaseDistribution,
    ) -> Any:

        values, scores = _get_observation_pairs(study, [param_name])

        n = len(scores)

        if n < self._n_startup_trials:
            return self._random_sampler.sample_independent(
                study, trial, param_name, param_distribution
            )

        indices_below, indices_above = _split_observation_pairs(scores, self._gamma(n))
        # `None` items are intentionally converted to `nan` and then filtered out.
        # For `nan` conversion, the dtype must be float.
        config_values = {k: np.asarray(v, dtype=float) for k, v in values.items()}
        below = _build_observation_dict(config_values, indices_below)
        above = _build_observation_dict(config_values, indices_above)

        mpe_below = _ParzenEstimator(
            below, {param_name: param_distribution}, self._parzen_estimator_parameters
        )
        mpe_above = _ParzenEstimator(
            above, {param_name: param_distribution}, self._parzen_estimator_parameters
        )
        samples_below = mpe_below.sample(self._rng, self._n_ei_candidates)
        log_likelihoods_below = mpe_below.log_pdf(samples_below)
        log_likelihoods_above = mpe_above.log_pdf(samples_below)
        ret = SimplifiedTPESampler._compare(samples_below, log_likelihoods_below, log_likelihoods_above)

        return param_distribution.to_external_repr(ret[param_name])

    @classmethod
    def _compare(
        cls,
        samples: Dict[str, np.ndarray],
        log_l: np.ndarray,
        log_g: np.ndarray,
    ) -> Dict[str, Union[float, int]]:

        sample_size = next(iter(samples.values())).size
        if sample_size:
            score = log_l - log_g
            if sample_size != score.size:
                raise ValueError(
                    "The size of the 'samples' and that of the 'score' "
                    "should be same. "
                    "But (samples.size, score.size) = ({}, {})".format(sample_size, score.size)
                )
            best = np.argmax(score)
            return {k: v[best].item() for k, v in samples.items()}
        else:
            raise ValueError(
                "The size of 'samples' should be more than 0."
                "But samples.size = {}".format(sample_size)
            )

    def after_trial(
        self,
        study: Study,
        trial: FrozenTrial,
        state: TrialState,
        values: Optional[Sequence[float]],
    ) -> None:
        assert state in [TrialState.COMPLETE, TrialState.FAIL, TrialState.PRUNED]
        self._random_sampler.after_trial(study, trial, state, values)



def _get_observation_pairs(
    study: Study,
    param_names: List[str]
) -> Tuple[
    Dict[str, List[Optional[float]]],
    List[float]
]:
    """Get observation pairs from the study.

    This function collects observation pairs from the complete or pruned trials of the study.

    An observation pair fundamentally consists of a parameter value and an objective value.
    However, due to the pruning mechanism of Optuna, final objective values are not always
    available. Therefore, this function uses intermediate values in addition to the final
    ones, and reports the value with its step count as ``(-step, value)``.
    Consequently, the structure of the observation pair is as follows:
    ``(param_value, (-step, value))``.

    The second element of an observation pair is used to rank observations in
    ``_split_observation_pairs`` method (i.e., observations are sorted lexicographically by
    ``(-step, value)``).
    """

    signs = []
    for d in study.directions:
        if d == StudyDirection.MINIMIZE:
            signs.append(1)
        else:
            signs.append(-1)

    states = (TrialState.COMPLETE, TrialState.PRUNED)

    scores = []
    values: Dict[str, List[Optional[float]]] = {param_name: [] for param_name in param_names}
    for trial in study.get_trials(deepcopy=False, states=states):
        # We extract score from the trial.
        scores.append(sign * v for sign, v in zip(signs, trial.values))

        # We extract param_value from the trial.
        for param_name in param_names:
            param_value: Optional[float]
            distribution = trial.distributions[param_name]
            param_value = distribution.to_internal_repr(trial.params[param_name])
            values[param_name].append(param_value)
        
    return values, scores


def _split_observation_pairs(
    loss_vals: List[float],
    n_below: int,
) -> Tuple[np.ndarray, np.ndarray]:
    # When constrains is not None, trials are split into below and above
    # according to the following rules.
    # 1. Feasible trials are better than infeasible trials.
    # 2. Infeasible trials are sorted by sum of how much they violate each constraint.
    # 3. Feasible trials are sorted by loss_vals.
    
    loss_values = np.asarray(loss_vals)

    index_loss_ascending = np.argsort(loss_values)
    # `np.sort` is used to keep chronological order.
    indices_below = np.sort(index_loss_ascending[:n_below])
    indices_above = np.sort(index_loss_ascending[n_below:])
    return indices_below, indices_above


def _build_observation_dict(
    config_values: Dict[str, np.ndarray], indices: np.ndarray
) -> Dict[str, np.ndarray]:

    observation_dict = {}
    for param_name, param_val in config_values.items():
        param_values = param_val[indices]
        observation_dict[param_name] = param_values[~np.isnan(param_values)]

    return observation_dict

