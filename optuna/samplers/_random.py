from typing import Any
from typing import Dict
from typing import Optional

import numpy

from optuna import distributions
from optuna._transform import _SearchSpaceTransform
from optuna.distributions import BaseDistribution
from optuna.samplers import BaseSampler
from optuna.study import Study
from optuna.trial import FrozenTrial

from optuna.distributions import FloatDistribution
from optuna.distributions import IntDistribution
from optuna.distributions import CategoricalDistribution
from optuna.distributions import PermutationDistribution

class RandomSampler(BaseSampler):
    """Sampler using random sampling.

    This sampler is based on *independent sampling*.
    See also :class:`~optuna.samplers.BaseSampler` for more details of 'independent sampling'.

    Example:

        .. testcode::

            import optuna
            from optuna.samplers import RandomSampler


            def objective(trial):
                x = trial.suggest_float("x", -5, 5)
                return x**2


            study = optuna.create_study(sampler=RandomSampler())
            study.optimize(objective, n_trials=10)

    Args:
        seed: Seed for random number generator.
    """

    def __init__(self, seed: Optional[int] = None) -> None:
        self._rng = numpy.random.RandomState(seed)

    def reseed_rng(self) -> None:
        self._rng.seed()

    def infer_relative_search_space(
        self, study: Study, trial: FrozenTrial
    ) -> Dict[str, BaseDistribution]:
        return {}

    def sample_relative(
        self, study: Study, trial: FrozenTrial, search_space: Dict[str, BaseDistribution]
    ) -> Dict[str, Any]:
        return {}

    def sample_independent(
        self,
        study: Study,
        trial: FrozenTrial,
        param_name: str,
        param_distribution: distributions.BaseDistribution,
    ) -> Any:
        if isinstance(param_distribution, (FloatDistribution, IntDistribution, CategoricalDistribution)):
            search_space = {param_name: param_distribution}
            trans = _SearchSpaceTransform(search_space)
            trans_params = self._rng.uniform(trans.bounds[:, 0], trans.bounds[:, 1])
            return trans.untransform(trans_params)[param_name]
        elif isinstance(param_distribution, PermutationDistribution):
            return list(self._rng.permutation(param_distribution.n))
        else:
            assert False, f"Unknown distribution object {param_distribution} is passed"
