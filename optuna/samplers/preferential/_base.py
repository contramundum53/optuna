from __future__ import annotations

import abc
from typing import Any
from typing import Dict
from typing import Sequence
from typing import Optional
from typing import Callable 
from optuna._typing import JSONSerializable

from optuna.distributions import BaseDistribution
from optuna.study.preferential import PreferentialStudy
from optuna.study import Study
from optuna.trial import FrozenTrial
from optuna.trial import TrialState

from optuna.samplers._base import BaseSampler

class BasePreferentialSampler(abc.ABC):
    class _ProxySampler(BaseSampler):
        def __init__(self, sampler: "BasePreferentialSampler") -> None:
            self._preferential_sampler = sampler

        def infer_relative_search_space(
            self, study: Study, trial: FrozenTrial
        ) -> Dict[str, BaseDistribution]:
            preferential_study = PreferentialStudy(study, self._preferential_sampler)
            return self._preferential_sampler.infer_relative_search_space(preferential_study, trial)

        def sample_relative(
            self, study: Study, trial: FrozenTrial, search_space: Dict[str, BaseDistribution]
        ) -> Dict[str, Any]:
            preferential_study = PreferentialStudy(study, self._preferential_sampler)
            return self._preferential_sampler.sample_relative(preferential_study, trial, search_space)

        def sample_independent(
            self,
            study: Study,
            trial: FrozenTrial,
            param_name: str,
            param_distribution: BaseDistribution,
        ) -> Any:
            preferential_study = PreferentialStudy(study, self._preferential_sampler)
            return self._preferential_sampler.sample_independent(preferential_study, trial, param_name, param_distribution)

        def after_trial(
            self,
            study: Study,
            trial: FrozenTrial,
            state: TrialState,
            values: Optional[Sequence[float]],
        ) -> None:
            preferential_study = PreferentialStudy(study, self._preferential_sampler)
            return self._preferential_sampler.after_trial(preferential_study, trial, state)

        def reseed_rng(self) -> None:
            return self._preferential_sampler.reseed_rng()
        
    def _get_proxy_sampler(self) -> _ProxySampler:
        return self._ProxySampler(self)
    
    def __str__(self) -> str:
        return self.__class__.__name__

    @abc.abstractmethod
    def infer_relative_search_space(
        self, study: PreferentialStudy, trial: FrozenTrial
    ) -> Dict[str, BaseDistribution]:
        raise NotImplementedError

    @abc.abstractmethod
    def sample_relative(
        self, study: PreferentialStudy, trial: FrozenTrial, search_space: Dict[str, BaseDistribution]
    ) -> Dict[str, Any]:
        raise NotImplementedError

    @abc.abstractmethod
    def sample_independent(
        self,
        study: PreferentialStudy,
        trial: FrozenTrial,
        param_name: str,
        param_distribution: BaseDistribution,
    ) -> Any:
        raise NotImplementedError

    def after_trial(
        self,
        study: PreferentialStudy,
        trial: FrozenTrial,
        state: TrialState,
    ) -> None:
        pass

    def reseed_rng(self) -> None:
        pass

    @abc.abstractmethod
    def ask_trials(
        self, 
        study: PreferentialStudy, 
        generate_new_trial: Callable[[dict[str, JSONSerializable]], FrozenTrial], 
    ) -> list[FrozenTrial]:
        raise NotImplementedError
        

