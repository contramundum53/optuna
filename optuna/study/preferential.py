from __future__ import annotations

from typing import Any
from typing import Container
from typing import Dict
from typing import Iterable
from typing import Optional
from typing import Callable

from optuna import logging
from optuna import samplers
from optuna import storages
from optuna import trial as trial_module
from optuna.distributions import BaseDistribution
from optuna.trial import FrozenTrial
from optuna.trial import TrialState
from optuna._typing import JSONSerializable
from optuna.study.study import Study, create_study, load_study
from optuna.study._optimize import _run_trial


_logger = logging.get_logger(__name__)
_SYSTEM_ATTR_PREFERENCES_KEY = "preferential:preferences"

class PreferentialStudy:
    def __init__(
        self,
        internal_study: Study,
        sampler: "samplers.BasePreferentialSampler" | None,
    ) -> None:
        self._internal_study = internal_study
        self._sampler = sampler

    @property
    def trials(self) -> list[FrozenTrial]:
        return self.get_trials(deepcopy=True, states=None)

    def get_trials(
        self,
        deepcopy: bool = True,
        states: Container[TrialState] | None = None,
    ) -> list[FrozenTrial]:
        return self._internal_study.get_trials(deepcopy=deepcopy, states=states)

    def _get_trials(
        self,
        deepcopy: bool = True,
        states: Container[TrialState] | None = None,
        use_cache: bool = False,
    ) -> list[FrozenTrial]:
        return self._internal_study._get_trials(deepcopy=deepcopy, states=states, use_cache=use_cache)

    @property
    def user_attrs(self) -> dict[str, Any]:
        return self._internal_study.user_attrs

    def ask_trials(self, generation_func: Callable[[trial_module.Trial], None]) -> list[trial_module.FrozenTrial]:    

        def create_new_trial(system_attrs: dict[str, JSONSerializable]) -> trial_module.FrozenTrial:
            def objective(trial: trial_module.Trial) -> float:
                for key, value in system_attrs.items():
                    trial.storage.set_trial_system_attr(trial._trial_id, key, value)
                    trial._cached_frozen_trial.system_attrs[key] = value
                generation_func(trial)
                return 0.0  # Dummy value

            return _run_trial(self._internal_study, objective, catch=())
        
        return self._sampler.ask_trials(self, create_new_trial)

    def _tell_preference_pairs(self, preference_pairs: list[tuple[int, int]]) -> None:
        system_attrs = self._internal_study._storage.get_study_system_attrs(study_id=self._internal_study._study_id)
        old_preferences: list[list[int]] = system_attrs[_SYSTEM_ATTR_PREFERENCES_KEY] if _SYSTEM_ATTR_PREFERENCES_KEY in system_attrs else []
        self._internal_study._storage.set_study_system_attr(
            study_id=self._internal_study._study_id, 
            key=_SYSTEM_ATTR_PREFERENCES_KEY, 
            value=old_preferences + preference_pairs,
        )

    def tell_preference(self, preference_best_to_worst: 
        list[trial_module.FrozenTrial | list[trial_module.FrozenTrial]]) -> None:

        preference_pairs: list[tuple[int, int]] = []

        for i in range(len(preference_best_to_worst) - 1):
            better = preference_best_to_worst[i]
            worse = preference_best_to_worst[i+1]

            if not isinstance(better, list):
                better = [better]
            if not isinstance(worse, list):
                worse = [worse]

            preference_pairs += [
                (b.number, w.number) for b in better for w in worse
            ]

        self._tell_preference_pairs(preference_pairs)
    
    def get_preferences(self) -> list[tuple[trial_module.FrozenTrial, trial_module.FrozenTrial]]:
        trials = self.get_trials(deepcopy=False)
        system_attrs = self._internal_study._storage.get_study_system_attrs(study_id=self._internal_study._study_id)
        preferences_numbers = system_attrs[_SYSTEM_ATTR_PREFERENCES_KEY] if _SYSTEM_ATTR_PREFERENCES_KEY in system_attrs else []
        return [
            (trials[better], trials[worse]) for better, worse in preferences_numbers
        ]

    def set_user_attr(self, key: str, value: Any) -> None:
        self._internal_study.set_user_attr(key, value)

    def enqueue_trial(
        self,
        params: Dict[str, Any],
        user_attrs: Optional[Dict[str, Any]] = None,
        skip_if_exists: bool = False,
    ) -> None:
        self._internal_study.enqueue_trial(params, user_attrs, skip_if_exists)

    def add_trial(self, trial: FrozenTrial) -> None:
        self._internal_study.add_trial(trial)

    def add_trials(self, trials: Iterable[FrozenTrial]) -> None:
        self._internal_study.add_trials(trials)

def create_preferential_study(
    *,
    storage: str | storages.BaseStorage | None = None,
    sampler: "samplers.BasePreferentialSampler" | None = None,
    study_name: str | None = None,
    load_if_exists: bool = False,
) -> PreferentialStudy:
    return PreferentialStudy(
        create_study(
            storage=storage, 
            sampler=sampler._get_proxy_sampler() if sampler is not None else None,
            study_name=study_name, 
            load_if_exists=load_if_exists
        ),
        sampler,
    )


def load_preferential_study(
    *,
    study_name: str | None,
    storage: str | storages.BaseStorage | None,
    sampler: "samplers.BasePreferentialSampler" | None = None,
) -> PreferentialStudy:
    return PreferentialStudy(
        load_study(
            study_name=study_name, 
            storage=storage, 
            sampler=sampler._get_proxy_sampler(),
        ),
        sampler,
    )