from typing import Any
from typing import Dict
from typing import Optional
from typing import Callable
from optuna._typing import JSONSerializable

from optuna import distributions
from optuna._transform import _SearchSpaceTransform
from optuna.distributions import BaseDistribution
from optuna.samplers.preferential._base import BasePreferentialSampler
from optuna.samplers._random import RandomSampler
from optuna.study.preferential import PreferentialStudy
from optuna.trial import FrozenTrial
from optuna.trial import TrialState

class PreferentialRandomSampler(BasePreferentialSampler):
    def __init__(self, batch_num: int, seed: Optional[int] = None) -> None:
        assert batch_num >= 2
        self._batch_num = batch_num
        self._sampler = RandomSampler(seed)

    def reseed_rng(self) -> None:
        self._sampler.reseed_rng()

    def infer_relative_search_space(
        self, study: PreferentialStudy, trial: FrozenTrial
    ) -> Dict[str, BaseDistribution]:
        return self._sampler.infer_relative_search_space(study._internal_study, trial)

    def sample_relative(
        self, study: PreferentialStudy, trial: FrozenTrial, search_space: Dict[str, BaseDistribution]
    ) -> Dict[str, Any]:
        return self._sampler.sample_relative(study._internal_study, trial, search_space)

    def sample_independent(
        self,
        study: PreferentialStudy,
        trial: FrozenTrial,
        param_name: str,
        param_distribution: distributions.BaseDistribution,
    ) -> Any:
        return self._sampler.sample_independent(study._internal_study, trial, param_name, param_distribution)
    
    def ask_trials(
        self, 
        study: PreferentialStudy, 
        generate_new_trial: Callable[[dict[str, JSONSerializable]], FrozenTrial], 
    ) -> list[FrozenTrial]:
        
        all_trials = study.get_trials(deepcopy=False)
        preferences = study.get_preferences()
        
        undominated_trial_numbers = {t.number for t in all_trials if t.state == TrialState.COMPLETE} - {worse.number for better, worse in preferences}
        undominated_trials = [all_trials[i] for i in undominated_trial_numbers]

        n_new_samples = max(0, self._batch_num - len(undominated_trials))
        new_trials = [generate_new_trial({}) for _ in range(n_new_samples)]

        return undominated_trials + new_trials
            

        
        
