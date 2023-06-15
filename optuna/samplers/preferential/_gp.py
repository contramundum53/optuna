
#%%
import copy

import numpy as np

# from matplotlib import pyplot as plt
# from scipy.stats import truncnorm, mvn
from scipy.stats import norm, multivariate_normal
from scipy.stats import multivariate_normal
from scipy import optimize
from tqdm import tqdm
import GPy
from scipy import special
from scipy.stats import qmc

maxfeval = 3 * 1e2
jitter = 1e-8

class PreferentialGP:
    """
    X_sort in \RR^{#duels \times 2 input_dim}: left side x is winner, right side x is looser

    If sampling = GIbbs, then we use Gibbs sampling
        YIFANG LI AND SUJIT K. GHOSH Efficient Sampling Methods for Truncated Multivariate Normal and Student-t Distributions Subject to Linear Inequality Constraints, Journal of Statistical Theory and Practice, 9:712–732, 2015
        Christian P Robert. Simulation of truncated normal variables. Statistics and computing, 5(2):121–125, 1995.
        Sébastien Da Veiga and Amandine Marrel. Gaussian process modeling with inequality constraints, 2012.
        Sébastien Da Veiga and Amandine Marrel. Gaussian process regression with linear inequality constraints. Reliability Engineering & System Safety, 195:106732, 2020.
    """


    def __init__(
        self,
        X,
        kernel,
        kernel_bounds,
        noise_std=1e-2,
        sample_size=1,
        burn_in=1000,
        thinning=1,
    ):
        self.input_dim = np.shape(X)[1] // 2
        self.num_duels = np.shape(X)[0]
        self.noise_std = noise_std
        self.kernel = kernel
        self.kernel_bounds = kernel_bounds

        self.sample_size = sample_size
        self.burn_in = burn_in
        self.thinning = thinning

        self.flatten_K_inv = None

        self.A = np.c_[-1 * np.eye(self.num_duels), np.eye(self.num_duels)]
        self.X = X
        self.flatten_X = np.r_[X[:, : self.input_dim], X[:, self.input_dim :]]

        self.initial_sample = None
        self.v_sample = None
        self.initial_points_sampler = qmc.Sobol(d=self.input_dim, seed=0)
        

    def inference(self, sample_size=None):
        self.flatten_K = self.kernel.K(self.flatten_X) + self.noise_std**2 * np.eye(
            2 * self.num_duels
        )
        self.K_v = self.A @ self.flatten_K @ self.A.T
        self.K_v_inv = np.linalg.inv(self.K_v)

        if sample_size is None:
            sample_size = self.sample_size

        # self.v_sample = orthants_MVN_iid_sampling(
        #     dim=self.num_duels,
        #     cov=self.K_v,
        #     sample_size=sample_size,
        # )

        # sampling from truncated multivariate normal
        self.v_sample = orthants_MVN_Gibbs_sampling(
            dim=self.num_duels,
            cov_inv=self.K_v_inv,
            burn_in=self.burn_in,
            thinning=self.thinning,
            sample_size=sample_size,
            initial_sample=self.initial_sample,
        )

    def add_data(self, X_win, X_loo):
        X_win = np.atleast_2d(X_win)
        X_loo = np.atleast_2d(X_loo)
        assert np.shape(X_win) == np.shape(
            X_loo
        ), "Shape of winner and looser in added data does not match"
        self.X = np.r_[self.X, np.c_[X_win, X_loo]]
        self.num_duels = self.num_duels + np.shape(X_win)[0]
        self.flatten_X = np.r_[self.X[:, : self.input_dim], self.X[:, self.input_dim :]]
        self.A = np.c_[-1 * np.eye(self.num_duels), np.eye(self.num_duels)]

        self.initial_sample = None

    def _covariance_X_v(self, X):
        X = np.atleast_2d(X)
        test_point_size = np.shape(X)[0]

        transform_matrix = np.r_[
            np.c_[np.eye(test_point_size), np.zeros((test_point_size, self.num_duels))],
            np.c_[np.zeros((2 * self.num_duels, test_point_size)), self.A.T],
        ]
        cov_X_flattenX = self.kernel.K(X, self.flatten_X)
        K_X_flattenX = np.c_[
            np.r_[
                self.kernel.K(X) + jitter * np.eye(test_point_size), cov_X_flattenX.T
            ],
            np.r_[cov_X_flattenX, self.flatten_K],
        ]

        return (
            transform_matrix.T[test_point_size:, :]
            @ K_X_flattenX
            @ transform_matrix[:, :test_point_size]
        )

    def one_sample_conditioned_predict(self, X, full_cov=False):
        K_X_v = self._covariance_X_v(X)

        tmp = K_X_v.T @ self.K_v_inv
        mean = tmp @ np.c_[self.v_sample[:, 0]]
        if full_cov:
            cov = self.kernel.K(X) - tmp @ K_X_v
            return mean, cov
        else:
            var = self.kernel.variance.values - np.einsum("ij,ji->i", tmp, K_X_v)
            return mean, var



############################################################################################
# util functions and classes
############################################################################################

from . import _tmvn_sampler
def orthants_MVN_iid_sampling(
    dim, cov, sample_size=1000,
):
    tmvn = _tmvn_sampler.TruncatedMVN(np.zeros(dim), np.copy(cov), np.zeros(dim), np.full(dim, np.inf))
    return tmvn.sample(sample_size)


def orthants_MVN_Gibbs_sampling(
    dim, cov_inv, burn_in=500, thinning=1, sample_size=1000, initial_sample=None
):
    if initial_sample is None:
        sample_chain = np.zeros((dim, 1))
    else:
        assert initial_sample.shape == (
            dim,
            1,
        ), "Shape of initial sample of Gibbs sampling is not (dim, 1)"
        sample_chain = initial_sample

    conditional_std = 1 / np.sqrt(np.diag(cov_inv))
    scaled_cov_inv = cov_inv / np.c_[np.diag(cov_inv)]
    sample_list = []
    for i in range((burn_in + thinning * (sample_size - 1)) * dim):
        j = i % dim
        conditional_mean = sample_chain[j] - scaled_cov_inv[j] @ sample_chain
        sample_chain[j] = (
            -1
            * one_side_trunc_norm_sampling(
                lower=conditional_mean[0] / conditional_std[j]
            )
            * conditional_std[j]
            + conditional_mean[0]
        )

        if ((i + 1) - burn_in * dim) % (
            dim * thinning
        ) == 0 and i + 1 - burn_in * dim >= 0:
            sample_list.append(sample_chain.copy())

    samples = np.hstack(sample_list)
    return samples


a_zero = 0.2570


def trunc_norm_sampling(lower=None, upper=None, mean=0, std=1):
    """
    See Sec.2.1 in YIFANG LI AND SUJIT K. GHOSH Efficient Sampling Methods for Truncated Multivariate Normal and Student-t Distributions Subject to Linear Inequality Constraints, Journal of Statistical Theory and Practice, 9:712–732, 2015
            Christian P Robert. Simulation of truncated normal variables. Statistics and computing, 5(2):121–125, 1995.
    """
    if lower is None and upper is None:
        return np.random.randn(1) * std + mean
    elif lower is None:
        upper = (upper - mean) / std
        return -1 * one_side_trunc_norm_sampling(lower=-upper) * std + mean
    elif upper is None:
        lower = (lower - mean) / std
        return one_side_trunc_norm_sampling(lower=lower) * std + mean
    elif lower <= 0 and 0 < upper:
        lower = (lower - mean) / std
        upper = (upper - mean) / std
        return (
            two_sided_trunc_norm_sampling_zero_containing(lower=lower, upper=upper)
            * std
            + mean
        )
    elif 0 <= lower:
        lower = (lower - mean) / std
        upper = (upper - mean) / std
        return (
            two_sided_trunc_norm_sampling_positive_lower(lower=lower, upper=upper) * std
            + mean
        )
    elif upper <= 0:
        lower = (lower - mean) / std
        upper = (upper - mean) / std
        return (
            -1
            * two_sided_trunc_norm_sampling_positive_lower(lower=-upper, upper=-lower)
            * std
            + mean
        )


def one_side_trunc_norm_sampling(lower=None):
    if lower > a_zero:
        alpha = (lower + np.sqrt(lower**2 + 4)) / 2.0
        while True:
            z = np.random.exponential(alpha) + lower
            rho_z = np.exp(-((z - alpha) ** 2) / 2.0)
            u = np.random.rand(1)
            if u <= rho_z:
                return z
    elif lower >= 0:
        while True:
            z = np.abs(np.random.randn(1))
            if lower <= z:
                return z
    else:
        while True:
            z = np.random.randn(1)
            if lower <= z:
                return z


def two_sided_trunc_norm_sampling_zero_containing(lower, upper):
    if upper <= lower * np.sqrt(2 * np.pi):
        M = 1.0 / np.sqrt(
            2 * np.pi
        )  # / (normcdf(upper) - normcdf(lower)) * (upper - lower)
        while True:
            z = np.random.rand(1) * (upper - lower) + lower
            u = np.random.rand(1)
            if u <= normpdf(z) / M:
                return z
    else:
        while True:
            z = np.random.randn(1)
            if lower <= z and z <= upper:
                return z


def two_sided_trunc_norm_sampling_positive_lower(lower, upper):
    if lower < a_zero:
        b_1_a = lower + np.sqrt(np.pi / 2.0) * np.exp(lower**2 / 2.0)
        if upper <= b_1_a:
            M = normpdf(
                lower
            )  # / (normcdf(upper) - normcdf(lower)) # * (upper - lower)
            while True:
                z = np.random.rand(1) * (upper - lower) + lower
                u = np.random.rand(1)
                if u <= normpdf(z) / M:
                    return z
        else:
            while True:
                z = np.abs(np.random.randn(1))
                if lower <= z and z <= upper:
                    return z
    else:
        tmp = np.sqrt(lower**2 + 4)
        b_2_a = lower + 2 / (lower + tmp) * np.exp(
            (lower**2 - lower * tmp) / 4.0 + 0.5
        )
        if upper <= b_2_a:
            M = normpdf(
                lower
            )  # / (normcdf(upper) - normcdf(lower)) # * (upper - lower)
            while True:
                z = np.random.rand(1) * (upper - lower) + lower
                u = np.random.rand(1)
                if u <= normpdf(z) / M:
                    return z
        else:
            alpha = (lower + np.sqrt(lower**2 + 4)) / 2.0
            while True:
                z = np.random.exponential(alpha) + lower
                if z <= upper:
                    rho_z = np.exp(-((z - alpha) ** 2) / 2.0)
                    u = np.random.rand(1)
                    if u <= rho_z:
                        return z


root_two = np.sqrt(2)

def normcdf(x):
    return 0.5 * (1 + special.erf(x / root_two))

def normpdf(x):
    pdf = np.zeros(np.shape(x))
    small_x_idx = np.abs(x) < 50
    pdf[small_x_idx] = np.exp(-x[small_x_idx] ** 2 / 2) / (np.sqrt(2 * np.pi))
    return pdf


from abc import ABCMeta, abstractmethod
import numpy as np
# import matplotlib.pyplot as plt
# from scipy.stats import norm
from scipy import special
from scipy import optimize
from scipy.stats import qmc
from scipy.special import owens_t


class PreferentialBO:


    def __init__(self, X, x_bounds, kernel, kernel_bounds, noise_std):
        self.input_dim = np.shape(X)[1] // 2
        self.bounds = x_bounds
        self.bounds_list = x_bounds.T.tolist()
        self.initial_points_sampler = qmc.Sobol(d=self.input_dim, seed=0)
        self.GPmodel = PreferentialGP(
                X=X, kernel=kernel, kernel_bounds=kernel_bounds, noise_std=noise_std
            )
        self.current_best = None
        self.noise_constant = 2.0 * self.GPmodel.noise_std**2
        self.iteration = 1
        self.beta_root = 2

    def next_input(self):
        if len(self.GPmodel.X) == 0:
            next_input1 = np.random.rand(self.input_dim) * (self.bounds[1] - self.bounds[0]) + self.bounds[0]
            next_input2 = np.random.rand(self.input_dim) * (self.bounds[1] - self.bounds[0]) + self.bounds[0]
            return np.atleast_2d(next_input1), np.atleast_2d(next_input2)

        x0s = (
            self.initial_points_sampler.random(n=20 * self.input_dim)
            * (self.bounds[1] - self.bounds[0])
            + self.bounds[0]
        )

        self.GPmodel.inference(sample_size=1)
        if self.current_best is None:
            mean_train, _ = self.GPmodel.one_sample_conditioned_predict(
                self.GPmodel.flatten_X
            )
            max_idx = np.argmax(mean_train)
            self.current_best = np.atleast_2d(self.GPmodel.flatten_X[max_idx])

        next_input1 = self.current_best
        f_min1 = 0

        x0s = np.r_[np.atleast_2d(self.current_best), x0s]
        self.x_1 = np.atleast_2d(next_input1)
        self.current_best_mean, _ = self.GPmodel.one_sample_conditioned_predict(
            next_input1
        )
        next_input2, f_min2 = minimize(self.EI, x0s, self.bounds_list)
    
        print("optimized acquisition function value:", -1 * f_min2)

        return np.atleast_2d(next_input1), np.atleast_2d(next_input2)

    def EI(self, X):
        X = np.atleast_2d(X)
        mean, var = self.GPmodel.one_sample_conditioned_predict(X)
        if var <= 0:
            print(mean, var)
            exit()
        std = np.sqrt(var)
        Z = (mean - self.current_best_mean) / std
        return -((Z * std) * normcdf(Z) + std * normpdf(Z)).ravel()

    def update(self, X_win, X_loo):
        self.current_best = X_win
        self.iteration += 1
        self.GPmodel.add_data(X_win, X_loo)


##################################################################################################
# util functions
##################################################################################################

def minimize(func, start_points, bounds, jac="2-point"):
    x = np.copy(start_points)
    func_values = list()

    for i in range(np.shape(x)[0]):
        res = optimize.minimize(
            func,
            x0=x[i],
            bounds=bounds,
            method="L-BFGS-B",
            options={"maxfun": 50},
            jac=jac,
        )
        func_values.append(res["fun"])
        x[i] = res["x"]

    min_index = np.argmin(func_values)
    return x[min_index], func_values[min_index]


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
from optuna.search_space import IntersectionSearchSpace

class PreferentialGPSampler(BasePreferentialSampler):
    def __init__(self) -> None:

        self._rng = np.random.RandomState()
        self._search_space = IntersectionSearchSpace()

        self._last_preference_idx = 0



    def reseed_rng(self) -> None:
        self._rng.seed()

    def infer_relative_search_space(
        self, study: PreferentialStudy, trial: FrozenTrial
    ) -> Dict[str, BaseDistribution]:
        search_space: Dict[str, BaseDistribution] = {}

        for name, distribution in self._search_space.calculate(study._internal_study).items():
            if distribution.single():
                continue
            search_space[name] = distribution

        return search_space


    def sample_relative(
        self, study: PreferentialStudy, trial: FrozenTrial, search_space: Dict[str, BaseDistribution]
    ) -> Dict[str, Any]:
        if search_space == {}:
            return {}
        kernel_bounds = np.array([0.1, 0.5])
        trans = _SearchSpaceTransform(
            search_space,
            transform_log=True, 
            transform_step=True, 
            transform_0_1=True
        )
        pref_bo = PreferentialBO(
            X=np.zeros((0, trans.bounds.shape[0] * 2), np.float64), 
            x_bounds=trans.bounds.T, 
            kernel=GPy.kern.RBF(
                input_dim=2,
                lengthscale=0.5 * (kernel_bounds[1] + kernel_bounds[0]),
                variance=1,
                ARD=True,
            ), 
            kernel_bounds=kernel_bounds,
            noise_std=0.01,
        )


        preferences = study.get_preferences()
        # new_preferences = preferences[self._last_preference_idx:]
        # self._last_preference_idx = len(preferences)
        for better, worse in preferences:
            better_params = trans.transform(better.params)
            worse_params = trans.transform(worse.params)
            pref_bo.update(np.atleast_2d(better_params), np.atleast_2d(worse_params))
        
        _, res = pref_bo.next_input()
        return trans.untransform(res[0, :])

    def sample_independent(
        self,
        study: PreferentialStudy,
        trial: FrozenTrial,
        param_name: str,
        param_distribution: distributions.BaseDistribution,
    ) -> Any:
        search_space = {param_name: param_distribution}
        trans = _SearchSpaceTransform(search_space)
        trans_params = self._rng.uniform(trans.bounds[:, 0], trans.bounds[:, 1])

        return trans.untransform(trans_params)[param_name]
    
    def ask_trials(
        self, 
        study: PreferentialStudy, 
        generate_new_trial: Callable[[dict[str, JSONSerializable]], FrozenTrial], 
    ) -> list[FrozenTrial]:
        
        all_trials = study.get_trials(deepcopy=False)
        preferences = study.get_preferences()
        
        undominated_trial_numbers = {t.number for t in all_trials if t.state == TrialState.COMPLETE} - {worse.number for better, worse in preferences}
        undominated_trials = [all_trials[i] for i in undominated_trial_numbers]

        n_new_samples = max(0, 2 - len(undominated_trials))
        new_trials = [generate_new_trial({}) for _ in range(n_new_samples)]

        return undominated_trials + new_trials
            
