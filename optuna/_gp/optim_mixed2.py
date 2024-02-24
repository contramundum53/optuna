#%%
from __future__ import annotations
import numpy as np
from optuna._gp.acqf import AcquisitionFunctionParams, eval_acqf_with_grad, eval_acqf_no_grad
from optuna._gp.search_space import ScaleType, sample_normalized_params, normalize_one_param
from typing import TYPE_CHECKING
from optuna.logging import get_logger
import math

if TYPE_CHECKING:
    import scipy.optimize as so
else:
    from optuna import _LazyImport
    so = _LazyImport("scipy.optimize")

_logger = get_logger(__name__)

def local_search(acqf_params: AcquisitionFunctionParams, initial_normalized_params: np.ndarray, tol: float, max_iter: int = 100) -> tuple[np.ndarray, float]:
    print(acqf_params, initial_normalized_params, tol, max_iter)
    scale_types, bounds, steps = acqf_params.search_space

    continuous_params = np.where(steps == 0.)[0]

    # This is a technique for speeding up optimization.
    # We use an isotropic kernel, so scaling the gradient will make
    # the hessian better-conditioned.
    continuous_param_scale = 1 / np.sqrt(acqf_params.kernel_params.inverse_squared_lengthscales[continuous_params])
    continuous_scaled_bounds = [(0, 1/s) for s in continuous_param_scale]

    noncontinuous_params = np.where(steps > 0)[0]
    noncontinuous_param_choices = [
        np.arange(bounds[i, 1]) if scale_types[i] == ScaleType.CATEGORICAL
        else normalize_one_param(
            param_value=np.arange(bounds[i, 0], bounds[i, 1] + 0.5 * steps[i], steps[i]),
            scale_type=ScaleType(scale_types[i]),
            bounds=(bounds[i, 0], bounds[i, 1]),
            step=steps[i],
        ) for i in noncontinuous_params
    ]
    noncontinuous_paramwise_xtol = [
        np.min(np.diff(choices), initial=np.inf) / 4 for choices in noncontinuous_param_choices
    ]


    normalized_params = initial_normalized_params.copy()
    fval = eval_acqf_no_grad(acqf_params, normalized_params)

    def optimize_continuous() -> bool:
        nonlocal normalized_params, fval

        old_continuous_params = normalized_params[continuous_params].copy()

        def fun_continuous_with_grad(x: np.ndarray) -> tuple[float, np.ndarray]:
            normalized_params[continuous_params] = x * continuous_param_scale
            (fval, grad) = eval_acqf_with_grad(acqf_params, normalized_params)
            # Flip sign because scipy minimizes functions.
            return (-fval, -grad[continuous_params] * continuous_param_scale)

        # print(old_continuous_params)
        # print("grad_err0", so.check_grad(lambda x: eval_acqf_with_grad(acqf_params, x)[0], lambda x: eval_acqf_with_grad(acqf_params, x)[1], old_continuous_params, epsilon=1e-6))
        # print("grad_err1", so.check_grad(lambda x: fun_continuous_with_grad(x)[0], lambda x: fun_continuous_with_grad(x)[1], old_continuous_params / continuous_param_scale, epsilon=1e-6))

        x_opt, fval_opt, info = so.fmin_l_bfgs_b(
            func=fun_continuous_with_grad,
            x0=normalized_params[continuous_params] / continuous_param_scale,
            bounds=continuous_scaled_bounds,
            pgtol=math.sqrt(tol),
            maxiter=200,
        )

        if info["warnflag"] == 2:
            _logger.warn(f"Optimization failed: {info['task']}")
            fval_opt = fun_continuous_with_grad(x_opt)[0]
        # else:
        #     assert fval_opt == fun_continuous_with_grad(x_opt)[0]
   
        if -fval_opt < fval:
            normalized_params[continuous_params] = old_continuous_params

            # if not np.isclose(eval_acqf_no_grad(acqf_params, normalized_params), fval):
            #     print("D")
            #     print(normalized_params)
            #     print(eval_acqf_no_grad(acqf_params, normalized_params), fval)
            #     assert False
            return False
        
        normalized_params[continuous_params] = x_opt * continuous_param_scale
        fval = -fval_opt

        # print(f"{info=}")

        # if not np.isclose(eval_acqf_no_grad(acqf_params, normalized_params), fval):
        #     print("E")
        #     print(normalized_params)
        #     print(eval_acqf_no_grad(acqf_params, normalized_params), fval)
        #     assert False

        return info["nit"] != 0  # True if parameters are updated.
    
    def optimize_exhaustive_search(param_idx: int, choices: np.ndarray) -> bool:
        nonlocal normalized_params, fval

        current_choice = normalized_params[param_idx]
        choices_except_current = choices[choices != normalized_params[param_idx]]

        all_params = np.repeat(normalized_params[None, :], len(choices_except_current), axis=0)
        all_params[:, param_idx] = choices_except_current
        fvals = eval_acqf_no_grad(acqf_params, all_params)
        best_idx = np.argmax(fvals)

        if fvals[best_idx] >= fval:
            fval = fvals[best_idx]
            normalized_params[param_idx] = choices_except_current[best_idx]

            # if not np.isclose(eval_acqf_no_grad(acqf_params, normalized_params), fval):
            #     print(f"{all_params=}")
            #     print(f"{eval_acqf_no_grad(acqf_params, all_params)=}")
            #     print(f"{best_idx=}")
            #     print("G")
            #     print(normalized_params)
            #     print(eval_acqf_no_grad(acqf_params, normalized_params), fval)
            #     assert np.all(all_params[best_idx] == normalized_params)
            #     assert np.allclose(eval_acqf_no_grad(acqf_params, all_params)[best_idx], eval_acqf_no_grad(acqf_params, normalized_params))
            #     assert False
            return True
        else:
            normalized_params[param_idx] = current_choice

            # if not np.isclose(eval_acqf_no_grad(acqf_params, normalized_params), fval):
            #     print("H")
            #     print(normalized_params)
            #     print(eval_acqf_no_grad(acqf_params, normalized_params), fval)
            #     assert False
            return False


    def optimize_discrete_line_search(param_idx: int, choices: np.ndarray, xtol: float) -> bool:
        nonlocal normalized_params, fval

        if len(choices) == 1:
            # Do not optimize anything when there's only one choice.
            return False
        
        def get_rounded_index(x: float) -> int:
            i = np.clip(np.searchsorted(choices, x), 1, len(choices) - 1)
            return i - 1 if abs(x - choices[i-1]) < abs(x - choices[i]) else i

        current_choice_i = get_rounded_index(normalized_params[param_idx])
        assert normalized_params[param_idx] == choices[current_choice_i]

        negval_cache = {current_choice_i: -fval}

        def inegfun_cached(i: int) -> float:
            nonlocal negval_cache
            # Function value at choices[i].
            cache_val = negval_cache.get(i) 
            if cache_val is not None:
                return cache_val
            normalized_params[param_idx] = choices[i]

            # Flip sign because scipy minimizes functions.
            negval = -eval_acqf_no_grad(acqf_params, normalized_params)
            negval_cache[i] = negval
            return negval

        def negfun_interpolated(x: float) -> float:
            if x < choices[0] or x > choices[-1]:
                return np.inf
            i1 = np.clip(np.searchsorted(choices, x), 1, len(choices) - 1)
            i0 = i1 - 1

            f0, f1 = inegfun_cached(i0), inegfun_cached(i1)

            w0 = (choices[i1] - x) / (choices[i1] - choices[i0])
            w1 = 1.0 - w0

            return w0 * f0 + w1 * f1
        

        lb_negval = inegfun_cached(0)
        ub_negval = inegfun_cached(len(choices) - 1)

        new_i0 = [0, current_choice_i, len(choices) - 1][np.argmin([lb_negval, -fval, ub_negval])]
        x_lb = choices[0]
        x_ub = choices[-1]
        x0 = choices[new_i0]
        EPS = 1e-12
        if new_i0 == 0:
            x_lb -= EPS
        elif new_i0 == len(choices) - 1:
            x_ub += EPS
        print(0, new_i0, len(choices) - 1)
        print(x_lb, x0, x_ub)
        res = so.minimize_scalar(negfun_interpolated, bracket=(x_lb, x0, x_ub), method="brent", tol=xtol)
        i_star = get_rounded_index(res.x)
        fval_new = -inegfun_cached(i_star)

        normalized_params[param_idx] = choices[i_star]
        fval = fval_new

        # if not np.isclose(eval_acqf_no_grad(acqf_params, normalized_params), fval):
        #     print("A")
        #     print(normalized_params)
        #     print(eval_acqf_no_grad(acqf_params, normalized_params), fval)
        #     assert False

        return i_star != current_choice_i
        
    
    # If the number of possible parameter values is small, we just perform an exhaustive search.
    # This is faster and better than the line search.
    MAX_INT_EXHAUSTIVE_SEARCH_PARAMS = 16

    last_changed_param = -1
    # if not np.isclose(eval_acqf_no_grad(acqf_params, normalized_params), fval):
    #     print("F")
    #     print(normalized_params)
    #     print(eval_acqf_no_grad(acqf_params, normalized_params), fval)
    #     assert False
    for _ in range(max_iter):
        if len(continuous_params) > 0:
            if last_changed_param == continuous_params[0]:
                return (normalized_params, fval)
            # if not np.isclose(eval_acqf_no_grad(acqf_params, normalized_params), fval):
            #     print("C")
            #     print(normalized_params)
            #     print(eval_acqf_no_grad(acqf_params, normalized_params), fval)
            #     assert False
            changed = optimize_continuous()
            if changed:
                last_changed_param = continuous_params[0]

        for i, choices, xtol in zip(noncontinuous_params, noncontinuous_param_choices, noncontinuous_paramwise_xtol):
            if last_changed_param == i:
                return (normalized_params, fval)
            
            # if not np.isclose(eval_acqf_no_grad(acqf_params, normalized_params), fval):
            #     print("B")
            #     print(normalized_params)
            #     print(eval_acqf_no_grad(acqf_params, normalized_params), fval)
            #     assert False

            if scale_types[i] == ScaleType.CATEGORICAL or len(choices) <= MAX_INT_EXHAUSTIVE_SEARCH_PARAMS:
                changed = optimize_exhaustive_search(i, choices)
            else:
                changed = optimize_discrete_line_search(i, choices, xtol)

            # if not np.isclose(eval_acqf_no_grad(acqf_params, normalized_params), fval):
            #     print("B'")
            #     print(normalized_params)
            #     print(eval_acqf_no_grad(acqf_params, normalized_params), fval)
            #     assert False
            if changed:
                last_changed_param = i
        
        if last_changed_param == -1:
            return (normalized_params, fval)
        
    _logger.warn("optim_mixed: Local search did not converge.")
    return (normalized_params, fval)


def optimize_acqf_mixed(acqf_params: AcquisitionFunctionParams, initial_xs: np.ndarray, n_additional_samples: int = 2048, n_local_search: int = 10, tol: float = 1e-4, seed: int | None = None) -> tuple[np.ndarray, float]:
    new_xs = sample_normalized_params(n_additional_samples, acqf_params.search_space, seed=seed)
    
    # Evaluate all values at initial samples
    f_vals = eval_acqf_no_grad(acqf_params, new_xs)
    assert isinstance(f_vals, np.ndarray)

    max_i = np.argmax(f_vals)

    # Choose the start points of local search (sample from prob. exp(f_vals) without replacement)
    np.random.seed(seed + 1)
    zs = -np.log(np.random.rand(new_xs.shape[0])) / np.maximum(1e-200, np.exp(f_vals - f_vals[max_i]))
    zs[max_i] = 0.0  # Always sample the best point
    idxs = np.argsort(zs)[:max(0, n_local_search - len(initial_xs))]

    best_x = new_xs[idxs[0], :]
    best_f = f_vals[idxs[0]]

    for x_guess in np.vstack([new_xs[idxs, :], initial_xs]):
        x, f = local_search(acqf_params, x_guess, tol=tol)
        if f > best_f:
            best_x = x
            best_f = f

    return best_x, best_f

# from optuna._gp.acqf import AcquisitionFunctionParams, AcquisitionFunctionType
# from optuna._gp.gp import KernelParams
# from optuna._gp.search_space import SearchSpace
# from numpy import array
# local_search(
#     AcquisitionFunctionParams(
#         acqf_type=AcquisitionFunctionType.LOG_EI, kernel_params=KernelParams(inverse_squared_lengthscales=array([0.46993232, 0.79485394]), kernel_scale=1.8490197119030316, noise_var=1.001154530229285e-06), X=array([[0.63230317, 0.95454545],
#        [0.        , 0.04545455],
#        [0.59178427, 0.04545455],
#        [0.        , 0.31818182],
#        [0.        , 0.04545455],
#        [0.10314193, 0.04545455],
#        [0.        , 0.04545455]]), search_space=SearchSpace(scale_types=array([0, 0]), bounds=array([[ 0., 10.],
#        [ 0., 10.]]), steps=array([0., 1.])), cov_Y_Y_inv=array([[ 1.57791990e+00,  5.33801371e-01, -1.04877463e+00,
#         -2.61116357e+00,  5.33801371e-01,  9.13712676e-01,
#          5.33801371e-01],
#        [ 5.33801371e-01,  6.65924804e+05,  1.46379989e+01,
#         -3.27429696e+00, -3.32921997e+05, -9.10790639e+01,
#         -3.32921997e+05],
#        [-1.04877463e+00,  1.46379989e+01,  1.25773185e+01,
#          1.29604128e+00,  1.46379989e+01, -5.58572691e+01,
#          1.46379989e+01],
#        [-2.61116357e+00, -3.27429696e+00,  1.29604128e+00,
#          1.02884594e+01, -3.27429696e+00,  3.65789501e-01,
#         -3.27429696e+00],
#        [ 5.33801371e-01, -3.32921997e+05,  1.46379989e+01,
#         -3.27429696e+00,  6.65924804e+05, -9.10790639e+01,
#         -3.32921998e+05],
#        [ 9.13712676e-01, -9.10790639e+01, -5.58572691e+01,
#          3.65789501e-01, -9.10790639e+01,  3.22891673e+02,
#         -9.10790639e+01],
#        [ 5.33801371e-01, -3.32921997e+05,  1.46379989e+01,
#         -3.27429696e+00, -3.32921998e+05, -9.10790639e+01,
#          6.65924804e+05]]), cov_Y_Y_inv_Y=array([-2.00054466,  2.63487175, -0.05068585,  0.34341881,  2.63487175,
#        -6.70368296,  2.63487175]), max_Y=0.6805177183431633, beta=None, acqf_stabilizing_noise=1e-12),
#         array([0.02696953,0.95454545]),
#         0.0001,
#         100
# )
# %%
