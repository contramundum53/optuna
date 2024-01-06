
import numpy as np
import numba
import math
import typing
from .._acqf import Acqf, eval_acqf_no_grad, eval_acqf_with_grad
from .._search_space import CATEGORICAL, LOG, sample_transformed_params
@numba.njit
def local_search(acqf: Acqf, x0: np.ndarray, tol: float) -> tuple[np.ndarray, float]:
    grad_scale = 1 / acqf.kernel_params.inv_sq_lengthscales
    param_types, bounds, steps = acqf.search_space
    
    def fun(x: np.ndarray) -> float:
        return eval_acqf_no_grad(acqf, x)
    def fun_with_grad(x: np.ndarray) -> tuple[float, np.ndarray]:
        return eval_acqf_with_grad(acqf, x)
    
    def continuous_armijo_line_search(
        fun: typing.Callable[[float], float], 
        f0: float,
        der0: float,
        max_x: float,
    ) -> tuple[float, float]:    
        c1 = 1e-4
        alpha = 0.5

        if abs(der0) < 1e-6:
            return (0, f0)
        
        x = max_x

        while True:
            if x < tol:
                return (0, f0)
            f = fun(x)
            if f < f0 + c1 * der0 * x:
                return (x, f)
            x *= alpha
        
    
    def discrete_armijo_line_search(
        fun: typing.Callable[[float], float], 
        x0: float,
        f0: float,
        param_type: int, 
        bounds: tuple[float, float], 
        step: float,
    ) -> tuple[float, float]:
        
        if param_type == CATEGORICAL:
            f_min = f0
            x_min = x0
            for x in range(int(bounds[1])):
                if x == x0:
                    continue
                f = fun(float(x))
                if f < f_min:
                    f_min = f
                    x_min = float(x)
            return x_min, f_min
        else:
            assert step > 0.0
            bounds2 = (bounds[0] - 0.5 * step, bounds[1] + 0.5 * step)
            if param_type == LOG:
                bounds2 = (math.log(bounds2[0]), math.log(bounds2[1]))
            i_ub: int = int((bounds[1] - bounds[0] + 0.5 * step) // step)

            
            def x_to_i(x: float) -> int:
                x = x * (bounds2[1] - bounds2[0]) + bounds2[0]
                if param_type == LOG:
                    x = math.exp(x)
                return int((x - bounds[0] + 0.5 * step) // step)
            def i_to_x(i: int) -> float:
                if i < 0:
                    return -np.inf
                if i > i_ub:
                    return np.inf
                x = bounds[0] + i * step
                if param_type == LOG:
                    x = math.log(x)
                ret = (x - bounds2[0]) / (bounds2[1] - bounds2[0])
                return ret
            i0 = x_to_i(x0)
            # print(x0, i0, i_ub)
            assert 0 <= i0 <= i_ub
            x1p = i_to_x(i0 + 1)
            x1n = i_to_x(i0 - 1)
            f1p = fun(x1p) if np.isfinite(x1p) else np.inf
            f1n = fun(x1n) if np.isfinite(x1n) else np.inf
            
            if f1p < f1n:
                sign_i, x1, f1 = 1, x1p, f1p
            else:
                sign_i, x1, f1 = -1, x1n, f1n
            if f1 >= f0:
                return (x0, f0)
            
            der0 = (f1 - f0) / (x1 - x0)

            j = i_ub - i0 if sign_i == 1 else i0
            assert 1 <= j

            c1 = 1e-4
            alpha = 0.5
            while True:
                if j == 1:
                    return (x1, f1)
                
                x = i_to_x(i0 + sign_i * j)
                f = fun(x)
                if f < f0 + c1 * der0 * (x - x0):
                    return (x, f)
                
                j = max(min((x_to_i(x0 + (x - x0) * alpha) - i0) * sign_i, j-1), 1)
    

    continuous_dims = np.where(steps == 0.)[0]
    non_continuous_dims = np.where(steps != 0.)[0]

    n_groups = (len(continuous_dims) > 0) + len(non_continuous_dims)
    x = x0.copy()

    patience = n_groups

    if len(continuous_dims) == 0:
        f_val = fun(x)
    while True:
        if len(continuous_dims) > 0:
            f_val, grad = fun_with_grad(x)
            grad[non_continuous_dims] = 0.0
            
            d = grad * grad_scale
            def f_cont(a: float) -> float:
                return -fun(x + a * d)

            a_max = np.inf
            for i in continuous_dims:
                if abs(d[i]) > 1e-6:
                    a_max = min(a_max, (1-x[i]) / d[i] if d[i] > 0 else -x[i] / d[i])
            

            a2, nf_val2 = continuous_armijo_line_search(f_cont, -f_val, -np.vdot(d, grad), a_max)
            if -nf_val2 > f_val:
                patience = n_groups
            else:
                patience -= 1
                if patience == 0:
                    return x, f_val  
            f_val = -nf_val2
            x += a2 * d

        for i in non_continuous_dims:
            def f_disc(y: float) -> float:
                x[i] = y
                return -fun(x)

            if param_types[i] != CATEGORICAL:
                assert x[i] <= 1
            y2, nf_val2 = discrete_armijo_line_search(f_disc, x[i], -f_val, param_types[i], bounds[i, :], steps[i])
            if param_types[i] != CATEGORICAL:
                assert y2 <= 1
            if -nf_val2 > f_val:
                patience = n_groups
            else:
                patience -= 1
                if patience == 0:
                    return x, f_val
            f_val = -nf_val2
            x[i] = y2

@numba.njit
def optimize_acqf_core(acqf: Acqf, x_samples: np.ndarray, n_local_search: int, tol: float) -> tuple[np.ndarray, float]:
    # Evaluate all values at initial samples
    f_vals = eval_acqf_no_grad(acqf, x_samples)
    max_i = np.argmax(f_vals)

    # Choose the start points of local search (sample from prob. exp(f_vals) without replacement)
    zs = -np.log(np.random.rand(x_samples.shape[0])) / np.maximum(1e-200, np.exp(f_vals - f_vals[max_i]))
    zs[max_i] = 0.0  # Always sample the best point
    idxs = np.argsort(zs)[:n_local_search]

    best_x = x_samples[idxs[0], :]
    best_f = f_vals[idxs[0]]
    
    for i in idxs:
        x, f = local_search(acqf, x_samples[i, :], tol=tol)
        if f > best_f:
            best_x[:] = x
            best_f = f
        
    return best_x, best_f

def optimize_acqf_mixed(acqf: Acqf, initial_xs: np.ndarray, n_additional_samples: int = 2048, n_local_search: int = 200, tol: float = 1e-4) -> tuple[np.ndarray, float]:
    new_xs = sample_transformed_params(n_additional_samples, acqf.search_space)
    return optimize_acqf_core(acqf=acqf, x_samples=np.vstack((initial_xs, new_xs)), n_local_search=n_local_search, tol=tol)
