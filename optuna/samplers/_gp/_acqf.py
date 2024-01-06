
from . import _gp
from . import _search_space
import numpy as np
import math
from typing import NamedTuple
from numba import njit

@njit()
def posterior_with_grad_from_KxX(cov_Y_Y_inv: np.ndarray, cov_Y_Y_inv_Y: np.ndarray, KxX: np.ndarray, Kxx: float) -> tuple[tuple[float, float], tuple[np.ndarray, np.ndarray]]:
    kalman_gain = cov_Y_Y_inv @ KxX
    mean = np.vdot(cov_Y_Y_inv_Y, KxX)
    var = Kxx - np.vdot(kalman_gain, KxX)
    dvar_dKxX = kalman_gain
    dvar_dKxX *= -2
    return (mean, np.maximum(var, 0)), (cov_Y_Y_inv_Y, dvar_dKxX)

@njit(inline="always")
def standard_logei_with_grad(z: float) -> tuple[float, float]:
    # E_{x ~ N(0, 1)}[max(0, x+z)]
    
    if z > -25:
        cdf = 0.5 * math.erfc(-z * math.sqrt(0.5))
        pdf = math.exp(-0.5 * z ** 2) * (1.0 / math.sqrt(2 * math.pi))
        return (math.log(z * cdf + pdf), cdf / (z * cdf + pdf))
    else:
        def erfcx_large(x: float) -> float:
            # Valid approximation of erfcx for around x >= 15
            assert x >= 15
            s = 1.0
            inv_2xx = 1.0 / (2 * x * x)
            for i in range(8, 0, -1):
                s = 1-s * (2 * i - 1) * inv_2xx
            return s / (math.sqrt(math.pi) * x)

        r = math.sqrt(0.5 * math.pi) * erfcx_large(-z * math.sqrt(0.5))
        val = -0.5 * z * z + math.log((z * r + 1) * (1.0 / math.sqrt(2 * math.pi)))
        grad = r/(z * r + 1)
        return (val, grad)

@njit(inline="always")
def logei_with_grad(mean: float, var: float, f0: float) -> tuple[float, float, float]:
    sigma = math.sqrt(var)
    st_val, st_grad = standard_logei_with_grad((mean - f0) / sigma)
    val = 0.5 * np.log(var) + st_val
    grad_mean = st_grad / sigma  
    grad_var = -st_grad * (mean - f0) / (2 * sigma * var) + 0.5 / var
    return (val, grad_mean, grad_var)


@njit(inline="always")
def matern52_kernel_with_grad_from_sqdist(sqdist: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    sqrt5d = np.sqrt(5 * sqdist)
    exp_part = np.exp(-sqrt5d)
    val = exp_part * ((1/3) * sqrt5d * sqrt5d + sqrt5d + 1)
    grad = (-5/6) * (sqrt5d + 1) * exp_part
    return (val, grad)

@njit()
def eval_logei_with_grad(kernel_params: _gp.KernelParams, X: np.ndarray, is_categorical: np.ndarray, cov_Y_Y_inv: np.ndarray, cov_Y_Y_inv_Y: np.ndarray, max_Y: float, x: np.ndarray) -> float:
    KxX = np.empty(X.shape[0])
    dKxX_dsqdist = np.empty(X.shape[0])

    for i in range(X.shape[0]):
        d2 = 0.0
        for k in range(x.shape[0]):
            if not is_categorical[k]:
                d2 += kernel_params.inv_sq_lengthscales[k] * (x[k] - X[i, k]) ** 2
            else:
                d2 += 0.0 if x[k] == X[i, k] else kernel_params.inv_sq_lengthscales[k]

        kval, dkval_dsqdist = matern52_kernel_with_grad_from_sqdist(d2)
        KxX[i] = kval * kernel_params.kernel_scale
        dKxX_dsqdist[i] = dkval_dsqdist * kernel_params.kernel_scale

    Kxx = _gp.MATERN_KERNEL0 * kernel_params.kernel_scale

    (mean, var), (dmean_dKxX, dvar_dKxX) = posterior_with_grad_from_KxX(cov_Y_Y_inv, cov_Y_Y_inv_Y, KxX, Kxx)
    val, dval_dmean, dval_dvar = logei_with_grad(mean, var + kernel_params.noise, max_Y)

    dval_dKxX = (dval_dmean * dmean_dKxX + dval_dvar * dvar_dKxX)
    dval_dsqdist = dval_dKxX
    dval_dsqdist *= dKxX_dsqdist

    dval_dx = np.zeros_like(x)
    for k in range(x.shape[0]):
        if not is_categorical[k]:
            for i in range(X.shape[0]):
                dval_dx[k] += 2 * dval_dsqdist[i] * (x[k] - X[i,k])
            dval_dx[k] *= kernel_params.inv_sq_lengthscales[k]

    return val, dval_dx

class Acqf(NamedTuple):
    kernel_params: _gp.KernelParams
    X: np.ndarray
    search_space: _search_space.SearchSpace
    cov_Y_Y_inv: np.ndarray
    cov_Y_Y_inv_Y: np.ndarray
    max_Y: np.ndarray

@njit()
def kernel(is_categorical: np.ndarray, kernel_params: _gp.KernelParams, X1: np.ndarray, X2: np.ndarray) -> np.ndarray:
    ret = np.empty((X1.shape[0], X2.shape[0]))
    for i in range(X1.shape[0]):
        for j in range(X2.shape[0]):
            d2 = 0.0
            for k in range(X1.shape[1]):
                if not is_categorical[k]:
                    d2 += kernel_params.inv_sq_lengthscales[k] * (X1[i, k] - X2[j, k]) ** 2
                else:
                    d2 += 0.0 if X1[i, k] == X2[j, k] else kernel_params.inv_sq_lengthscales[k]
            kval, _ = matern52_kernel_with_grad_from_sqdist(d2)
            ret[i, j] = kval * kernel_params.kernel_scale
    return ret

@njit()
def create_acqf(kernel_params: _gp.KernelParams, search_space: _search_space.SearchSpace, X: np.ndarray, Y: np.ndarray) -> Acqf:
    K = kernel(search_space.param_type == _search_space.CATEGORICAL, kernel_params, X, X)
    cov_Y_Y_inv = np.linalg.inv(K + kernel_params.noise * np.eye(X.shape[0]))
    cov_Y_Y_inv_Y = cov_Y_Y_inv @ Y

    return Acqf(
        kernel_params=kernel_params,
        X=X,
        search_space=search_space,
        cov_Y_Y_inv=cov_Y_Y_inv,
        cov_Y_Y_inv_Y=cov_Y_Y_inv_Y,
        max_Y=np.max(Y),
    )

@njit()
def eval_acqf_with_grad(acqf: Acqf, x: np.ndarray) -> tuple[float, np.ndarray]:
    if x.ndim == 1:
        return eval_logei_with_grad(
            kernel_params=acqf.kernel_params,
            X=acqf.X,
            is_categorical=acqf.search_space.param_type == _search_space.CATEGORICAL,
            cov_Y_Y_inv=acqf.cov_Y_Y_inv,
            cov_Y_Y_inv_Y=acqf.cov_Y_Y_inv_Y,
            max_Y=acqf.max_Y,
            x=x,
        )
    else:
        raise NotImplementedError()

@njit()
def eval_acqf_no_grad(acqf: Acqf, x: np.ndarray) -> np.ndarray | float:
    if x.ndim == 1:
        return eval_acqf_with_grad(acqf, x)[0]
    elif x.ndim == 2:
        ret = np.empty(x.shape[0])
        for i in range(x.shape[0]):
            ret[i] = eval_acqf_with_grad(acqf, x[i])[0]
        return ret
