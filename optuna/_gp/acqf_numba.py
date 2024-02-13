from __future__ import annotations
from optuna._gp import gp
from optuna._gp import search_space as gp_search_space
from optuna._gp.acqf import AcquisitionFunctionParams, AcquisitionFunctionType
import numpy as np
import math
from numba import njit

CATEGORICAL = int(gp_search_space.ScaleType.CATEGORICAL)
@njit()
def posterior_with_grad_from_cov_fx_fX(cov_Y_Y_inv: np.ndarray, cov_Y_Y_inv_Y: np.ndarray, cov_fx_fX: np.ndarray, cov_fx_fx: float) -> tuple[tuple[float, float], tuple[np.ndarray, np.ndarray]]:
    kalman_gain = cov_Y_Y_inv @ cov_fx_fX
    mean = np.vdot(cov_Y_Y_inv_Y, cov_fx_fX)
    var = cov_fx_fx - np.vdot(kalman_gain, cov_fx_fX)
    dvar_dcov_fx_fX = kalman_gain
    dvar_dcov_fx_fX *= -2
    return (mean, np.maximum(var, 0)), (cov_Y_Y_inv_Y, dvar_dcov_fx_fX)

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
def eval_logei_with_grad(kernel_params: gp.KernelParams, X: np.ndarray, is_categorical: np.ndarray, cov_Y_Y_inv: np.ndarray, cov_Y_Y_inv_Y: np.ndarray, max_Y: float, x: np.ndarray, stabilizing_noise: float) -> tuple[float, np.ndarray]:
    cov_fx_fX = np.empty(X.shape[0])
    dcov_fx_fX_dsqdist = np.empty(X.shape[0])

    for i in range(X.shape[0]):
        d2 = 0.0
        for k in range(x.shape[0]):
            if not is_categorical[k]:
                d2 += kernel_params.inverse_squared_lengthscales[k] * (x[k] - X[i, k]) ** 2
            else:
                d2 += 0.0 if x[k] == X[i, k] else kernel_params.inverse_squared_lengthscales[k]

        kval, dkval_dsqdist = matern52_kernel_with_grad_from_sqdist(d2)
        cov_fx_fX[i] = kval * kernel_params.kernel_scale
        dcov_fx_fX_dsqdist[i] = dkval_dsqdist * kernel_params.kernel_scale

    cov_fx_fx = kernel_params.kernel_scale

    (mean, var), (dmean_dcov_fx_fX, dvar_dcov_fx_fX) = posterior_with_grad_from_cov_fx_fX(cov_Y_Y_inv, cov_Y_Y_inv_Y, cov_fx_fX, cov_fx_fx)
    val, dval_dmean, dval_dvar = logei_with_grad(mean, var + stabilizing_noise, max_Y)

    dval_dcov_fx_fX = (dval_dmean * dmean_dcov_fx_fX + dval_dvar * dvar_dcov_fx_fX)
    dval_dsqdist = dval_dcov_fx_fX
    dval_dsqdist *= dcov_fx_fX_dsqdist

    dval_dx = np.zeros_like(x)
    for k in range(x.shape[0]):
        if not is_categorical[k]:
            for i in range(X.shape[0]):
                dval_dx[k] += 2 * dval_dsqdist[i] * (x[k] - X[i,k])
            dval_dx[k] *= kernel_params.inverse_squared_lengthscales[k]
    return val, dval_dx


@njit()
def eval_acqf_with_grad(acqf_params: AcquisitionFunctionParams, x: np.ndarray) -> tuple[float, np.ndarray]:
    if acqf_params.acqf_type != AcquisitionFunctionType.LOG_EI:
        raise NotImplementedError()
    
    if x.ndim == 1:
        return eval_logei_with_grad(
            kernel_params=acqf_params.kernel_params,
            X=acqf_params.X,
            is_categorical=acqf_params.search_space.scale_types == CATEGORICAL,
            cov_Y_Y_inv=acqf_params.cov_Y_Y_inv,
            cov_Y_Y_inv_Y=acqf_params.cov_Y_Y_inv_Y,
            max_Y=acqf_params.max_Y,
            x=x,
            stabilizing_noise=acqf_params.acqf_stabilizing_noise
        )
    else:
        raise NotImplementedError()

@njit()
def eval_acqf_no_grad(acqf_params: AcquisitionFunctionParams, x: np.ndarray) -> np.ndarray | float:
    if x.ndim == 1:
        return eval_acqf_with_grad(acqf_params, x)[0]
    elif x.ndim == 2:
        ret = np.empty(x.shape[0])
        for i in range(x.shape[0]):
            ret[i] = eval_acqf_with_grad(acqf_params, x[i])[0]
        return ret
    else:
        raise NotImplementedError()