from __future__ import annotations

import math
import typing
from typing import Callable
from typing import TYPE_CHECKING
from typing import NamedTuple
import warnings

import numpy as np


if TYPE_CHECKING:
    import scipy.optimize as so
    import torch
else:
    from optuna._imports import _LazyImport

    so = _LazyImport("scipy.optimize")
    torch = _LazyImport("torch")


# This GP implementation uses the following notation:
# X[len(trials), len(params)]: observed parameter values.
# Y[len(trials)]: observed objective values.
# x[(batch_len,) len(params)]: parameter value to evaluate. Possibly batched.
# cov_fX_fX[len(trials), len(trials)]: kernel matrix of X = V[f(X)]
# cov_fx_fX[(batch_len,) len(trials)]: kernel matrix of x and X = Cov[f(x), f(X)]
# cov_fx_fx: kernel value (scalar) of x = V[f(x)].
#     Since we use a Matern 5/2 kernel, we assume this value to be a constant.
# cov_Y_Y_inv[len(trials), len(trials)]: inv of the covariance matrix of Y = (V[f(X) + noise])^-1
# cov_Y_Y_inv_Y[len(trials)]: cov_Y_Y_inv @ Y
# max_Y: maximum of Y (Note that we transform the objective values such that it is maximized.)
# d2: squared distance between two points


class Matern52Kernel(torch.autograd.Function):
    @staticmethod
    def forward(ctx: typing.Any, squared_distance: torch.Tensor) -> torch.Tensor:  # type: ignore
        sqrt5d = torch.sqrt(5 * squared_distance)
        exp_part = torch.exp(-sqrt5d)
        val = exp_part * ((5 / 3) * squared_distance + sqrt5d + 1)
        # Notice that the derivative is taken w.r.t. d^2, but not w.r.t. d.
        deriv = (-5 / 6) * (sqrt5d + 1) * exp_part
        ctx.save_for_backward(deriv)
        return val

    @staticmethod
    def backward(ctx: typing.Any, grad: torch.Tensor) -> torch.Tensor:  # type: ignore
        # Let x be squared_distance, f(x) be forward(ctx, x), and g(f) be a provided function,
        # then deriv := df/dx, grad := dg/df, and deriv * grad = df/dx * dg/df = dg/dx.
        (deriv,) = ctx.saved_tensors
        return deriv * grad


def matern52_kernel_from_squared_distance(squared_distance: torch.Tensor) -> torch.Tensor:
    # sqrt5d = sqrt(5 * squared_distance)
    # exp(sqrt5d) * (1/3 * sqrt5d ** 2 + sqrt5d + 1)
    #
    # We cannot let PyTorch differentiate the above expression because
    # the gradient runs into 0/0 at squared_distance=0.
    return Matern52Kernel.apply(squared_distance)  # type: ignore


class KernelParamsTensor(NamedTuple):
    # Kernel parameters to fit.
    inverse_squared_lengthscales: torch.Tensor  # [len(params)]
    kernel_scale: torch.Tensor  # Scalar
    noise_var: torch.Tensor  # Scalar


class KernelParams(NamedTuple):
    inverse_squared_lengthscales: np.ndarray
    kernel_scale: float
    noise_var: float


def kernel(
    is_categorical: torch.Tensor,  # [len(params)]
    kernel_params: KernelParamsTensor,
    X1: torch.Tensor,  # [...batch_shape, n_A, len(params)]
    X2: torch.Tensor,  # [...batch_shape, n_B, len(params)]
) -> torch.Tensor:  # [...batch_shape, n_A, n_B]
    # kernel(x1, x2) = kernel_scale * matern52_kernel_from_squared_distance(
    #                     d2(x1, x2) * inverse_squared_lengthscales)
    # d2(x1, x2) = sum_i d2_i(x1_i, x2_i)
    # d2_i(x1_i, x2_i) = (x1_i - x2_i) ** 2  # if x_i is continuous
    # d2_i(x1_i, x2_i) = 1 if x1_i != x2_i else 0  # if x_i is categorical

    d2 = (X1[..., :, None, :] - X2[..., None, :, :]) ** 2

    # Use the Hamming distance for categorical parameters.
    d2[..., is_categorical] = (d2[..., is_categorical] > 0.0).type(torch.float64)
    d2 = (d2 * kernel_params.inverse_squared_lengthscales).sum(dim=-1)
    return matern52_kernel_from_squared_distance(d2) * kernel_params.kernel_scale


def kernel_at_zero_distance(
    kernel_params: KernelParamsTensor,
) -> torch.Tensor:  # [...batch_shape, n_A, n_B]
    # kernel(x, x) = kernel_scale
    return kernel_params.kernel_scale


def posterior(
    kernel_params: KernelParamsTensor,
    X: torch.Tensor,  # [len(trials), len(params)]
    is_categorical: torch.Tensor,  # bool[len(params)]
    cov_Y_Y_inv: torch.Tensor,  # [len(trials), len(trials)]
    cov_Y_Y_inv_Y: torch.Tensor,  # [len(trials)]
    x: torch.Tensor,  # [(batch,) len(params)]
) -> tuple[torch.Tensor, torch.Tensor]:  # (mean: [(batch,)], var: [(batch,)])
    cov_fx_fX = kernel(is_categorical, kernel_params, x[..., None, :], X)[..., 0, :]
    cov_fx_fx = kernel_at_zero_distance(kernel_params)

    # mean = cov_fx_fX @ inv(cov_fX_fX + noise * I) @ Y
    # var = cov_fx_fx - cov_fx_fX @ inv(cov_fX_fX + noise * I) @ cov_fx_fX.T
    mean = cov_fx_fX @ cov_Y_Y_inv_Y  # [batch]
    var = cov_fx_fx - (cov_fx_fX * (cov_fx_fX @ cov_Y_Y_inv)).sum(dim=-1)  # [batch]
    # We need to clamp the variance to avoid negative values due to numerical errors.
    return (mean, torch.clamp(var, min=0.0))


def marginal_log_likelihood(
    X: torch.Tensor,  # [len(trials), len(params)]
    Y: torch.Tensor,  # [len(trials)]
    is_categorical: torch.Tensor,  # [len(params)]
    kernel_params: KernelParamsTensor,
) -> torch.Tensor:  # Scalar
    # -0.5 * log((2pi)^n |C|) - 0.5 * Y^T C^-1 Y, where C^-1 = cov_Y_Y_inv
    # We apply the cholesky decomposition to efficiently compute log(|C|) and C^-1.

    cov_fX_fX = kernel(is_categorical, kernel_params, X, X)

    cov_Y_Y_chol = torch.linalg.cholesky(
        cov_fX_fX + kernel_params.noise_var * torch.eye(X.shape[0], dtype=torch.float64)
    )
    # log |L| = 0.5 * log|L^T L| = 0.5 * log|C|
    logdet = 2 * torch.log(torch.diag(cov_Y_Y_chol)).sum()
    # cov_Y_Y_chol @ cov_Y_Y_chol_inv_Y = Y --> cov_Y_Y_chol_inv_Y = inv(cov_Y_Y_chol) @ Y
    cov_Y_Y_chol_inv_Y = torch.linalg.solve_triangular(cov_Y_Y_chol, Y[:, None], upper=False)[:, 0]
    return -0.5 * (
        logdet
        + X.shape[0] * math.log(2 * math.pi)
        # Y^T C^-1 Y = Y^T inv(L^T L) Y --> cov_Y_Y_chol_inv_Y @ cov_Y_Y_chol_inv_Y
        + (cov_Y_Y_chol_inv_Y @ cov_Y_Y_chol_inv_Y)
    )


def fit_kernel_params(
    X: np.ndarray,  # [len(trials), len(params)]
    Y: np.ndarray,  # [len(trials)]
    is_categorical: np.ndarray,  # [len(params)]
    log_prior: Callable[[KernelParamsTensor], torch.Tensor],
    minimum_noise: float,
    initial_kernel_params: KernelParams | None = None,
) -> KernelParams:
    n_params = X.shape[1]

    def loss_func(raw_params: np.ndarray) -> tuple[float, np.ndarray]:
        raw_params_tensor = torch.from_numpy(raw_params)
        raw_params_tensor.requires_grad_(True)
        params = KernelParamsTensor(
            inverse_squared_lengthscales=torch.exp(raw_params_tensor[:n_params]),
            kernel_scale=torch.exp(raw_params_tensor[n_params]),
            noise_var=torch.exp(raw_params_tensor[n_params + 1]) + minimum_noise,
        )
        loss = -marginal_log_likelihood(
            torch.from_numpy(X), torch.from_numpy(Y), torch.from_numpy(is_categorical), params
        ) - log_prior(params)
        loss.backward()  # type: ignore
        return loss.item(), raw_params_tensor.grad.detach().numpy()  # type: ignore

    default_initial_kernel_params = KernelParams(
        inverse_squared_lengthscales=np.ones(n_params),
        kernel_scale=1.0,
        noise_var=1.0,
    )
    if initial_kernel_params is None:
        initial_kernel_params = default_initial_kernel_params

    # jac=True means loss_func returns the gradient for gradient descent.
    def optimize(initial_kernel_params: KernelParams) -> np.ndarray:
        # We apply log transform to enforce the positivity of the kernel parameters.
        # Note that we cannot just use the constraint because of the numerical unstability
        # of the marginal log likelihood.
        # We also enforce the noise parameter to be greater than `minimum_noise` to avoid
        # pathological behavior of maximum likelihood estimation.
        initial_raw_params = np.concatenate(
            [
                np.log(initial_kernel_params.inverse_squared_lengthscales),
                [
                    np.log(initial_kernel_params.kernel_scale),
                    np.log(initial_kernel_params.noise_var - minimum_noise),
                ],
            ]
        )

        # loss_func may throw RuntimeError due to numerical errors.
        res = so.minimize(loss_func, initial_raw_params, jac=True)
        if not res.success:
            raise RuntimeError(f"Optimization failed: {res.message}")
        return res.x

    try:
        # First try optimizing the kernel parameters with the provided
        # initial_kernel_params, but if it fails, rerun the optimization
        # with the default initial_kernel_params.
        # This increases the robustness of the optimization.
        raw_params_opt = optimize(initial_kernel_params)
    except RuntimeError:
        try:
            raw_params_opt = optimize(default_initial_kernel_params)
        except RuntimeError as e:
            warnings.warn(
                f"The optimization of the kernel parameters failed. ({e})"
                "The kernel parameters are set to the initial values."
            )
            return default_initial_kernel_params

    return KernelParams(
        inverse_squared_lengthscales=np.exp(raw_params_opt[:n_params]),
        kernel_scale=np.exp(raw_params_opt[n_params]),
        noise_var=np.exp(raw_params_opt[n_params + 1]) + minimum_noise,
    )
