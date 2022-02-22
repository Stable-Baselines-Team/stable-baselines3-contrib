import numpy as np
import pytest
import torch as th
from stable_baselines3.common.utils import set_random_seed

from sb3_contrib import TRPO
from sb3_contrib.common.utils import conjugate_gradient_solver, flat_grad, quantile_huber_loss


def test_quantile_huber_loss():
    assert np.isclose(quantile_huber_loss(th.zeros(1, 10), th.ones(1, 10)), 2.5)
    assert np.isclose(quantile_huber_loss(th.zeros(1, 10), th.ones(1, 10), sum_over_quantiles=False), 0.25)

    with pytest.raises(ValueError):
        quantile_huber_loss(th.zeros(1, 4, 4), th.zeros(1, 4))
    with pytest.raises(ValueError):
        quantile_huber_loss(th.zeros(1, 4), th.zeros(1, 1, 4))
    with pytest.raises(ValueError):
        quantile_huber_loss(th.zeros(4, 4), th.zeros(3, 4))
    with pytest.raises(ValueError):
        quantile_huber_loss(th.zeros(4, 4, 4, 4), th.zeros(4, 4, 4, 4))


def test_cg():
    # Test that conjugate gradient can actually solve
    # Ax = b when the A^-1 is known
    set_random_seed(4)
    A = th.ones(3, 3)
    # Symmetric matrix
    A[0, 1] = 2
    A[1, 0] = 2
    x = th.ones(3) + th.rand(3)
    b = A @ x

    def matrix_vector_dot_func(vector):
        return A @ vector

    x_approx = conjugate_gradient_solver(matrix_vector_dot_func, b, max_iter=5, residual_tol=1e-10)
    assert th.allclose(x_approx, x)


def test_flat_grad():
    n_parameters = 12  # 3 * (2 *  2)
    x = th.nn.Parameter(th.ones(2, 2, requires_grad=True))
    y = (x**2).sum()
    flat_grad_out = flat_grad(y, [x, x, x])
    assert len(flat_grad_out.shape) == 1
    # dy/dx = 2
    assert th.allclose(flat_grad_out, th.ones(n_parameters) * 2)


def test_trpo_warnings():
    """Test that TRPO warns and errors correctly on
    problematic rollout buffer sizes"""

    # Only 1 step: advantage normalization will return NaN
    with pytest.raises(AssertionError):
        TRPO("MlpPolicy", "Pendulum-v1", n_steps=1)
    # One step not advantage normalization: ok
    TRPO("MlpPolicy", "Pendulum-v1", n_steps=1, normalize_advantage=False, batch_size=1)

    # Truncated mini-batch
    with pytest.warns(UserWarning):
        TRPO("MlpPolicy", "Pendulum-v1", n_steps=6, batch_size=8)
