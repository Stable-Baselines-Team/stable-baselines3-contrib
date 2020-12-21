import numpy as np
import pytest
import torch as th

from sb3_contrib.common.utils import quantile_huber_loss


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
