import numpy as np
import pytest
import torch as th
from sb3_contrib.common.maskable.distributions import (
    MaskableBernoulliDistribution,
    MaskableCategorical,
    MaskableCategoricalDistribution,
    MaskableMultiCategoricalDistribution,
)


class TestMaskableCategorical:
    def test_applying_mask(self):
        """
        Show that probs change as a result of masking
        """

        starting_probs = th.Tensor([[0.2, 0.2, 0.6], [1, 0, 0]])
        expected_probs = th.Tensor([[0, 0.25, 0.75], [0, 0.5, 0.5]])
        mask = np.array([[False, True, True], [False, True, True]])

        distribution = MaskableCategorical(probs=starting_probs)
        distribution.apply_masking(mask)
        assert th.allclose(distribution.probs, expected_probs)

    def test_modifying_mask(self):
        """
        Show that masks apply independently of each other
        """

        starting_probs = th.Tensor([[0.2, 0.2, 0.6], [1, 0, 0]])
        expected_probs = th.Tensor([[0.5, 0.5, 0], [0, 1, 0]])
        first_mask = np.array([[False, True, True], [False, True, True]])
        second_mask = np.array([[True, True, False], [False, True, False]])

        # pytorch converts probs to logits in a way that loses some precision and makes
        # 0 probability outcomes slightly non-zero.
        atol = 2e-07
        distribution = MaskableCategorical(probs=starting_probs)
        assert th.allclose(distribution.probs, starting_probs, atol=atol)

        target_distribution = MaskableCategorical(probs=expected_probs)

        distribution.apply_masking(first_mask)
        distribution.apply_masking(second_mask)

        assert th.allclose(distribution.probs, target_distribution.probs, atol=atol)

    def test_removing_mask(self):
        """
        Show that masking may be unapplied to recover original probs
        """

        starting_probs = th.Tensor([[0.2, 0.2, 0.6], [1, 0, 0]])
        mask = np.array([[False, True, True], [False, True, True]])

        distribution = MaskableCategorical(probs=starting_probs)
        target_distribution = MaskableCategorical(probs=starting_probs)
        distribution.apply_masking(mask)
        distribution.apply_masking(None)
        assert th.allclose(distribution.probs, target_distribution.probs)

    def test_masking_affects_entropy(self):
        # All outcomes equally likely
        NUM_DIMS = 3
        logits = th.Tensor([[0] * NUM_DIMS])
        dist = MaskableCategorical(logits=logits)

        # For each possible number of valid actions v, show that e^entropy == v
        for v in range(1, NUM_DIMS + 1):
            masks = [j < v for j in range(NUM_DIMS)]
            dist.apply_masking(masks)
            assert int(th.exp(dist.entropy())) == v


class TestMaskableCategoricalDistribution:
    def test_distribution_must_be_initialized(self):
        """
        Cannot use distribution before it has logits
        """

        dist = MaskableCategoricalDistribution(1)
        with pytest.raises(AssertionError):
            dist.log_prob(th.randn(1))

        with pytest.raises(AssertionError):
            dist.entropy()

        with pytest.raises(AssertionError):
            dist.sample()

        with pytest.raises(AssertionError):
            dist.mode()

        with pytest.raises(AssertionError):
            dist.apply_masking(None)

    def test_logits_must_align_with_dims(self):
        NUM_DIMS = 3
        dist = MaskableCategoricalDistribution(NUM_DIMS)

        # There should be one logit per dim, we're one short
        logits = th.randn(1, NUM_DIMS - 1)
        with pytest.raises(ValueError):
            dist.proba_distribution(logits)

        # That's better
        logits = th.randn(1, NUM_DIMS)
        dist.proba_distribution(logits)

        # Other numbers of dimensions are acceptable as long as they can be realigned
        logits = th.randn(NUM_DIMS)
        dist.proba_distribution(logits)
        logits = th.randn(1, NUM_DIMS, 1)
        dist.proba_distribution(logits)

    def test_dim_masking(self):
        NUM_DIMS = 2
        dist = MaskableCategoricalDistribution(NUM_DIMS)

        logits = th.Tensor([[0] * NUM_DIMS])
        dist.proba_distribution(logits)

        assert (dist.distribution.probs == 0.5).all()
        assert int(th.exp(dist.entropy())) == NUM_DIMS

        for i in range(NUM_DIMS):
            mask = np.array([False] * NUM_DIMS)
            mask[i] = True
            dist.apply_masking(mask)
            probs = dist.distribution.probs
            assert probs.sum() == 1
            assert probs[i] == 1
            assert int(th.exp(dist.entropy())) == 1

        dist.apply_masking(None)
        assert (dist.distribution.probs == 0.5).all()
        assert int(th.exp(dist.entropy())) == NUM_DIMS


class TestMaskableBernoulliDistribution:
    def test_distribution_must_be_initialized(self):
        """
        Cannot use distribution before it has logits
        """

        dist = MaskableBernoulliDistribution(1)

        with pytest.raises(AssertionError):
            dist.log_prob(th.randn(1))

        with pytest.raises(AssertionError):
            dist.entropy()

        with pytest.raises(AssertionError):
            dist.sample()

        with pytest.raises(AssertionError):
            dist.mode()

        with pytest.raises(AssertionError):
            dist.apply_masking(None)

    def test_logits_must_align_with_dims(self):
        NUM_DIMS = 3
        dist = MaskableBernoulliDistribution(NUM_DIMS)

        # There should be two logits per dim, we're one short
        logits = th.randn(1, 2 * NUM_DIMS - 1)
        with pytest.raises(RuntimeError):
            dist.proba_distribution(logits)

        # That's better
        logits = th.randn(1, 2 * NUM_DIMS)
        dist.proba_distribution(logits)

        # Other numbers of dimensions are acceptable as long as they can be realigned
        logits = th.randn(2 * NUM_DIMS)
        dist.proba_distribution(logits)
        logits = th.randn(1, 2 * NUM_DIMS, 1)
        dist.proba_distribution(logits)


    def test_dim_masking(self):
        NUM_DIMS = 1
        dist = MaskableBernoulliDistribution(NUM_DIMS)

        logits = th.Tensor([[0] * 2 * NUM_DIMS])
        dist.proba_distribution(logits)

        assert len(dist.distributions) == NUM_DIMS
        assert (dist.distributions[0].probs == 0.5).all()
        assert int(th.exp(dist.entropy())) == 2 * NUM_DIMS

        for i in range(NUM_DIMS):
            mask = np.array([False] * 2 * NUM_DIMS)
            mask[i] = True
            dist.apply_masking(mask)
            probs = dist.distributions[0].probs
            assert probs.sum() == 1
            assert probs[i] == 1
            assert int(th.exp(dist.entropy())) == 1

        dist.apply_masking(None)
        assert (dist.distributions[0].probs == 0.5).all()
        assert int(th.exp(dist.entropy())) == 2 * NUM_DIMS
