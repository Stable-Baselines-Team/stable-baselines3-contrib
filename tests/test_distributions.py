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
            assert int(dist.entropy().exp()) == v


class TestMaskableCategoricalDistribution:
    def test_distribution_must_be_initialized(self):
        """
        Cannot use distribution before it has logits
        """

        DIMS = 2
        dist = MaskableCategoricalDistribution(DIMS)
        with pytest.raises(AssertionError):
            dist.log_prob(th.randint(DIMS - 1, (1, 3)))

        with pytest.raises(AssertionError):
            dist.entropy()

        with pytest.raises(AssertionError):
            dist.sample()

        with pytest.raises(AssertionError):
            dist.mode()

        with pytest.raises(AssertionError):
            dist.apply_masking(None)

        # But now we can
        action_logits = th.randn(1, DIMS)
        dist.proba_distribution(action_logits)
        actions = th.randint(DIMS - 1, (3, 1))
        dist.log_prob(actions)
        dist.entropy()
        dist.sample()
        dist.mode()
        # Test api
        dist.actions_from_params(action_logits)
        dist.log_prob_from_params(action_logits)
        dist.apply_masking(None)

    def test_logits_must_align_with_dims(self):
        NUM_DIMS = 3
        dist = MaskableCategoricalDistribution(NUM_DIMS)

        # There should be one logit per dim, we're one short
        logits = th.randn(1, NUM_DIMS - 1)
        with pytest.raises(RuntimeError):
            dist.proba_distribution(logits)

        # That's better
        logits = th.randn(1, NUM_DIMS)
        dist.proba_distribution(logits)

        # Other numbers of dimensions are acceptable as long as they can be realigned
        logits = th.randn(NUM_DIMS)
        dist.proba_distribution(logits)
        logits = th.randn(3, NUM_DIMS, 3)
        dist.proba_distribution(logits)

    def test_dim_masking(self):
        NUM_DIMS = 2
        dist = MaskableCategoricalDistribution(NUM_DIMS)

        logits = th.Tensor([[0] * NUM_DIMS])
        dist.proba_distribution(logits)

        assert (dist.distribution.probs == 0.5).all()
        assert int(dist.entropy().exp()) == NUM_DIMS

        for i in range(NUM_DIMS):
            mask = np.array([False] * NUM_DIMS)
            mask[i] = True
            dist.apply_masking(mask)
            probs = dist.distribution.probs
            assert probs.sum() == 1
            assert probs[0][i] == 1
            assert int(dist.entropy().exp()) == 1

        dist.apply_masking(None)
        assert (dist.distribution.probs == 0.5).all()
        assert int(dist.entropy().exp()) == NUM_DIMS


class TestMaskableMultiCategoricalDistribution:
    def test_distribution_must_be_initialized(self):
        """
        Cannot use distribution before it has logits
        """

        DIMS_PER_CAT = 2
        NUM_CATS = 2
        dist = MaskableMultiCategoricalDistribution([DIMS_PER_CAT] * NUM_CATS)

        with pytest.raises(AssertionError):
            dist.log_prob(th.randint(DIMS_PER_CAT - 1, (3, NUM_CATS)))

        with pytest.raises(AssertionError):
            dist.entropy()

        with pytest.raises(AssertionError):
            dist.sample()

        with pytest.raises(AssertionError):
            dist.mode()

        with pytest.raises(AssertionError):
            dist.apply_masking(None)

        # But now we can
        action_logits = th.randn(1, DIMS_PER_CAT * NUM_CATS)
        dist.proba_distribution(action_logits)
        actions = th.randint(DIMS_PER_CAT - 1, (3, NUM_CATS))
        dist.log_prob(actions)
        dist.entropy()
        dist.sample()
        dist.mode()
        # Test api
        dist.actions_from_params(action_logits)
        dist.log_prob_from_params(action_logits)
        dist.apply_masking(None)

    def test_logits_must_align_with_dims(self):
        DIMS_PER_CAT = 3
        NUM_CATS = 2
        dist = MaskableMultiCategoricalDistribution([DIMS_PER_CAT] * NUM_CATS)

        # There should be one logit per dim, we're one short
        logits = th.randn(1, DIMS_PER_CAT * NUM_CATS - 1)
        with pytest.raises(RuntimeError):
            dist.proba_distribution(logits)

        # That's better
        logits = th.randn(1, DIMS_PER_CAT * NUM_CATS)
        dist.proba_distribution(logits)

        # Other numbers of dimensions are acceptable as long as they can be realigned
        logits = th.randn(DIMS_PER_CAT * NUM_CATS)
        dist.proba_distribution(logits)
        logits = th.randn(3, DIMS_PER_CAT * NUM_CATS, 3)
        dist.proba_distribution(logits)

    def test_dim_masking(self):
        DIMS_PER_CAT = 2
        NUM_CATS = 3
        dist = MaskableMultiCategoricalDistribution([DIMS_PER_CAT] * NUM_CATS)

        logits = th.Tensor([[0] * DIMS_PER_CAT * NUM_CATS])
        dist.proba_distribution(logits)

        assert len(dist.distributions) == NUM_CATS
        for i in range(NUM_CATS):
            assert (dist.distributions[i].probs == 0.5).all()
        assert int(dist.entropy().exp()) == DIMS_PER_CAT**NUM_CATS

        for i in range(DIMS_PER_CAT):
            mask = np.array([False] * DIMS_PER_CAT * NUM_CATS)
            for j in range(NUM_CATS):
                mask[j * DIMS_PER_CAT + i] = True

            dist.apply_masking(mask)
            for j in range(NUM_CATS):
                probs = dist.distributions[j].probs
                assert probs.sum() == 1
                assert probs[0][i] == 1

            assert int(dist.entropy().exp()) == 1

        dist.apply_masking(None)
        for i in range(NUM_CATS):
            assert (dist.distributions[i].probs == 0.5).all()
        assert int(dist.entropy().exp()) == DIMS_PER_CAT**NUM_CATS


class TestMaskableBernoulliDistribution:
    def test_distribution_must_be_initialized(self):
        """
        Cannot use distribution before it has logits
        """

        DIMS = 2
        dist = MaskableBernoulliDistribution(DIMS)

        with pytest.raises(AssertionError):
            dist.log_prob(th.randint(1, (2, DIMS)))

        with pytest.raises(AssertionError):
            dist.entropy()

        with pytest.raises(AssertionError):
            dist.sample()

        with pytest.raises(AssertionError):
            dist.mode()

        with pytest.raises(AssertionError):
            dist.apply_masking(None)

        # But now we can
        action_logits = th.randn(1, 2 * DIMS)
        dist.proba_distribution(action_logits)
        actions = th.randint(1, (2, DIMS))
        dist.log_prob(actions)
        dist.entropy()
        dist.sample()
        dist.mode()
        # Test api
        dist.actions_from_params(action_logits)
        dist.log_prob_from_params(action_logits)
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
        logits = th.randn(3, 2 * NUM_DIMS, 3)
        dist.proba_distribution(logits)

    def test_dim_masking(self):
        NUM_DIMS = 2
        BINARY_STATES = 2
        dist = MaskableBernoulliDistribution(NUM_DIMS)

        logits = th.Tensor([[0] * BINARY_STATES * NUM_DIMS])
        dist.proba_distribution(logits)

        assert len(dist.distributions) == NUM_DIMS
        for i in range(NUM_DIMS):
            assert (dist.distributions[i].probs == 0.5).all()
        assert int(dist.entropy().exp()) == BINARY_STATES * NUM_DIMS

        for i in range(BINARY_STATES):
            mask = np.array([False] * BINARY_STATES * NUM_DIMS)
            for j in range(NUM_DIMS):
                mask[j * BINARY_STATES + i] = True

            dist.apply_masking(mask)
            for j in range(NUM_DIMS):
                probs = dist.distributions[j].probs
                assert probs.sum() == 1
                assert probs[0][i] == 1

            assert int(dist.entropy().exp()) == 1

        dist.apply_masking(None)
        for i in range(NUM_DIMS):
            assert (dist.distributions[i].probs == 0.5).all()
        assert int(dist.entropy().exp()) == BINARY_STATES * NUM_DIMS
