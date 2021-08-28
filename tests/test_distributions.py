import numpy as np
import pytest
import torch as th

from sb3_contrib.common.distributions import MaskableBernoulliDistribution, MaskableCategorical


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


class TestMaskableBernoulliDistribution:
    def test_logits_must_align_with_dims(self):
        NUM_DIMS = 3
        dist = MaskableBernoulliDistribution(NUM_DIMS)

        # There should be two logits per dim, we're one short
        logits = th.randn(1, 5)
        with pytest.raises(RuntimeError):
            dist.proba_distribution(logits)

        # That's better
        logits = th.randn(1, 6)
        dist.proba_distribution(logits)

    def test_dim_masking(self):
        NUM_DIMS = 1
        dist = MaskableBernoulliDistribution(NUM_DIMS)

        logits = th.Tensor([[0, 0]])
        dist.proba_distribution(logits)

        assert len(dist.distributions) == NUM_DIMS
        assert (dist.distributions[0].probs == 0.5).all()

        mask1 = np.array([True, False])
        dist.apply_masking(mask1)
        probs = dist.distributions[0].probs[0]
        assert probs[0] == 1
        assert probs[1] == 0

        mask2 = np.array([False, True])
        dist.apply_masking(mask2)
        probs = dist.distributions[0].probs[0]
        assert probs[0] == 0
        assert probs[1] == 1

        dist.apply_masking(None)
        assert (dist.distributions[0].probs == 0.5).all()
