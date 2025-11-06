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

    def test_numerical_stability_with_masking(self):
        """
        Test that masking does not cause numerical precision issues.
        This test is related to issue #81 and PR #302.

        The bug occurred when using -1e8 as the masked logit value, which could cause
        numerical precision issues in rare cases: the probabilities computed via
        logits_to_probs would not sum exactly to 1.0 within PyTorch's tolerance,
        causing a ValueError during Categorical distribution initialization when
        validate_args=True.

        With the fix (using -inf and storing pristine logits), this should not occur.

        Note: The original bug was intermittent and difficult to reproduce
        deterministically. This test verifies the correct behavior of the fix rather
        than attempting to trigger the original bug.
        """
        # Use logits with various ranges to test numerical stability
        test_cases = [
            # Small logits
            th.tensor([[0.1, 0.2, 0.3, 0.4, 0.5]]),
            # Mixed positive/negative
            th.tensor([[-1.0, 2.0, -0.5, 1.5, 0.0]]),
            # Large logits (more susceptible to precision issues)
            th.tensor([[10.0, -5.0, 3.0, -2.0, 0.5]]),
            # Very large batch similar to bug report
            th.randn(64, 400) * 2.0,
        ]

        for logits in test_cases:
            # Test with validation enabled (validate_args=True)
            # This is where the bug would manifest in the old code
            dist = MaskableCategorical(logits=logits, validate_args=True)

            # Apply various masks - this triggers re-initialization of the distribution
            # which is where the numerical precision issue would occur
            num_actions = logits.shape[-1]
            batch_size = logits.shape[0]

            # Test with different mask patterns
            masks = [
                # Mask out every other action
                th.tensor([[i % 2 == 0 for i in range(num_actions)]] * batch_size),
                # Mask out first half
                th.tensor([[i < num_actions // 2 for i in range(num_actions)]] * batch_size),
                # Random mask
                th.rand(batch_size, num_actions) > 0.3,
            ]

            for mask in masks:
                # Ensure at least one action is valid per batch
                for i in range(batch_size):
                    if not mask[i].any():
                        mask[i, 0] = True

                # This should not raise a ValueError about Simplex constraint
                dist.apply_masking(mask)

                # Verify that probs are valid (sum to 1.0)
                prob_sums = dist.probs.sum(dim=-1)
                assert th.allclose(prob_sums, th.ones_like(prob_sums), atol=1e-6), f"Probs don't sum to 1: {prob_sums}"

                # Verify entropy can be computed without NaN/inf issues
                entropy = dist.entropy()
                assert th.isfinite(entropy).all(), f"Entropy not finite: {entropy}"

                # Verify masked actions have very low or zero probability
                # With -inf masking (PR #302 fix), they should be exactly 0
                # With -1e8 masking (old code), they would be very small but non-zero
                masked_actions = ~mask
                if masked_actions.any():
                    masked_probs = dist.probs[masked_actions]
                    # After PR #302, masked probabilities should be exactly or very close to 0
                    assert th.allclose(
                        masked_probs, th.zeros_like(masked_probs), atol=1e-7
                    ), f"Masked probs not near zero: {masked_probs[:10]}"

            # Test with None mask (removing masking)
            dist.apply_masking(None)
            prob_sums = dist.probs.sum(dim=-1)
            assert th.allclose(prob_sums, th.ones_like(prob_sums), atol=1e-6)

    def test_entropy_with_all_but_one_masked(self):
        """
        Test entropy calculation when all but one action is masked.
        This is an edge case that should result in zero entropy (no uncertainty).
        Related to issue #81 and PR #302.
        """
        NUM_DIMS = 5
        logits = th.randn(10, NUM_DIMS)  # Random logits for batch of 10

        dist = MaskableCategorical(logits=logits, validate_args=True)

        # Mask all but one action (different valid action for each batch element)
        for i in range(NUM_DIMS):
            mask = th.zeros(10, NUM_DIMS, dtype=th.bool)
            mask[:, i] = True  # Only action i is valid

            dist.apply_masking(mask)

            # With only one valid action, entropy should be 0 (or very close to 0)
            entropy = dist.entropy()
            assert th.allclose(entropy, th.zeros_like(entropy), atol=1e-5)

            # The valid action should have probability 1.0
            assert th.allclose(dist.probs[:, i], th.ones(10), atol=1e-5)

            # All other actions should have probability 0.0
            for j in range(NUM_DIMS):
                if j != i:
                    assert th.allclose(dist.probs[:, j], th.zeros(10), atol=1e-5)

    def test_repeated_masking_stability(self):
        """
        Test that repeatedly applying different masks maintains numerical stability.
        This test verifies the fix from PR #302 where pristine logits are stored
        and used for each masking operation, avoiding accumulated numerical errors.
        Related to issue #81 and PR #302.
        """
        # Start with some logits
        original_logits = th.tensor([[1.0, 2.0, 3.0, 4.0, 5.0]])
        dist = MaskableCategorical(logits=original_logits.clone(), validate_args=True)

        # Apply a series of different masks
        masks = [
            th.tensor([[True, True, False, False, False]]),
            th.tensor([[False, False, True, True, True]]),
            th.tensor([[True, False, True, False, True]]),
            th.tensor([[False, True, False, True, False]]),
            th.tensor([[True, True, True, True, True]]),  # All valid
        ]

        for mask in masks:
            dist.apply_masking(mask)

            # Verify probabilities are valid
            prob_sum = dist.probs.sum(dim=-1)
            assert th.allclose(prob_sum, th.ones_like(prob_sum), atol=1e-6), f"Probs sum: {prob_sum}, expected 1.0"

            # Verify masked actions have 0 probability
            masked_out = ~mask
            masked_probs = dist.probs[masked_out]
            if masked_probs.numel() > 0:
                assert th.allclose(masked_probs, th.zeros_like(masked_probs), atol=1e-7), f"Masked probs: {masked_probs}"

            # Verify entropy is finite and non-negative
            entropy = dist.entropy()
            assert th.isfinite(entropy).all(), f"Entropy contains inf/nan: {entropy}"
            assert (entropy >= 0).all(), f"Entropy is negative: {entropy}"

        # After all masks, remove masking and verify we get consistent results
        dist.apply_masking(None)
        prob_sum = dist.probs.sum(dim=-1)
        assert th.allclose(prob_sum, th.ones_like(prob_sum), atol=1e-6)

    def test_masked_actions_have_zero_probability(self):
        """
        Test that masked actions have exactly zero probability with proper masking.

        This test verifies that masked actions get zero probability, which is important
        for the fix in PR #302. While both -1e8 and -inf produce zero probabilities
        after softmax due to underflow, using -inf is mathematically more correct
        and avoids potential numerical issues in edge cases.

        Related to issue #81 and PR #302.
        """
        # Test with various logit scales
        test_logits = [
            th.tensor([[0.0, 1.0, 2.0, 3.0]]),  # Small scale
            th.tensor([[10.0, 20.0, 30.0, 40.0]]),  # Large scale
            th.randn(5, 10) * 5.0,  # Random batch
        ]

        for logits in test_logits:
            dist = MaskableCategorical(logits=logits, validate_args=True)

            # Create a mask that masks out alternating actions
            mask = th.zeros_like(logits, dtype=th.bool)
            mask[:, ::2] = True  # Keep even indices, mask odd indices

            dist.apply_masking(mask)

            # Check that masked actions have exactly zero probability
            masked_indices = ~mask
            if masked_indices.any():
                masked_probs = dist.probs[masked_indices]
                # Both old (-1e8) and new (-inf) implementations should produce 0 here
                # due to softmax underflow, but -inf is more robust
                assert th.allclose(
                    masked_probs, th.zeros_like(masked_probs), atol=1e-10
                ), f"Masked actions should have ~0 probability, got: {masked_probs[:10]}"

            # Verify unmasked actions have non-zero probabilities
            unmasked_probs = dist.probs[mask]
            assert th.all(unmasked_probs > 0.0), "Unmasked actions should have positive probability"

            # Verify probabilities sum to 1
            prob_sums = dist.probs.sum(dim=-1)
            assert th.allclose(prob_sums, th.ones_like(prob_sums), atol=1e-6)

    def test_entropy_numerical_stability_with_masking(self):
        """
        Test entropy calculation numerical stability with masked actions.

        This specifically tests the improved entropy calculation from PR #302.
        The old entropy calculation could have issues with -1e8 logits, while
        the new calculation properly handles -inf values.

        Related to issue #81 and PR #302.
        """
        # Test with various scenarios including edge cases
        test_cases = [
            # All but one action masked (entropy should be ~0)
            (th.tensor([[1.0, 2.0, 3.0, 4.0]]), th.tensor([[True, False, False, False]])),
            # Half actions masked
            (th.tensor([[1.0, 2.0, 3.0, 4.0]]), th.tensor([[True, True, False, False]])),
            # Large batch with various masks
            (th.randn(32, 10), th.rand(32, 10) > 0.3),
        ]

        for logits, mask in test_cases:
            # Ensure at least one action is valid per batch
            for i in range(mask.shape[0]):
                if not mask[i].any():
                    mask[i, 0] = True

            dist = MaskableCategorical(logits=logits, validate_args=True)
            dist.apply_masking(mask)

            # Compute entropy - should not produce NaN or inf
            entropy = dist.entropy()
            assert th.isfinite(entropy).all(), f"Entropy should be finite, got: {entropy}"
            assert (entropy >= 0).all(), f"Entropy should be non-negative, got: {entropy}"

            # For single valid action, entropy should be close to 0
            single_action_mask = mask.sum(dim=-1) == 1
            if single_action_mask.any():
                single_action_entropy = entropy[single_action_mask]
                assert th.allclose(
                    single_action_entropy, th.zeros_like(single_action_entropy), atol=1e-4
                ), f"Single action entropy should be ~0, got: {single_action_entropy}"


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
