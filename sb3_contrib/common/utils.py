import torch as th


def quantile_huber_loss(current_quantile: th.Tensor, target_quantile: th.Tensor) -> th.Tensor:
    """
    The quantile-regression loss, as described in the QR-DQN and TQC papers.
    Partially taken from https://github.com/bayesgroup/tqc_pytorch

    :param current_quantile: current estimate of quantile value,
        must be either (batch_size, n_quantiles) or (batch_size, n_critics, n_quantiles)
    :param target_quantile: target quantile value, must be the same shape with current_quantile
    :return: the loss
    """
    n_quantiles = current_quantile.shape[-1]

    # Cumulative probabilities to calculate quantile values.
    cum_prob = (th.arange(n_quantiles, device=current_quantile.device, dtype=th.float) + 0.5) / n_quantiles
    if current_quantile.ndim == 3:
        cum_prob = cum_prob.view(1, 1, -1, 1)  # For TQC
    elif current_quantile.ndim == 2:
        cum_prob = cum_prob.view(1, -1, 1)  # For QR-DQN

    pairwise_delta = target_quantile.unsqueeze(-2) - current_quantile.unsqueeze(-1)
    abs_pairwise_delta = th.abs(pairwise_delta)
    huber_loss = th.where(abs_pairwise_delta > 1, abs_pairwise_delta - 0.5, pairwise_delta ** 2 * 0.5)
    loss = (th.abs(cum_prob - (pairwise_delta.detach() < 0).float()) * huber_loss).mean()
    return loss
