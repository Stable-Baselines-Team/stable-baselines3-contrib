import torch as th


def quantile_huber_loss(current_quantile: th.Tensor, target_quantile: th.Tensor) -> th.Tensor:
    """
    The quantile-regression loss, as described in the QR-DQN and TQC papers.
    Partially taken from https://github.com/bayesgroup/tqc_pytorch

    :param current_quantile: current estimate of quantile value
    :param target_quantile: target quantile value
    :return: the loss
    """
    n_quantiles = current_quantile.shape[-1]
    tau = (th.arange(n_quantiles, device=current_quantile.device).float() + 0.5) / n_quantiles

    if current_quantile.ndim == 3:
        # For TQC.
        tau = tau.view(1, 1, -1, 1)
        pairwise_delta = target_quantile[:, None, None, :] - current_quantile[:, :, :, None]
    elif current_quantile.ndim == 2:
        # For QR-DQN.
        tau = tau.view(1, -1, 1)
        pairwise_delta = target_quantile[:, None, :] - current_quantile[:, :, None]
    else:
        NotImplementedError

    abs_pairwise_delta = th.abs(pairwise_delta)
    huber_loss = th.where(abs_pairwise_delta > 1, abs_pairwise_delta - 0.5, pairwise_delta ** 2 * 0.5)
    loss = (th.abs(tau - (pairwise_delta.detach() < 0).float()) * huber_loss).mean()
    return loss
