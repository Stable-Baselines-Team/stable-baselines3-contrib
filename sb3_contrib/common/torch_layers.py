import torch

__all__ = ["BatchRenorm1d", "BatchRenorm"]


class BatchRenorm(torch.nn.Module):
    """
    BatchRenorm Module (https://arxiv.org/abs/1702.03275).
    Adapted to Pytorch from
    https://github.com/araffin/sbx/blob/master/sbx/common/jax_layers.py

    BatchRenorm is an improved version of vanilla BatchNorm. Contrary to BatchNorm,
    BatchRenorm uses the running statistics for normalizing the batches after a warmup phase.
    This makes it less prone to suffer from "outlier" batches that can happen
    during very long training runs and, therefore, is more robust during long training runs.

    During the warmup phase, it behaves exactly like a BatchNorm layer. After the warmup phase,
    the running statistics are used for normalization. The running statistics are updated during
    training mode. During evaluation mode, the running statistics are used for normalization but
    not updated.

    :param num_features: Number of features in the input tensor.
    :param eps: A value added to the variance for numerical stability.
    :param momentum: The value used for the ra_mean and ra_var (running average) computation.
        It controls the rate of convergence for the batch renormalization statistics.
    :param affine: A boolean value that when set to True, this module has learnable
            affine parameters. Default: True
    :param warmup_steps: Number of warum steps that are performed before the running statistics
            are used for normalization. During the warump phase, the batch statistics are used.
    """

    def __init__(
        self,
        num_features: int,
        eps: float = 0.001,
        momentum: float = 0.01,
        affine: bool = True,
        warmup_steps: int = 100_000,
    ):
        super().__init__()
        # Running average mean and variance
        self.register_buffer("ra_mean", torch.zeros(num_features, dtype=torch.float))
        self.register_buffer("ra_var", torch.ones(num_features, dtype=torch.float))
        self.register_buffer("steps", torch.tensor(0, dtype=torch.long))
        self.scale = torch.nn.Parameter(torch.ones(num_features, dtype=torch.float))
        self.bias = torch.nn.Parameter(torch.zeros(num_features, dtype=torch.float))

        self.affine = affine
        self.eps = eps
        self.step = 0
        self.momentum = momentum
        self.num_features = num_features
        # Clip scale and bias of the affine transform
        self.rmax = 3.0
        self.dmax = 5.0
        self.warmup_steps = warmup_steps

    def _check_input_dim(self, x: torch.Tensor) -> None:
        raise NotImplementedError()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Normalize the input tensor.

        :param x: Input tensor
        :return: Normalized tensor.
        """

        if self.training:
            # Compute batch statistics
            batch_mean = x.mean(dim=0)
            batch_var = x.var(dim=0)
            batch_std = (batch_var + self.eps).sqrt()

            # Use batch statistics during initial warm up phase.
            # Note: in the original paper, after some warmup phase (batch norm phase of 5k steps)
            # the constraints are linearly relaxed to r_max/d_max over 40k steps
            # Here we only have a warmup phase
            if self.steps > self.warmup_steps:

                running_std = (self.ra_var + self.eps).sqrt()
                # scale
                r = (batch_std / running_std).detach()
                r = r.clamp(1 / self.rmax, self.rmax)
                # bias
                d = ((batch_mean - self.ra_mean) / running_std).detach()
                d = d.clamp(-self.dmax, self.dmax)

                # BatchNorm normalization, using minibatch stats and running average stats
                custom_mean = batch_mean - d * batch_var.sqrt() / r
                custom_var = batch_var / (r**2)

            else:
                custom_mean, custom_var = batch_mean, batch_var

            # Update Running Statistics
            self.ra_mean += self.momentum * (batch_mean.detach() - self.ra_mean)
            self.ra_var += self.momentum * (batch_var.detach() - self.ra_var)
            self.steps += 1

        else:
            # Use running statistics during evaluation mode
            custom_mean, custom_var = self.ra_mean, self.ra_var

        # Normalize
        x = (x - custom_mean[None]) / (custom_var[None] + self.eps).sqrt()

        if self.affine:
            x = self.scale * x + self.bias

        return x

    def extra_repr(self) -> str:
        return (
            f"num_features={self.num_features}, momentum={self.momentum}, "
            f"warmup_steps={self.warmup_steps}, affine={self.affine}"
        )


class BatchRenorm1d(BatchRenorm):
    def _check_input_dim(self, x: torch.Tensor) -> None:
        if x.dim() == 1:
            raise ValueError(f"Expected 2D or 3D input (got {x.dim()}D input)")
