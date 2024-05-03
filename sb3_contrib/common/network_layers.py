import torch

__all__ = ["BatchRenorm1d"]


import torch
import torch.nn as nn


class BatchRenorm(torch.jit.ScriptModule):
    """
    BatchRenorm Module (https://arxiv.org/abs/1702.03275).
    Adapted from flax.linen.normalization.BatchNorm

    BatchRenorm is an improved version of vanilla BatchNorm. Contrary to BatchNorm,
    BatchRenorm uses the running statistics for normalizing the batches after a warmup phase.
    This makes it less prone to suffer from "outlier" batches that can happen
    during very long training runs and, therefore, is more robust during long training runs.

    During the warmup phase, it behaves exactly like a BatchNorm layer.

    Args:
        num_features: Number of features in the input tensor.
        eps: A value added to the variance for numerical stability.
        momentum: The value used for the running_mean and running_var computation.
        affine: A boolean value that when set to True, this module has learnable
            affine parameters. Default: True
    """

    def __init__(
        self,
        num_features: int,
        eps: float = 0.001,
        momentum: float = 0.01,
        affine: bool = True,
    ):
        super().__init__()
        self.register_buffer("running_mean", torch.zeros(num_features, dtype=torch.float))
        self.register_buffer("running_var", torch.ones(num_features, dtype=torch.float))
        self.register_buffer("num_batches_tracked", torch.tensor(0, dtype=torch.long))
        self.scale = torch.nn.Parameter(torch.ones(num_features, dtype=torch.float))
        self.bias = torch.nn.Parameter(torch.zeros(num_features, dtype=torch.float))

        self.affine = affine
        self.eps = eps
        self.step = 0
        self.momentum = momentum
        self.rmax = 3.0
        self.dmax = 5.0

    def _check_input_dim(self, x: torch.Tensor) -> None:
        raise NotImplementedError()

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        if self.training:
            batch_mean = x.mean(0)
            batch_var = x.var(0)
            batch_std = (batch_var + self.eps).sqrt()

            # Use batch statistics during initial warm up phase.
            if self.num_batches_tracked > 100_000:

                running_std = (self.running_var + self.eps).sqrt()
                running_mean = self.running_mean

                r = (batch_std / running_std).detach()
                r = r.clamp(1 / self.rmax, self.rmax)
                d = ((batch_mean - running_mean) / running_std).detach()
                d = d.clamp(-self.dmax, self.dmax)

                m = batch_mean - d * batch_var.sqrt() / r
                v = batch_var / (r**2)

            else:
                m, v = batch_mean, batch_var

            # Update Running Statistics
            self.running_mean += self.momentum * (batch_mean.detach() - self.running_mean)
            self.running_var += self.momentum * (batch_var.detach() - self.running_var)
            self.num_batches_tracked += 1

        else:
            m, v = self.running_mean, self.running_var

        # Normalize
        x = (x - m[None]) / (v[None] + self.eps).sqrt()

        if self.affine:
            x = self.scale * x + self.bias

        return x


class BatchRenorm1d(BatchRenorm):
    def _check_input_dim(self, x: torch.Tensor) -> None:
        if x.dim() == 1:
            raise ValueError("expected 2D or 3D input (got {x.dim()}D input)")
