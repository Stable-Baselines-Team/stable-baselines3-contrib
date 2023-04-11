import torch
from torch import nn
from typing import Any, Dict, List, Optional, Tuple, Type, TypeVar, Union, Callable
from stable_baselines3.common.utils import get_device

# Code from RLlib: https://github.com/ray-project/ray/blob/master/rllib/models/torch/attention_net.py

def sequence_mask(
    lengths,
    maxlen: Optional[int] = None,
    dtype=None,
    time_major: bool = False,
):
    """Offers same behavior as tf.sequence_mask for torch.
    Thanks to Dimitris Papatheodorou
    (https://discuss.pytorch.org/t/pytorch-equivalent-for-tf-sequence-mask/
    39036).
    :param lengths: The tensor of individual lengths to mask by.
    :param maxlen: The maximum length to use for the time axis. If None, use
        the max of `lengths`.
    :param dtype: The torch dtype to use for the resulting mask.
    :param time_major: Whether to return the mask as [B, T] (False; default) or
        as [T, B] (True).
    :return: The sequence mask resulting from the given input and parameters.
    """
    # If maxlen not given, use the longest lengths in the `lengths` tensor.
    if maxlen is None:
        maxlen = int(lengths.max())

    mask = ~(
        torch.ones((len(lengths), maxlen)).to(lengths.device).cumsum(dim=1).t()
        > lengths
    )
    # Time major transformation.
    if not time_major:
        mask = mask.t()

    # By default, set the mask to be boolean.
    mask.type(dtype or torch.bool)

    return mask

class SlimFC(nn.Module):
    """Simple PyTorch version of `linear` function"""

    def __init__(
        self,
        in_size: int,
        out_size: int,
        initializer: Any = None,
        activation_fn: Any = None,
        use_bias: bool = True,
        bias_init: float = 0.0,
    ):
        """Creates a standard FC layer, similar to torch.nn.Linear
        :param in_size: Input size for FC Layer
        :param out_size: Output size for FC Layer
        :param initializer: Initializer function for FC layer weights
        :param activation_fn: Activation function at the end of layer
        :param use_bias: Whether to add bias weights or not
        :param bias_init: Initalize bias weights to bias_init const
        """
        super(SlimFC, self).__init__()
        layers = []
        # Actual nn.Linear layer (including correct initialization logic).
        linear = nn.Linear(in_size, out_size, bias=use_bias)
        if initializer is None:
            initializer = nn.init.xavier_uniform_
        initializer(linear.weight)
        if use_bias is True:
            nn.init.constant_(linear.bias, bias_init)
        layers.append(linear)
        # Activation function (if any; default=None (linear)).
        if activation_fn is not None:
            layers.append(activation_fn())
        # Put everything in sequence.
        self._model = nn.Sequential(*layers)

    def forward(self, x):
        return self._model(x)


class GRUGate(nn.Module):
    """Implements a gated recurrent unit for use in AttentionNet"""

    def __init__(self, dim: int, init_bias: int = 0.0, **kwargs):
        """
        :param input_shape (torch.Tensor): dimension of the input
        :param init_bias: Bias added to every input to stabilize training
        """
        super().__init__(**kwargs)
        # Xavier initialization of torch tensors
        self._w_r = nn.Parameter(torch.zeros(dim, dim))
        self._w_z = nn.Parameter(torch.zeros(dim, dim))
        self._w_h = nn.Parameter(torch.zeros(dim, dim))
        nn.init.xavier_uniform_(self._w_r)
        nn.init.xavier_uniform_(self._w_z)
        nn.init.xavier_uniform_(self._w_h)
        self.register_parameter("_w_r", self._w_r)
        self.register_parameter("_w_z", self._w_z)
        self.register_parameter("_w_h", self._w_h)

        self._u_r = nn.Parameter(torch.zeros(dim, dim))
        self._u_z = nn.Parameter(torch.zeros(dim, dim))
        self._u_h = nn.Parameter(torch.zeros(dim, dim))
        nn.init.xavier_uniform_(self._u_r)
        nn.init.xavier_uniform_(self._u_z)
        nn.init.xavier_uniform_(self._u_h)
        self.register_parameter("_u_r", self._u_r)
        self.register_parameter("_u_z", self._u_z)
        self.register_parameter("_u_h", self._u_h)

        self._bias_z = nn.Parameter(
            torch.zeros(
                dim,
            ).fill_(init_bias)
        )
        self.register_parameter("_bias_z", self._bias_z)

    def forward(self, inputs, **kwargs):
        # Pass in internal state first.
        h, X = inputs

        r = torch.tensordot(X, self._w_r, dims=1) + torch.tensordot(
            h, self._u_r, dims=1
        )
        r = torch.sigmoid(r)

        z = (
            torch.tensordot(X, self._w_z, dims=1)
            + torch.tensordot(h, self._u_z, dims=1)
            - self._bias_z
        )
        z = torch.sigmoid(z)

        h_next = torch.tensordot(X, self._w_h, dims=1) + torch.tensordot(
            (h * r), self._u_h, dims=1
        )
        h_next = torch.tanh(h_next)

        return (1 - z) * h + z * h_next


class SkipConnection(nn.Module):
    """Skip connection layer.
    Adds the original input to the output (regular residual layer) OR uses
    input as hidden state input to a given fan_in_layer.
    """

    def __init__(self, layer: nn.Module, fan_in_layer: Optional[nn.Module] = None, **kwargs):
        """Initializes a SkipConnection nn Module object.
        :param layer (nn.Module): Any layer processing inputs.
        :param fan_in_layer (Optional[nn.Module]): An optional
            layer taking two inputs: The original input and the output
            of `layer`.
        """
        super().__init__(**kwargs)
        self._layer = layer
        self._fan_in_layer = fan_in_layer

    def forward(self, inputs, **kwargs):
        # del kwargs
        outputs = self._layer(inputs, **kwargs)
        # Residual case, just add inputs to outputs.
        if self._fan_in_layer is None:
            outputs = outputs + inputs
        # Fan-in e.g. RNN: Call fan-in with `inputs` and `outputs`.
        else:
            # NOTE: In the GRU case, `inputs` is the state input.
            outputs = self._fan_in_layer((inputs, outputs))

        return outputs


class RelativePositionEmbedding(nn.Module):
    """Creates a [seq_length x seq_length] matrix for rel. pos encoding.
    Denoted as Phi in [2] and [3]. Phi is the standard sinusoid encoding
    matrix.
    :param seq_length: The max. sequence length (time axis).
    :param out_dim: The number of nodes to go into the first Tranformer
        layer with.
    :return: The encoding matrix Phi.
    """

    def __init__(self, out_dim, **kwargs):
        super().__init__()
        self.out_dim = out_dim

        out_range = torch.arange(0, self.out_dim, 2.0)
        inverse_freq = 1 / (10000 ** (out_range / self.out_dim))
        self.register_buffer("inverse_freq", inverse_freq)

    def forward(self, seq_length):
        pos_input = torch.arange(seq_length - 1, -1, -1.0, dtype=torch.float).to(
            self.inverse_freq.device
        )
        sinusoid_input = torch.einsum("i,j->ij", pos_input, self.inverse_freq)
        pos_embeddings = torch.cat(
            [torch.sin(sinusoid_input), torch.cos(sinusoid_input)], dim=-1
        )
        return pos_embeddings[:, None, :]


class RelativeMultiHeadAttention(nn.Module):
    """A RelativeMultiHeadAttention layer as described in [3].
    Uses segment level recurrence with state reuse.
    """

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        num_heads: int,
        head_dim: int,
        input_layernorm: bool = False,
        output_activation: Union[str, callable] = None,
        **kwargs
    ):
        """Initializes a RelativeMultiHeadAttention nn.Module object.
        :param in_dim (int):
        :param out_dim: The output dimension of this module. Also known as
            "attention dim".
        :param num_heads: The number of attention heads to use.
            Denoted `H` in [2].
        :param head_dim: The dimension of a single(!) attention head
            Denoted `D` in [2].
        :param input_layernorm: Whether to prepend a LayerNorm before
            everything else. Should be True for building a GTrXL.
        :param output_activation (Union[str, callable]): Optional activation
            function or activation function specifier (str).
            Should be "relu" for GTrXL.
        :param **kwargs:
        """
        super().__init__(**kwargs)

        # No bias or non-linearity.
        self._num_heads = num_heads
        self._head_dim = head_dim

        # 3=Query, key, and value inputs.
        self._qkv_layer = SlimFC(
            in_size=in_dim, out_size=3 * num_heads * head_dim, use_bias=False
        )

        self._linear_layer = SlimFC(
            in_size=num_heads * head_dim,
            out_size=out_dim,
            use_bias=False,
            activation_fn=output_activation,
        )

        self._uvar = nn.Parameter(torch.zeros(num_heads, head_dim))
        self._vvar = nn.Parameter(torch.zeros(num_heads, head_dim))
        nn.init.xavier_uniform_(self._uvar)
        nn.init.xavier_uniform_(self._vvar)
        self.register_parameter("_uvar", self._uvar)
        self.register_parameter("_vvar", self._vvar)

        self._pos_proj = SlimFC(
            in_size=in_dim, out_size=num_heads * head_dim, use_bias=False
        )
        self._rel_pos_embedding = RelativePositionEmbedding(out_dim)

        self._input_layernorm = None
        if input_layernorm:
            self._input_layernorm = torch.nn.LayerNorm(in_dim)
        
        #print('in_dim', in_dim)

    def forward(self, inputs, memory=None):
        T = inputs.shape[1] #list(inputs.size())[1]  # length of segment (time)
        H = self._num_heads  # number of attention heads
        d = self._head_dim  # attention head dimension

        # Add previous memory chunk (as const, w/o gradient) to input.
        # Tau (number of (prev) time slices in each memory chunk).
        Tau = memory.shape[1] #list(memory.shape)[1]
        inputs = torch.cat((memory.detach(), inputs), dim=1)

        # Apply the Layer-Norm.
        if self._input_layernorm is not None:
            inputs = self._input_layernorm(inputs)

        qkv = self._qkv_layer(inputs)

        queries, keys, values = torch.chunk(input=qkv, chunks=3, dim=-1)
        # Cut out Tau memory timesteps from query.
        #if memory is not None:
        queries = queries[:, -T:]

        queries = torch.reshape(queries, [-1, T, H, d])
        keys = torch.reshape(keys, [-1, Tau + T, H, d])
        values = torch.reshape(values, [-1, Tau + T, H, d])

        R = self._pos_proj(self._rel_pos_embedding(Tau + T))
        R = torch.reshape(R, [Tau + T, H, d])

        # b=batch
        # i and j=time indices (i=max-timesteps (inputs); j=Tau memory space)
        # h=head
        # d=head-dim (over which we will reduce-sum)
        score = torch.einsum("bihd,bjhd->bijh", queries + self._uvar, keys)
        pos_score = torch.einsum("bihd,jhd->bijh", queries + self._vvar, R)
        score = score + self.rel_shift(pos_score)
        score = score / d**0.5

        # causal mask of the same length as the sequence
        mask = sequence_mask(torch.arange(Tau + 1, Tau + T + 1), dtype=score.dtype).to(score.device)
        mask = mask[None, :, :, None]

        masked_score = score * mask + 1e30 * (mask.float() - 1.0)
        wmat = nn.functional.softmax(masked_score, dim=2)

        out = torch.einsum("bijh,bjhd->bihd", wmat, values)
        shape = list(out.shape)[:2] + [H * d]
        out = torch.reshape(out, shape)

        return self._linear_layer(out)

    @staticmethod
    def rel_shift(x):
        # Transposed version of the shift approach described in [3].
        # https://github.com/kimiyoung/transformer-xl/blob/
        # 44781ed21dbaec88b280f74d9ae2877f52b492a5/tf/model.py#L31
        x_size = list(x.shape)

        x = torch.nn.functional.pad(x, (0, 0, 1, 0, 0, 0, 0, 0))
        x = torch.reshape(x, [x_size[0], x_size[2] + 1, x_size[1], x_size[3]])
        x = x[:, 1:, :, :]
        x = torch.reshape(x, x_size)

        return x


class GTrXLNet(nn.Module):
    """
    Constructs an Attention that receives the output from a previous features extractor or directly the observations (if no features extractor is applied) as an input and outputs a latent representation for the policy and a value network.
    :param feature_dim: Dimension of the feature vector (can be the output of a CNN)
    :param n_layers: Number of layers (MHA + Position-wise MLP)
    :param attention_dim: Dimension of the attention latent space
    :param num_heads: Number of heads of the MHA
    :param memory_inference: (not used)
    :param memory_training: (not used)
    :param head_dim: Heads dimension of the MHA
    :param position_wise_mlp_dim: Dimension of the Position-wise MLP
    :param init_gru_gate_bias: Bias initialization of the GRU gates
    :param device: PyTorch device.
    """
    def __init__(
        self,
        feature_dim: int,
        n_layers: int = 1,
        attention_dim: int = 64,
        num_heads: int = 2,
        memory_inference: int = 50,
        memory_training: int = 50,
        head_dim: int = 32,
        position_wise_mlp_dim: int = 32,
        init_gru_gate_bias: float = 2.0,
        device: Union[torch.device, str] = "auto",
    ) -> None:
        super().__init__()

        device = get_device(device)
        self.input_size = feature_dim
        self.n_layers = n_layers
        self.attention_dim = attention_dim
        self.num_heads = num_heads
        self.memory_inference = memory_inference
        self.memory_training = memory_training
        self.head_dim = head_dim
        
        self.linear_layer = SlimFC(in_size=feature_dim, out_size=self.attention_dim)
        self.layers = [self.linear_layer]

        attention_layers = []
        for i in range(self.n_layers):
            # RelativeMultiHeadAttention part.
            MHA_layer = SkipConnection(
                RelativeMultiHeadAttention(
                    in_dim=self.attention_dim,
                    out_dim=self.attention_dim,
                    num_heads=num_heads,
                    head_dim=head_dim,
                    input_layernorm=True,
                    output_activation=nn.ReLU,
                ),
                fan_in_layer=GRUGate(self.attention_dim, init_gru_gate_bias),
            )

            # Position-wise MultiLayerPerceptron part.
            list_e_layer = [torch.nn.LayerNorm(self.attention_dim),
                            SlimFC(
                                    in_size=self.attention_dim,
                                    out_size=position_wise_mlp_dim,
                                    use_bias=False,
                                    activation_fn=nn.ReLU,
                            ),
                            SlimFC(
                                    in_size=position_wise_mlp_dim,
                                    out_size=self.attention_dim,
                                    use_bias=False,
                                    activation_fn=nn.ReLU,
                            )]
            E_layer = SkipConnection(
                nn.Sequential(*list_e_layer),
                fan_in_layer=GRUGate(self.attention_dim, init_gru_gate_bias),
            )

            # Build a list of all attanlayers in order.
            attention_layers.extend([MHA_layer, E_layer])

        # Create a Sequential such that all parameters inside the attention
        # layers are automatically registered with this top-level model.
        self.attention_layers = nn.Sequential(*attention_layers).to(device)
        self.layers.extend(attention_layers)

        # Final layers if num_outputs not None.
        self.logits = None
        self.values_out = None
        # Last value output.
        self._value_out = None
        # Postprocess GTrXL output with another hidden layer.
        # if self.num_outputs is not None:
        #     self.logits = SlimFC(
        #         in_size=self.attention_dim,
        #         out_size=self.num_outputs,
        #         activation_fn=nn.ReLU,
        #     )

        #     # Value function used by all RLlib Torch RL implementations.
        #     self.values_out = SlimFC(
        #         in_size=self.attention_dim, out_size=1, activation_fn=None
        #     )
        # else:
        #     self.num_outputs = self.attention_dim
        self.num_outputs = self.attention_dim
        

    def forward(self, features: torch.Tensor, memory: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        memory_outs = []
        # print('features:', features.size(), ' | memory:', memory.size())
        for i in range(len(self.layers)):
            # MHA layers which need memory passed in.
            if i % 2 == 1:
                features = self.layers[i](features, memory=memory[i//2].unsqueeze(0))
            # Either self.linear_layer (initial obs -> attn. dim layer) or
            # MultiLayerPerceptrons. The output of these layers is always the
            # memory for the next forward pass.
            else:
                features = self.layers[i](features)
                memory_outs.append(features)

        # Discard last output (not needed as a memory since it's the last
        # layer).
        memory_outs = memory_outs[:-1]

        if self.logits is not None:
            out = self.logits(features)
            self._value_out = self.values_out(features)
            out_dim = self.num_outputs
        else:
            out = features
            out_dim = self.attention_dim
        out = features
        out_dim = self.attention_dim

        # print('out:', torch.reshape(out, [-1, out_dim]).size(), ' | memory_out:', torch.concat([
        #     torch.reshape(m, [1, -1, self.attention_dim]) for m in memory_outs
        # ], dim=0).size())

        return torch.reshape(out, [-1, out_dim]), torch.concat([
            torch.reshape(m, [1, -1, self.attention_dim]) for m in memory_outs
        ], dim=0)