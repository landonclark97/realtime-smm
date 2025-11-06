from __future__ import annotations

from typing import Iterable, List, Optional

import torch
import torch.nn as nn


ACTIVATIONS = {
    "relu": nn.ReLU(inplace=True),
    "gelu": nn.GELU(),
    "leaky_relu": nn.LeakyReLU(),
    "prelu": nn.PReLU(),
    "tanh": nn.Tanh(),
    "sigmoid": nn.Sigmoid(),
    "silu": nn.SiLU(inplace=True),
}

def _get_activation(name: str) -> nn.Module:
    key = name.strip().lower()
    activation = ACTIVATIONS.get(key)
    if activation is None:
        raise ValueError(f"Unsupported activation: {name}")
    return activation


class MLP(nn.Module):
    """Configurable multilayer perceptron.

    Args:
        input_dim: Number of input features.
        hidden_dims: Iterable of hidden layer sizes.
        output_dim: Number of output features.
        activation: Activation function name used between all layers (e.g., 'relu', 'gelu').
        dropout: Dropout probability applied after each hidden layer (0 disables).
        layer_norm: If True, applies LayerNorm after each linear layer (before activation).
        bias: Whether to include bias terms in linear layers.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dims: Iterable[int],
        output_dim: int,
        *,
        activation: str = "relu",
        dropout: float = 0.0,
        layer_norm: bool = False,
        bias: bool = True,
    ) -> None:
        super().__init__()
        hidden: List[int] = list(int(h) for h in hidden_dims)
        if any(h <= 0 for h in hidden):
            raise ValueError("hidden_dims must contain positive integers")
        if input_dim <= 0 or output_dim <= 0:
            raise ValueError("input_dim and output_dim must be positive")

        act = _get_activation(activation)
        layers: List[nn.Module] = []

        dims: List[int] = [int(input_dim), *hidden, int(output_dim)]
        for i in range(len(dims) - 1):
            in_f, out_f = dims[i], dims[i + 1]
            lin = nn.Linear(in_f, out_f, bias=bias)
            self._init_linear(lin, activation)
            layers.append(lin)

            is_last = i == len(dims) - 2
            if not is_last:
                if layer_norm:
                    layers.append(nn.LayerNorm(out_f))
                layers.append(_get_activation(activation))
                if dropout and dropout > 0.0:
                    layers.append(nn.Dropout(p=float(dropout)))

        self.net = nn.Sequential(*layers)

    @staticmethod
    def _init_linear(layer: nn.Linear, activation: str) -> None:
        # Kaiming for ReLU/SiLU; Xavier otherwise
        act = activation.strip().lower()
        if act in ("relu", "silu", "swish"):
            nn.init.kaiming_uniform_(layer.weight, a=0.0, nonlinearity="relu")
        elif act in ("gelu",):
            # GELU pairs well with xavier normal/uniform
            nn.init.xavier_uniform_(layer.weight)
        else:
            nn.init.xavier_uniform_(layer.weight)
        if layer.bias is not None:
            nn.init.zeros_(layer.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        return self.net(x)
