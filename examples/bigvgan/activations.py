import math

import torch
from torch import addcdiv, sin, square
from torch.autograd import Function
from torch.nn import Module, Parameter
from torch.nn import functional as F


class SnakeFunction(Function):
    """
    Autograd function implementing the serpentine-like sine-based periodic activation function.

    .. math::
         \text{Snake}_a := x + \frac{1}{a} \sin^2(ax)

    This function computes the forward and backward pass for the Snake activation, which helps in better
    extrapolating to unseen data, particularly when dealing with periodic functions.

    Attributes:
        ctx (torch.autograd.function._ContextMethodMixin): Context object used for saving and retrieving tensors.
    """

    @staticmethod
    def forward(ctx, x: torch.Tensor, a: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for the Snake activation function.

        Args:
            x (torch.Tensor): Input tensor.
            a (torch.Tensor): Trainable parameter controlling the frequency of the sine function.

        Returns:
            torch.Tensor: Result of applying the Snake activation function to the input tensor.
        """
        ctx.save_for_backward(x, a)

        # Handle case where `a` is zero to avoid division by zero errors.
        return torch.where(a == 0, x, addcdiv(x, square(sin(a * x)), a))

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> tuple[torch.Tensor | None, torch.Tensor | None]:
        """
        Backward pass for the Snake activation function.

        Args:
            grad_output (torch.Tensor): The gradient of the loss with respect to the output.

        Returns:
            tuple[torch.Tensor, torch.Tensor]: The gradients of the loss with respect to `x` and `a`.
        """
        x, a = ctx.saved_tensors

        # Calculate the gradient of the input `x`
        sin2ax = sin(2 * a * x) if any(ctx.needs_input_grad) else None
        grad_x = grad_output * (1 + sin2ax) if ctx.needs_input_grad[0] else None

        # Calculate the gradient of the parameter `a`
        grad_a = (
            grad_output * torch.where(a == 0, square(x), sin2ax * x / a - square(sin(a * x) / a))
            if ctx.needs_input_grad[1]
            else None
        )

        return grad_x, grad_a


class Snake(Module):
    """
    Implementation of the Snake activation function as a torch module.

    .. math::
         \text{Snake}_a := x + \frac{1}{a} \sin^2(ax) = x - \frac{1}{2a}\cos(2ax) + \frac{1}{2a}

    This activation function is designed to better extrapolate unseen data, particularly periodic functions.

    Parameters:
        in_features (int or list): The shape or number of input features.
        a (float, optional): Initial value of the trainable parameter `a`, controlling the sine frequency. Defaults to None.
        trainable (bool, optional): If `True`, the parameter `a` will be trainable. Defaults to True.

    Examples:
        >>> snake_layer = Snake(256)
        >>> x = torch.randn(256)
        >>> x = snake_layer(x)

    References:
        - This activation function is from this paper by Liu Ziyin, Tilman Hartwig, Masahito Ueda:
        https://arxiv.org/abs/2006.08195
    """

    def __init__(
        self,
        in_features: int | list[int],
        a: float | None = None,
        trainable: bool = True,
    ):
        """
        Initialize the Snake activation layer.

        Args:
            in_features (int or list): Shape of the input, either a single integer or a list of integers indicating feature dimensions.
            a (float, optional): Initial value for the parameter `a`, which controls the sine frequency. If not provided, `a` will be initialized to a random value from an exponential distribution.
            trainable (bool, optional): If `True`, the parameter `a` will be trained during backpropagation.
        """
        super(Snake, self).__init__()
        self.in_features = in_features if isinstance(in_features, list) else [in_features]

        # Ensure initial_a is a floating point tensor
        if isinstance(in_features, int):
            initial_a = torch.full((in_features,), a, dtype=torch.float32)  # Explicitly set dtype to float32
        else:
            initial_a = torch.full(
                in_features, a, dtype=torch.float32
            )  # Assuming in_features is a list/tuple of dimensions

        if trainable:
            self.a = Parameter(initial_a)
        else:
            self.register_buffer("a", initial_a)

        if trainable:
            self.a = Parameter(initial_a)
        else:
            self.register_buffer("a", initial_a)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for the Snake activation layer.

        Args:
            x (torch.Tensor): Input tensor to the layer.

        Returns:
            torch.Tensor: Result of applying the Snake activation function.
        """
        return SnakeFunction.apply(x, self.a)


class Activation1d(Module):
    def __init__(
        self,
        activation,
        up_ratio: int = 2,
        down_ratio: int = 2,
        up_kernel_size: int = 12,
        down_kernel_size: int = 12,
    ):
        super().__init__()
        self.up_ratio = up_ratio
        self.down_ratio = down_ratio
        self.act = activation
        self.upsample = UpSample1d(up_ratio, up_kernel_size)
        self.downsample = DownSample1d(down_ratio, down_kernel_size)

    # x: [B,C,T]
    def forward(self, x):
        x = self.upsample(x)
        x = self.act(x)
        x = self.downsample(x)

        return x


class UpSample1d(Module):
    def __init__(self, ratio=2, kernel_size=None):
        super().__init__()
        self.ratio = ratio
        self.kernel_size = int(6 * ratio // 2) * 2 if kernel_size is None else kernel_size
        self.stride = ratio
        self.pad = self.kernel_size // ratio - 1
        self.pad_left = self.pad * self.stride + (self.kernel_size - self.stride) // 2
        self.pad_right = self.pad * self.stride + (self.kernel_size - self.stride + 1) // 2
        filter = kaiser_sinc_filter1d(cutoff=0.5 / ratio, half_width=0.6 / ratio, kernel_size=self.kernel_size)
        self.register_buffer("filter", filter)

    # x: [B, C, T]
    def forward(self, x):
        _, C, _ = x.shape

        x = F.pad(x, (self.pad, self.pad), mode="replicate")
        x = self.ratio * F.conv_transpose1d(x, self.filter.expand(C, -1, -1), stride=self.stride, groups=C)
        x = x[..., self.pad_left : -self.pad_right]

        return x


class DownSample1d(Module):
    def __init__(self, ratio=2, kernel_size=None):
        super().__init__()
        self.ratio = ratio
        self.kernel_size = int(6 * ratio // 2) * 2 if kernel_size is None else kernel_size
        self.lowpass = LowPassFilter1d(
            cutoff=0.5 / ratio,
            half_width=0.6 / ratio,
            stride=ratio,
            kernel_size=self.kernel_size,
        )

    def forward(self, x):
        xx = self.lowpass(x)

        return xx


# This code is adopted from adefossez's julius.lowpass.LowPassFilters under the MIT License
# https://adefossez.github.io/julius/julius/lowpass.html
#   LICENSE is in incl_licenses directory.
def kaiser_sinc_filter1d(cutoff, half_width, kernel_size):  # return filter [1,1,kernel_size]
    even = kernel_size % 2 == 0
    half_size = kernel_size // 2

    # For kaiser window
    delta_f = 4 * half_width
    A = 2.285 * (half_size - 1) * math.pi * delta_f + 7.95
    if A > 50.0:
        beta = 0.1102 * (A - 8.7)
    elif A >= 21.0:
        beta = 0.5842 * (A - 21) ** 0.4 + 0.07886 * (A - 21.0)
    else:
        beta = 0.0
    window = torch.kaiser_window(kernel_size, beta=beta, periodic=False)

    # ratio = 0.5/cutoff -> 2 * cutoff = 1 / ratio
    if even:
        time = torch.arange(-half_size, half_size) + 0.5
    else:
        time = torch.arange(kernel_size) - half_size
    if cutoff == 0:
        filter_ = torch.zeros_like(time)
    else:
        filter_ = 2 * cutoff * window * torch.sinc(2 * cutoff * time)
        """
        Normalize filter to have sum = 1, otherwise we will have a small leakage of the constant component in the input signal.
        """
        filter_ /= filter_.sum()
        filter = filter_.view(1, 1, kernel_size)

    return filter


class LowPassFilter1d(Module):
    def __init__(
        self,
        cutoff=0.5,
        half_width=0.6,
        stride: int = 1,
        padding: bool = True,
        padding_mode: str = "replicate",
        kernel_size: int = 12,
    ):
        """
        kernel_size should be even number for stylegan3 setup, in this implementation, odd number is also possible.
        """
        super().__init__()
        if cutoff < -0.0:
            raise ValueError("Minimum cutoff must be larger than zero.")
        if cutoff > 0.5:
            raise ValueError("A cutoff above 0.5 does not make sense.")
        self.kernel_size = kernel_size
        self.even = kernel_size % 2 == 0
        self.pad_left = kernel_size // 2 - int(self.even)
        self.pad_right = kernel_size // 2
        self.stride = stride
        self.padding = padding
        self.padding_mode = padding_mode
        filter = kaiser_sinc_filter1d(cutoff, half_width, kernel_size)
        self.register_buffer("filter", filter)

    # Input [B, C, T]
    def forward(self, x):
        _, C, _ = x.shape

        if self.padding:
            x = F.pad(x, (self.pad_left, self.pad_right), mode=self.padding_mode)
        out = F.conv1d(x, self.filter.expand(C, -1, -1), stride=self.stride, groups=C)

        return out
