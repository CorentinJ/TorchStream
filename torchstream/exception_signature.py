"""
This module helps us catch specific exceptions using substrings in their messages, so that we avoid swallowing
legitimate exceptions that happen to be of the same type.
"""

from typing import Iterable, Tuple

ExceptionWithSubstring = Tuple[Exception, str]


# When forwarding to a conv an input tensor smaller than the conv's kernel size
TORCH_CONV_INPUT_SMALLER_THAN_KERNEL = (RuntimeError, "Kernel size can't be greater than actual input size")
# When forwarding to a transposed conv an input tensor that would lead to a too small output size after trimming
TORCH_TCONV_TRIMMING_TOO_LARGE = (RuntimeError, "Output size is too small")
# When using reflect padding with a padding size larger (on either side) than the input size
TORCH_REFLECT_PADDING_LARGER_THAN_INPUT = (
    RuntimeError,
    "Padding size should be less than the corresponding input dimension",
)
# Using instance norm with a single input
TORCH_INSTANCENORM_SINGLE_ELEMENT = (ValueError, "Expected more than 1 spatial element when training")
# Torchaudio's STFT
TORCHAUDIO_STFT_SMALLER_THAN_NFFT = (RuntimeError, "expected 0 < n_fft <")

DEFAULT_ZERO_SIZE_EXCEPTIONS = [
    TORCH_CONV_INPUT_SMALLER_THAN_KERNEL,
    TORCH_TCONV_TRIMMING_TOO_LARGE,
    TORCH_REFLECT_PADDING_LARGER_THAN_INPUT,
    TORCH_INSTANCENORM_SINGLE_ELEMENT,
    TORCHAUDIO_STFT_SMALLER_THAN_NFFT,
]


def matches_any_exception(
    e: Exception,
    exception_signatures: Iterable[Exception | ExceptionWithSubstring],
) -> bool:
    """
    Returns whether the given exception matches any of the provided exception signatures by:
    - Either the type alone
    - Either the type and a substring contained in the exception message
    """
    for exception_signature in exception_signatures:
        if isinstance(exception_signature, tuple):
            exception_type, substring = exception_signature
            if isinstance(e, exception_type) and substring in str(e):
                return True
        else:
            if isinstance(e, exception_signature):
                return True
    return False
