from typing import Tuple, Union

import numpy as np
from torch import Tensor

from torchstream.buffers.stream_buffer import StreamBuffer


class Stream:
    # TODO: tensorspec?
    def __init__(self, in_dims: Union[int, Tuple[int, ...]], out_dims: Union[int, Tuple[int, ...]]):
        self._in_dims = tuple(in_dims)
        self._out_dims = tuple(out_dims)

        self._output_closed = False

        self._in_buffs = None  # type: Tuple[StreamBuffer]

    @property
    def in_dim(self) -> int:
        if len(self._in_dims) != 1:
            raise ValueError("Stream has multiple input dimensions")
        return self._in_dims[0]

    @property
    def in_dims(self) -> Tuple[int, ...]:
        return self._in_dims

    def out_dim(self) -> int:
        if len(self._out_dims) != 1:
            raise ValueError("Stream has multiple output dimensions")
        return self._out_dims[0]

    @property
    def out_dims(self) -> Tuple[int, ...]:
        return self._out_dims

    @property
    def input_closed(self) -> bool:
        return all(in_buff.input_closed for in_buff in self._in_buffs) if self._in_buffs is not None else False

    @property
    def output_closed(self) -> bool:
        return self._output_closed

    def close_input(self):
        for in_buff in self._in_buffs:
            in_buff.close_input()

    # TODO?: overloads for clarity on the types
    def __call__(
        self, *inputs: Union[Tensor, np.ndarray], is_last_input: bool = False
    ) -> Union[Tensor, np.ndarray, Tuple[Tensor, np.ndarray]]:
        if self._in_buffs is None:
            self._in_buffs = tuple(StreamBuffer(dim=dim) for dim in self._in_dims)

        if len(inputs) != len(self._in_buffs):
            raise ValueError(
                f"Number of inputs ({len(inputs)}) does not match number of input dimensions {len(self._in_dims)}"
            )

        for in_buff, input_ in zip(self._in_buffs, inputs):
            in_buff.feed(input_, close_input=is_last_input)

        outputs = self._step()

        if self.input_closed:
            self._output_closed = True

        return outputs

    def _step(self) -> Union[Tensor, np.ndarray, Tuple[Tensor, np.ndarray]]:
        raise NotImplementedError()
