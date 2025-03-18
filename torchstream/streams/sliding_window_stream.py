from numbers import Number
from typing import Union

from torchstream import Stream
from torchstream.buffers.buffer_read import BufferRead
from torchstream.buffers.stream_buffer import StreamBuffer
from torchstream.streams.left_right_pad_stream import LeftRightPadStream
from torchstream.streams.stream_step import stream_step


class SlidingWindowStream(Stream):
    def __init__(
        self, win_size: int, stride: int, pad: Union[tuple, int]=None, pad_mode="constant", drop_last=False
    ):
        # FIXME! padding
        if pad is None:
            pad = ((win_size - 1) // 2, win_size // 2)
        if isinstance(pad, Number):
            pad = (pad, pad)
        self.pad_stream = LeftRightPadStream(pad, pad_mode) if pad != (0, 0) else None

        self.win_size = win_size
        self.stride = stride
        # N.B.: can be negative
        self.overlap = self.win_size - self.stride
        self.drop_last = drop_last
        self._to_drop = 0

        super().__init__((self.pad_stream, self._self_step) if self.pad_stream else self._self_step)

    # FIXME: overloads
    def out_n_wins(self, read: BufferRead) -> int:
        return max(0, read.size - self.win_size + self.stride) // self.stride

    # def _out_size(self, read: BufferRead) -> Union[int, Tuple]:
    #     n_wins = self.out_n_wins(read)
    #     return self.win_size + (n_wins - 1) * self.stride if n_wins else 0

    @stream_step()
    def _self_step(self, buf: StreamBuffer):
        if self._to_drop:
            self._to_drop -= buf.drop(self._to_drop)

        if buf.input_closed and not self.drop_last:
            read_size = buf.size
        else:
            n_wins = max(0, buf.size - self.win_size + self.stride) // self.stride
            read_size = self.win_size + (n_wins - 1) * self.stride
            # FIXME: neg numbers
            assert read_size >= 0

        assert buf.size >= read_size, "Internal error"
        out = buf.peek(read_size)

        # We don't want to keep any data in the buffer on the last step
        drop_size = buf.size if buf.input_closed else (read_size - self.overlap)
        self._to_drop = drop_size - buf.drop(drop_size)

        return out
