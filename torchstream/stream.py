from typing import Iterator, Optional, Tuple, overload

from torchstream.sequence.dtype import SeqArrayLike
from torchstream.sequence.sequence import SeqSpec, Sequence


class Stream:
    """
    TODO!: doc
    """

    def __init__(
        self,
        input_spec: SeqSpec,
        output_spec: Optional[SeqSpec] = None,
    ):
        self.in_spec = input_spec
        self.out_spec = output_spec or input_spec

        self._total_in_fed = 0
        self._total_out_produced = 0

        self._closed = False

        self._in_buff = Sequence(self.in_spec)

    @property
    def total_in_fed(self) -> int:
        return self._total_in_fed

    @property
    def total_out_produced(self) -> int:
        return self._total_out_produced

    @property
    def is_closed(self) -> bool:
        return self._closed

    @overload
    def __call__(
        self, input: Sequence, /, *, is_last_input: bool = False, raise_on_starve: bool = False
    ) -> Sequence: ...
    @overload
    def __call__(
        self, *in_arrs: SeqArrayLike, is_last_input: bool = False, raise_on_starve: bool = False
    ) -> Sequence: ...
    def __call__(
        self, *inputs: Sequence | SeqArrayLike, is_last_input: bool = False, raise_on_starve: bool = False
    ) -> Sequence:
        if self.is_closed:
            raise RuntimeError("Cannot step with stream: it is closed")

        if is_last_input:
            self._closed = True

        if len(inputs):
            prev_size = self._in_buff.size
            self._in_buff.feed(*inputs)
            self._total_in_fed += self._in_buff.size - prev_size

        try:
            out_seq = self._step(self._in_buff)
            if not isinstance(out_seq, Sequence):
                if isinstance(out_seq, tuple):
                    out_seq = self.out_spec.new_sequence_from_data(*out_seq)
                else:
                    out_seq = self.out_spec.new_sequence_from_data(out_seq)
        except NotEnoughInputError:
            if raise_on_starve:
                raise
            else:
                out_seq = self.out_spec.new_empty_sequence()

        self._total_out_produced += out_seq.size

        return out_seq

    def close_input(self) -> Sequence:
        return self(is_last_input=True)

    def _step(self, in_buff: Sequence, is_last_input: bool) -> Sequence | Tuple[SeqArrayLike, ...] | SeqArrayLike:
        """
        TODO! instruct how to override

        :raises NotEnoughInputsError: if the stream cannot perform a step because it does not have enough inputs. This
        is typically a low severity error that can be caught by the caller in order to wait for more inputs before
        stepping again...
        """
        raise NotImplementedError()

    # TODO: offer options to specify variable chunk sizes
    @overload
    def forward_in_chunks_iter(self, input: Sequence, chunk_size: int) -> Iterator[Sequence]: ...
    @overload
    def forward_in_chunks_iter(self, *in_arrs: SeqArrayLike, chunk_size: int) -> Iterator[Sequence]: ...
    def forward_in_chunks_iter(self, *inputs, chunk_size: int) -> Iterator[Sequence]:
        """
        Convenience method to forward an input sequence in chunks of fixed size through the stream. The stream will
        be closed on the last step automatically. The data is not consumed if the input is provided as a Sequence.
        """
        if isinstance(inputs[0], Sequence):
            ext_in_buff = inputs[0].copy()
        else:
            ext_in_buff = self.in_spec.new_sequence_from_data(*inputs)

        while ext_in_buff.size:
            yield self(ext_in_buff.read(chunk_size), is_last_input=not ext_in_buff.size)

    # TODO: offer options to specify variable chunk sizes
    @overload
    def forward_in_chunks(self, input: Sequence, chunk_size: int) -> Sequence: ...
    @overload
    def forward_in_chunks(self, *in_arrs: SeqArrayLike, chunk_size: int) -> Sequence: ...
    def forward_in_chunks(self, *inputs, chunk_size: int) -> Sequence:
        """
        Convenience method to forward an input sequence in chunks of fixed size through the stream and return the
        full output sequence. This is typically used for testing a stream, given that it defeats the purpose of
        streaming. The stream will be closed on the last step automatically. The data is not consumed if the input
        is provided as a Sequence.
        """
        out_buff = self.out_spec.new_empty_sequence()
        for out_chunk in self.forward_in_chunks_iter(*inputs, chunk_size=chunk_size):
            out_buff.feed(out_chunk)
        return out_buff


class NotEnoughInputError(Exception):
    """
    TODO: doc
    """

    pass
