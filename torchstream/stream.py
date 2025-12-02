from typing import Iterator, Optional, Tuple, overload

from torchstream.sequence.dtype import SeqArrayLike
from torchstream.sequence.sequence import SeqSpec, Sequence


class Stream:
    """
    Base class for streaming a transform over sequential data fed in chunks.

    A Stream instance lives for the duration of the inference of a single streaming request. Each new inference request
    must instantiate its own Stream subclass instance.

    The Stream class offers the following features:
    - Management of a transform's statefulness outside of the transform itself
    - Rigorous specification of input and output format
    - Graceful handling of transforms failing due to a lack of input data
    - Convenience methods for forwarding data step by step, in an iterator style or offline (for testing purposes)

    Subclasses must override the `_step` method.

    :param in_spec: Specification of the input sequence format. The _step method will receive inputs matching this
    specification.
    :param out_spec: Specification of the output sequence format. The _step method must return outputs matching this
    specification. If None, it is assumed to be the same as the input specification.
    """

    def __init__(
        self,
        in_spec: SeqSpec,
        out_spec: Optional[SeqSpec] = None,
    ):
        self.in_spec = in_spec
        self.out_spec = out_spec or in_spec

        self._n_steps = 0
        self._total_in_fed = 0
        self._total_out_produced = 0

        self._closed = False

        self._in_buff = Sequence(self.in_spec)

    @property
    def n_steps(self) -> int:
        """
        Number of times the stream has been succesfully stepped so far. Steps that yield a zero-sized output do not
        count towards this value.
        """
        return self._n_steps

    @property
    def total_in_fed(self) -> int:
        """
        Total size of the input sequence received in __call__() so far.
        """
        return self._total_in_fed

    @property
    def total_out_produced(self) -> int:
        """
        Total size of the output sequence produced by __call__() so far.
        """
        return self._total_out_produced

    @property
    def is_closed(self) -> bool:
        """
        Whether the stream has been closed. A closed stream will no longer accept inputs. A stream is closed upon
        calling __call__() with is_last_input=True. The close_input() method is a convenience wrapper to do so without
        having to provide any input.
        """
        return self._closed

    @overload
    def __call__(
        self, input: Sequence, /, *, is_last_input: bool = False, allow_zero_size_outputs: bool = False
    ) -> Sequence: ...
    @overload
    def __call__(
        self, *in_arrs: SeqArrayLike, is_last_input: bool = False, allow_zero_size_outputs: bool = False
    ) -> Sequence: ...
    def __call__(
        self, *inputs: Sequence | SeqArrayLike, is_last_input: bool = False, allow_zero_size_outputs: bool = False
    ) -> Sequence:
        """
        Forwards input data through the stream.

        The given data is buffered internally and the whole buffer is given to the _step() method, which is responsible
        for consuming the appropriate amount of input data from the buffer and producing output data.

        Stream implementations are expected not to hold back any output (unless said output is tentative, depending on
        future inputs), so all the available output will be returned by this method.

        :param inputs: Input data to feed to the stream. Can be provided either as a single Sequence instance or as
        as individual arrays matching the input specification.
        NOTE: this interface does not expose any mechanism to pass non-sequential data to the stream. You are
        expected to provide such data at the stream's constructor.
        TODO?: consider passing kwargs for this
        :param is_last_input: Whether this is the last input data to be provided to the stream. Some streams will
        behave differently in the presence of this flag. If you do not provide it, your stream might output less
        data than it could have.
        :param allow_zero_size_outputs: When enabled, any stream that raises a NotEnoughInputError during stepping
        will not raise the error and instead return a zero-sized output sequence.
        """
        if self.is_closed:
            raise RuntimeError("Cannot step with stream: it is closed")

        if is_last_input:
            self._closed = True

        if len(inputs):
            prev_size = self._in_buff.size
            self._in_buff.feed(*inputs)
            self._total_in_fed += self._in_buff.size - prev_size

        try:
            out_seq = self._step(self._in_buff, is_last_input=is_last_input)

            self._n_steps += 1

            if not isinstance(out_seq, Sequence):
                if isinstance(out_seq, tuple):
                    out_seq = self.out_spec.new_sequence_from_data(*out_seq)
                else:
                    out_seq = self.out_spec.new_sequence_from_data(out_seq)
        except NotEnoughInputError:
            if not allow_zero_size_outputs:
                raise
            else:
                out_seq = self.out_spec.new_empty_sequence()

        self._total_out_produced += out_seq.size

        if self.is_closed:
            self._in_buff.clear()

        return out_seq

    def close_input(self) -> Sequence:
        """
        Equivalent to calling __call__() with is_last_input=True, no input data and allow_zero_size_outputs=True.

        Some streams may yield outputs even when no input is provided on closing. The only legitimate reason for this
        is when a transform holds back output because it is only correct if the input is complete (=closed). For
        example, adding right padding to a streaming input is only done at the end of the input sequence.
        """
        return self(is_last_input=True, allow_zero_size_outputs=True)

    def _step(self, in_buff: Sequence, is_last_input: bool) -> Sequence | Tuple[SeqArrayLike, ...] | SeqArrayLike:
        """
        TODO! instruct how to override

        principles

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
            # FIXME! expose allow_zero_size_outputs
            yield self(ext_in_buff.read(chunk_size), is_last_input=not ext_in_buff.size, allow_zero_size_outputs=True)

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
