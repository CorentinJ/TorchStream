from typing import Optional, Tuple

from torchstream.sequence.seq_spec import SeqSpec
from torchstream.sequence.stream_buffer import StreamBuffer


class Stream:
    def __init__(
        self,
        input_spec: SeqSpec,
        output_spec: Optional[SeqSpec] = None,
    ):
        self.in_spec = input_spec
        self.out_spec = output_spec or input_spec

        self._output_closed = False

        self._in_buffs = self.in_spec.new_empty_buffers()

    # FIXME: settle on the names for both these properties
    @property
    def input_closed(self) -> bool:
        return all(in_buff.input_closed for in_buff in self._in_buffs)

    @property
    def output_closed(self) -> bool:
        return self._output_closed

    def close_input(self):
        for in_buff in self._in_buffs:
            in_buff.close_input()

    def __call__(
        self,
        *inputs: StreamBuffer,
        is_last_input: bool = False,
        on_starve="raise",
    ) -> StreamBuffer | Tuple[StreamBuffer]:
        # TODO: asserts -> exceptions
        assert not self.input_closed
        assert len(inputs) == len(self._in_buffs)

        for in_buff, input_ in zip(self._in_buffs, inputs):
            in_buff.feed(input_, close_input=is_last_input)

        try:
            # TODO! validate this output with the spec
            outputs = self._step(*self._in_buffs)
            outputs = StreamBuffer(self.out_spec, outputs)
        except NotEnoughInputError:
            if on_starve == "raise":
                raise
            elif on_starve == "empty":
                outputs = StreamBuffer.empty(self.out_spec)
        except:
            raise
        finally:
            if self.input_closed:
                self._output_closed = True

        return outputs

    # TODO: settle on return seq vs arrays
    def _step(self, *in_seqs: StreamBuffer) -> StreamBuffer | Tuple[StreamBuffer]:
        """
        :raises NotEnoughInputsError: if the stream cannot perform a step because it does not have enough inputs. This
        is typically a low severity error that can be caught by the caller in order to wait for more inputs before
        stepping again..
        .

        """
        raise NotImplementedError()


class NotEnoughInputError(Exception):
    """
    TODO: doc
    """

    pass
