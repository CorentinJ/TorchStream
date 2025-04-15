from typing import Optional, Tuple

from torchstream.buffers.stream_buffer import StreamBuffer
from torchstream.sequence_spec import SeqSpec, Sequence


class Stream:
    # TODO: multiple inputs/outputs
    def __init__(
        self,
        input_spec: SeqSpec,
        output_spec: Optional[SeqSpec] = None,
    ):
        self.in_spec = input_spec
        self.out_spec = output_spec or input_spec

        self._output_closed = False

        self._in_buffs = (StreamBuffer(input_spec),)

    @property
    def input_closed(self) -> bool:
        return all(in_buff.input_closed for in_buff in self._in_buffs)

    @property
    def output_closed(self) -> bool:
        return self._output_closed

    def close_input(self):
        for in_buff in self._in_buffs:
            in_buff.close_input()

    def __call__(self, *inputs: Sequence, is_last_input: bool = False) -> Sequence | Tuple[Sequence]:
        # TODO: asserts -> exceptions
        assert not self.input_closed
        assert len(inputs) == len(self._in_buffs)

        for in_buff, input_ in zip(self._in_buffs, inputs):
            in_buff.feed(input_, close_input=is_last_input)

        try:
            outputs = self._step()
        except:
            raise
        finally:
            if self.input_closed:
                self._output_closed = True

        return outputs

    # FIXME: might be better to provide buffers as arguments actually, so we can let users unpack the buffers with
    # meaningful names in the function signature.
    def _step(self) -> Sequence | Tuple[Sequence]:
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
