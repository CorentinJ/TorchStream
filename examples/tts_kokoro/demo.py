import logging
from functools import partial

import torch

from torchstream.patching.call_intercept import exit_early, intercept_calls
from torchstream.sequence.sequence import SeqSpec
from torchstream.sliding_window.sliding_window_params import SlidingWindowParams
from torchstream.sliding_window.sliding_window_params_solver import find_sliding_window_params
from torchstream.sliding_window.sliding_window_stream import SlidingWindowStream

# Setup logging to see the solver's message
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

# Change the tensor repr to show the shape and device, which saves a lot of time for debugging
old_repr = torch.Tensor.__repr__
torch.Tensor.__repr__ = lambda t: f"{tuple(t.shape)} {str(t.dtype).replace('torch.', '')} {str(t.device)} {old_repr(t)}"

# uv pip install pip
# .venv\Scripts\python.exe -m spacy download en_core_web_sm
# spacy.load("en_core_web_sm")

from kokoro import KPipeline

pipeline = KPipeline(lang_code="en-us", repo_id="hexgrad/Kokoro-82M")
device = pipeline.model.device
text = """
[Kokoro](/kˈOkəɹO/) is an open-weight TTS model with 82 million parameters. Despite its lightweight architecture, it delivers comparable quality to larger models while being significantly faster and more cost-efficient. With Apache-licensed weights, [Kokoro](/kˈOkəɹO/) can be deployed anywhere from production environments to personal projects.
"""

import soundfile as sf

# Normal, non-streaming inference
*_, audio = next(pipeline(text, voice="af_heart"))
sf.write("demo_audio.wav", audio, 24000)


# Let's analyze the shapes of the decoder's forward pass inputs and outputs
with intercept_calls("kokoro.istftnet.Decoder.forward", store_in_out=True) as interceptor:
    *_, audio = next(pipeline(text, voice="af_heart"))
    (decoder, ref_asr, ref_f0_curve, ref_n, ref_s), _, ref_audio = interceptor.calls_in_out[0]

# We have three sequential tensors: asr, f0_curve, n, and one constant tensor s. We have audio as output.
# Let's define this specification to help us with data manipulation:
decoder_in_spec = SeqSpec(
    # asr
    (1, 512, -1, device),
    # f0_curve (twice the time resolution of asr -> we put -2 to scale accordingly)
    (1, -2, device),
    # n (same as above)
    (1, -2, device),
    # s is a fixed input of size (1, 128), it does not fit as sequential data
)
# Audio is 1-dimensional, but is output with the batch & channel dimensions
decoder_out_spec = SeqSpec(1, 1, -1, device=device)
# We'll wrap the forward pass to have a function that only takes sequential inputs, in the order above
decoder_trsfm = partial(pipeline.model.decoder.forward, s=ref_s)

with intercept_calls("torch.nn.functional.instance_norm", lambda x, *args: x):
    with intercept_calls("torch.cumsum", lambda x, dim: x):
        sli_params = find_sliding_window_params(decoder_trsfm, decoder_in_spec, decoder_out_spec)[0]

# In subsequent runs we won't need to run the solver again, we can reuse the found parameters. We'll only need to
# call the solver again if we change hyperparameters or the model's architecture.
sli_params = SlidingWindowParams(
    kernel_size_in=28,
    stride_in=1,
    left_pad=14,
    right_pad=14,
    kernel_size_out=675,
    stride_out=600,
    left_out_trim=185,
    right_out_trim=490,
)

# At this point we can already get a decent streaming inference going. It won't yet be exactly the same as the
# non-streaming inference because we are not handling the statefulness of the cumsum & instance norm layers, so that
# will lead to noticeable artifacts at chunk boundaries.
decoder_input = decoder_in_spec.new_sequence_from_data(ref_asr, ref_f0_curve, ref_n)
stream = SlidingWindowStream(decoder_trsfm, sli_params, decoder_in_spec, decoder_out_spec)
audio = stream.forward_in_chunks(decoder_input, chunk_size=100).data[0]
sf.write("demo_audio_streamed.wav", audio[0, 0], 24000)


# Let's improve. There are several dozen calls to instance_norm in the model but only one early call to cumsum, so
# we will start with that easier one.
with intercept_calls("torch.nn.functional.instance_norm", lambda x, *args: x):
    with intercept_calls("torch.cumsum", store_in_out=True) as interceptor:
        # Let's see what the cumsum input looks like in the non-streaming case
        decoder_trsfm(*decoder_input.data)
        (ref_cumsum_in,), _, ref_cumsum_out = interceptor.calls_in_out[0]
        print("Non-streaming cumsum input shape:", tuple(ref_cumsum_in.shape))

        # And what it gets when streaming like earlier
        stream = SlidingWindowStream(decoder_trsfm, sli_params, decoder_in_spec, decoder_out_spec)
        stream.forward_in_chunks(decoder_input, chunk_size=120)
        stream_cumsum_ins = [args[0] for args, kwargs, out in interceptor.calls_in_out[1:]]
        print("Streaming cumsum input shapes:\n\t" + "\n\t".join(map(str, [tuple(x.shape) for x in stream_cumsum_ins])))
        print("Total cumsum input size seen in streaming:", str(sum(x.shape[1] for x in stream_cumsum_ins)))

# We see that we get larger inputs in streaming, because we provide the past context at each step. This is definitely
# something to take into consideration if we want to reproduce the same values as non-streaming inference.
# You'll notice this context size is a constant 56. It's easy to demonstrate why.
# Let's make the decoder's forward pass exit right before cumsum. Search for the sliding window parameters of this
# operation to obtain the mapping to the cumsum input.
dec_trsfm_cumsum_exit = exit_early(decoder_trsfm, target_to_exit_on="torch.cumsum", out_proc_fn=lambda x, dim: x)
cumsum_in_spec = SeqSpec(1, -1, 9, device=device)
with intercept_calls("torch.nn.functional.instance_norm", lambda x, *args: x):
    find_sliding_window_params(dec_trsfm_cumsum_exit, decoder_in_spec, cumsum_in_spec)

# The solution are parameters with an output stride and kernel of 2. Cumsum just gets our input size scaled by 2.
# Therefore, at each streaming step after the first, we get twice the context size of the full model
print(f"Full model context size * 2 = {sli_params.streaming_context_size * 2}")

# Let's verify our claims
cumsum_in_buff = cumsum_in_spec.new_empty_sequence()
for i, x in enumerate(stream_cumsum_ins):
    if i > 0:
        cumsum_in_buff.feed(x[:, sli_params.streaming_context_size * 2 :])
    else:
        cumsum_in_buff.feed(x)
print(
    "Max difference between streaming & sync cumsum input:",
    torch.abs(cumsum_in_buff.data[0] - ref_cumsum_in).max().item(),
)


# And now we can trivially implement a stateful function specific to this model's cumsum
def get_streaming_cumsum():
    accum_value = 0.0

    def streaming_cumsum(x, dim, original_fn):
        nonlocal accum_value
        out = original_fn(x, dim=dim) + accum_value

        assert dim == 1
        accum_value = out[:, -2 * sli_params.streaming_context_size - 1, :]

        return out

    return streaming_cumsum


with intercept_calls("torch.nn.functional.instance_norm", lambda x, *args: x):
    with intercept_calls(
        "torch.cumsum", handler_fn=get_streaming_cumsum(), store_in_out=True, pass_original_fn=True
    ) as interceptor:
        # NOTE: you can vary the chunk size (set at least 28) to see that the implementation still holds
        chunk_size = 120
        decoder_input.stream_apply(decoder_trsfm, sli_params, chunk_size=chunk_size, out_spec=decoder_out_spec)

        # Compare our cumsum outputs in stream vs non-streaming
        print("Max difference between streaming & sync cumsum output with the stateful cumsum:")
        for i, (call_in, _, call_out) in enumerate(interceptor.calls_in_out):
            end_idx = min((i + 1) * chunk_size * 2, ref_cumsum_out.shape[1])
            start_idx = end_idx - call_out.shape[1]
            abs_diff = ref_cumsum_out[:, start_idx:end_idx, :] - call_out
            print(f"Output chunk #{i} at indices [{start_idx}:{end_idx}] max abs diff: {abs_diff.abs().max().item()}")
