import inspect
import logging
import time
from textwrap import dedent

import numpy as np
import streamlit as st
from kokoro import KPipeline
from kokoro.model import KModel

from torchstream import intercept_calls

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


st.subheader("4. Streaming Kokoro TTS")

"""
In this example we will stream a full TTS pipeline from a static text input to a streaming audio output. We will be 
using the [open-source Kokoro-TTS model](https://huggingface.co/hexgrad/Kokoro-82M) by Hexgrad.

The challenges encountered in streaming this model are typical of what you might encounter in streaming other 
full fledged models. Hence if you get through this example, you should be well equipped to tackle streaming
other models of your own.

You can go through this demo either with CUDA or with CPU inference. These are very different performance profiles 
and both are worth considering in a project. An internal model will usually be deployed on a server with GPU(s), where 
**the point of streaming will be to reduce latency** by having a small and consistent Time To First Sound (TTFS), 
i.e. the time at which a user connected to the server starts hearing the audio playback. For models deployed on the 
user side, usually running on CPU, the goal is the same but **streaming can make the difference between a usable and 
an unusable experience**, as CPU inference times are often much higher.
"""

device = st.radio("**Select device for inference:**", ("cuda", "cpu"))


@st.cache_resource
def load_kokoro_pipeline(device: str):
    return KPipeline(lang_code="en-us", repo_id="hexgrad/Kokoro-82M", device=device)


st.code(
    """
from kokoro import KPipeline

def load_kokoro_pipeline():
    return KPipeline(lang_code="en-us", repo_id="hexgrad/Kokoro-82M")

pipeline = load_kokoro_pipeline()
"""
)

pipeline = load_kokoro_pipeline(device)
device = pipeline.model.device


with st.echo():
    sample_rate = 24_000

    text = (
        "[Kokoro](/kˈOkəɹO/) is an open-weight TTS model with 82 million parameters. "
        "Despite its lightweight architecture, it delivers comparable quality to "
        "larger models while being significantly faster and more cost-efficient. "
        "With Apache-licensed weights, [Kokoro](/kˈOkəɹO/) can be deployed "
        "anywhere from production environments to personal projects."
    )


@st.cache_data()
def tts_infer(device_type: str):
    # Do a single warmup run to get better benchmarks
    next(pipeline(text, voice="af_heart"))

    with st.echo():
        # Normal, non-streaming inference
        start_time = time.perf_counter()
        *_, wave = next(pipeline(text, voice="af_heart"))
        inference_time = time.perf_counter() - start_time

    return wave, inference_time


wave, inference_time = tts_infer(device.type)

st.audio(wave.cpu().numpy(), sample_rate=sample_rate)
st.caption(f"{len(wave) / sample_rate:.2f}s audio generated on {device.type} in {inference_time:.2f}s")


"""
#### Where to start?
We are no longer dealing with a model that takes tensors in and spits tensors out. We're dealing with a **pipeline**: 
it takes human readable text as input and produces human audible audio as output. It does so with multiple stages of 
processing, and involves two different models internally: a text to phoneme model and a phoneme to audio model. 

It's not at all trivial to figure out what we want to stream here. It's easier when you've worked on the models you 
want to stream beforehand - but with TorchStream it's possible to stream models with **little prior knowledge of them**, 
even **without modifying the source code**!

#### On to exploration!

Your best friend for snooping around neural networks is the **debugger**. By running through the major steps of the 
pipeline one by one you can figure out the ideal place to start streaming from. It also acts as a makeshift profiler¹: 
you get a sense of which parts are computationally intensive by stepping over each line. Keep in mind that the purpose 
of streaming is always to **reduce latency**, so we are looking for the hot spots in the pipeline.
"""

st.caption(
    "¹ Using a real profiler like [CProfile](https://docs.python.org/3/library/profile.html) would be ideal, "
    "but instrumenting it is out of scope for this example. Torchstream might evolve to include profiling utilities "
    "in the future."
)

"""
If you were to step into this line with a debugger:
```python
    *_, wave = next(pipeline(text, voice="af_heart"))
```

You'd eventually arrive to `kokoro.model.KModel.forward_with_tokens()`, where the main bulk of the computation is 
happening. You can tell so without any advanced profiling, because the time before entering and after exiting 
this function seems instant. What is happening in this function?
"""

st.code(dedent(inspect.getsource(KModel.forward_with_tokens)), language="python")

"""
Well, a lot. There seems to be an attention-based model involved (the `bert` line), and these often have an 
**infinite receptive field** that renders them non-streamable. There is also an LSTM layer, which is a type of 
**autoregressive** layer. Autoregressive layers are usually a green flag for streaming, but these are bidirectional 
LSTMS so they need to see the full input sequence before producing any output.

Succesfully streaming a complex transform really only goes two ways:
- You can either stream it entirely from start to finish (the case in our last 3 examples)
- Or there are operations that have a very large/infinite receptive field that you cannot stream, and **you can 
only stream after them**. These layers above likely will put us in this category.

When you think about it, it makes a lot of sense. Among the common sequential media types (text, audio, video), **text
is rarely one format that we want streaming as input**. We make phone calls and video calls but we send complete and 
finished text messages. We process audio and video in real-time, but we usually receive text in full and process it 
inside our heads at our own pace.

Even if you were to build a TTS engine that processes text as it is being typed, chances are you'd end up waiting for 
full sentences before generating anything, because the global sentence structure will matter a lot for prosody.

Back to our function. This last line:
```python
    audio = self.decoder(asr, F0_pred, N_pred, ref_s[:, :128]).squeeze()
```
is the slowest part by far, as per my manual profiling. Let's be scientific about it and see what time it takes
getting to that line vs. running the whole pipeline.

##### Running the whole pipeline
"""


@st.cache_data()
def full_pipeline_bench(device_type: str):
    with st.echo():
        n_runs = 5
        deltas = []
        for _ in range(n_runs):
            start_time = time.perf_counter()
            *_, wave = next(pipeline(text, voice="af_heart"))
            deltas.append(round(time.perf_counter() - start_time, 3))

        st.write(deltas)

    return deltas


full_times = full_pipeline_bench(device.type)

"""
##### Running only up to the decoder

Torchstream comes with a little utility function for interrupting a deeply nested call immediately upon calling a 
certain function. That will let us benchmark up to the decoder call only without having to touch the source code.
"""


@st.cache_data()
def partial_pipeline_bench(device_type: str):
    with st.echo():
        from torchstream import make_exit_early

        # NOTE: we define a wrapper because pipeline acts as a generator function, breaking make_exit_early
        def infer_one(text):
            return next(pipeline(text, voice="af_heart"))

        # Create a version of infer_one that exits early upon entering the decoder's forward pass
        early_exit_infer = make_exit_early(infer_one, "kokoro.istftnet.Decoder.forward")

        # Test it out, we should be getting the decoder's inputs arguments as the output of this function
        (decoder, *decoder_args), decoder_kwargs = early_exit_infer(text)
        st.code("\n".join(f"Arg #{i}: shape {tensor.shape}" for i, tensor in enumerate(decoder_args)))

    """
    All is well. These 4 tensors are the arguments to the decoder's forward pass:
    """
    st.code("def forward(self, asr, F0_curve, N, s):\n    ...")

    """
    Now to benchmark:
    """

    with st.echo():
        n_runs = 5
        deltas = []
        for _ in range(n_runs):
            start_time = time.perf_counter()
            early_exit_infer(text)
            deltas.append(round(time.perf_counter() - start_time, 3))
        st.write(deltas)

    return deltas


partial_times = partial_pipeline_bench(device.type)

avg_full_time = np.mean(full_times)
avg_partial_time = np.mean(partial_times)

f"""
On average, the full pipeline took **{avg_full_time:.2f}s** to run on **{device.type}**, with 
**{avg_partial_time:.2f}s** of that time being spent before the decoder. The steps after and including the decoder 
are thus responsible for **~{(avg_full_time - avg_partial_time) / avg_full_time * 100:.0f}%** of the total inference 
time. This is a dynamic script and your mileage may vary², but I have consistently seen this value above 80%.
"""

with st.container(border=True):
    """
    -> If we can **stream the decoder alone**, we'll significantly reduce the major source of latency of this pipeline! 
    End users will get a much shorter Time To First Sound (TTFS).
    """

st.caption(
    "² the sequence size should also be a factor in these benchmarks because the performance of neural networks on "
    "sequential data is usually _sublinear_ w.r.t. sequence length, especially on smaller inputs. We won't explore this "
    "further in this example."
)

"""
#### Streaming the decoder

We're streaming the decoder, so let's specify what its inputs and outputs are. We took a glance above but let's look 
again, using the convenient `intercept_calls` context manager. It acts as a passthrough by default and can store the 
inputs and outputs of each call to a target function.
"""

st.code(
    """
from torchstream import intercept_calls

with intercept_calls("kokoro.istftnet.Decoder.forward", store_in_out=True) as interceptor:
    *_, audio = next(pipeline(text, voice="af_heart"))
    (decoder, ref_asr, ref_f0_curve, ref_n, ref_s), _, ref_audio = interceptor.calls_in_out[0]
"""
)


@st.cache_data()
def in_out_inspect(text: str, device_type: str):
    with intercept_calls("kokoro.istftnet.Decoder.forward", store_in_out=True) as interceptor:
        *_, audio = next(pipeline(text, voice="af_heart"))
        (decoder, ref_asr, ref_f0_curve, ref_n, ref_s), _, ref_audio = interceptor.calls_in_out[0]

        st.code(
            f"Pipeline input text: {text}\n\n"
            "Decoder inputs:\n"
            f"- asr: {tuple(ref_asr.shape)} {str(ref_asr.dtype)} {str(ref_asr.device)}\n"
            f"- f0_curve: {tuple(ref_f0_curve.shape)} {str(ref_f0_curve.dtype)} {str(ref_f0_curve.device)}\n"
            f"- n: {tuple(ref_n.shape)} {str(ref_n.dtype)} {str(ref_n.device)}\n"
            f"- s: {tuple(ref_s.shape)} {str(ref_s.dtype)} {str(ref_s.device)}\n\n"
            "Decoder output:\n"
            f"- audio: {tuple(ref_audio.shape)} {str(ref_audio.dtype)} {str(ref_audio.device)}"
        )

    return decoder, ref_asr, ref_f0_curve, ref_n, ref_s, ref_audio


"""
Let's try a "Hello world!" input
"""

in_out_inspect("Hello world!", device.type)


"""
For the second input let's put the longer original text (and let's store the results for later use)
"""

decoder, ref_asr, ref_f0_curve, ref_n, ref_s, ref_audio = in_out_inspect(text, device.type)

"""
It looks like we have three sequential tensors: asr, f0_curve, n, and one constant tensor s. The sequential inputs 
have their last dimension as sequence dimensions, and both f0_curve and n have twice the time resolution of asr. We 
have audio as output. 

It is common when streaming to have heterogeneous inputs like this: some sequential, some constant, with different 
shapes and even different time resolutions. Torchstream handles this for you with the `SeqSpec` and `Sequence` 
classes that allow you to treat jointly these combined types as a single sequence.
"""

with st.echo():
    from functools import partial

    from torchstream.sequence.sequence import SeqSpec

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

    # Here we handle the constant input by creating a wrapper that always injects it
    decoder_trsfm = partial(decoder.forward, s=ref_s)

"""
It's the first time we're dealing with multi-tensor data for streaming so let's take a quick peek at how 
`SeqSpec` and `Sequence` make it work for us:
"""
st.code("""
>>> decoder_input = decoder_in_spec.new_sequence_from_data(ref_asr, ref_f0_curve, ref_n)
>>> decoder_input.size
830
>>> decoder_input[:10]
Sequence of size 10 with SeqSpec(
   Tensor #0: (1, 512, -1) cuda:0 torch.float32
   Tensor #1: (1, -2) cuda:0 torch.float32
   Tensor #2: (1, -2) cuda:0 torch.float32
)
>>> decoder_input[:3].data
(tensor([[[ 0.4332,  0.4332,  0.4332],
         [-0.0088, -0.0088, -0.0088],
         [-0.0069, -0.0069, -0.0069],
         ...,
         [-0.0095, -0.0095, -0.0095],
         [-0.0820, -0.0820, -0.0820],
         [-0.0030, -0.0030, -0.0030]]]), 
 tensor([[-0.0255,  0.1937,  0.2450,  0.1267,  0.1129,  0.0103]]),
 tensor([[-9.6040, -9.5364, -9.3788, -9.3498, -9.2966, -9.2976]])
)
""")

"""
It takes care of slicing the individual arrays across the right dimension and with the right scale. It lets you write 
code that is **entirely agnostic** of the input and output format, and it exposes convenience functions tailored 
for streaming (`feed()`, `drop()`, `apply()`, ...) that heavily shorten the amount of code you need to write.

Alright, let's get to streaming:
"""

st.code(
    """
from torchstream.sliding_window.sliding_window_params_solver import find_sliding_window_params

sli_params = find_sliding_window_params(
    decoder_trsfm,
    decoder_in_spec,
    decoder_out_spec,
)[0]
"""
)

st.exception(
    RuntimeError(
        "RuntimeError: Your transform outputs NaNs covering the entire output (in_size=300, out_size=180000). "
        "This likely means that an operation in your transform broadcasts an input element to all output elements, "
        "like a mean, batchnorm, etc... One solution might be to patch any such operation using torchstream's "
        "intercept_calls context manager to be an identity function for the duration of the solver run, and "
        "approximate it later for streaming."
    )
)

"""
Ouch. We've got at least one transform in our decoder that has an infinite receptive field. From this point onwards, 
we know that we will be streaming **an approximation** of the original model. But rest assured that the approximation 
can be very good.

"""


quit()

# Consistent TTFS compared to sync

import logging
from functools import partial

from torch.nn import InstanceNorm1d
from torch.nn import functional as F

from torchstream.patching.call_intercept import intercept_calls, make_exit_early
from torchstream.sequence.sequence import SeqSpec
from torchstream.sliding_window.sliding_window_params import SlidingWindowParams
from torchstream.sliding_window.sliding_window_stream import SlidingWindowStream
from kokoro import KModel

with intercept_calls("torch.nn.functional.instance_norm", lambda x, *args: x):
    with intercept_calls("torch.cumsum", lambda x, dim: x):
        sli_params = find_sliding_window_params(
            decoder_trsfm,
            decoder_in_spec,
            decoder_out_spec,
            # We'll be dealing with long sequences, bump up the limit
            max_in_out_seq_size=1_000_000,
        )[0]

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
audio = stream.forward_in_chunks(decoder_input, chunk_size=40).data[0].cpu().numpy()
sf.write("demo_audio_streamed_v2.wav", audio[0, 0], 24000)


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
        stream.forward_in_chunks(decoder_input, chunk_size=40)
        stream_cumsum_ins = [args[0] for args, kwargs, out in interceptor.calls_in_out[1:]]
        print("Streaming cumsum input shapes:\n\t" + "\n\t".join(map(str, [tuple(x.shape) for x in stream_cumsum_ins])))
        print("Total cumsum input size seen in streaming:", str(sum(x.shape[1] for x in stream_cumsum_ins)))

# We see that we get larger inputs in streaming, because we provide the past context at each step. This is definitely
# something to take into consideration if we want to reproduce the same values as non-streaming inference.
# You'll notice this context size is a constant 56. It's easy to demonstrate why.
# Let's make the decoder's forward pass exit right before cumsum. Search for the sliding window parameters of this
# operation to obtain the mapping to the cumsum input.
dec_trsfm_cumsum_exit = make_exit_early(decoder_trsfm, target_to_exit_on="torch.cumsum", out_proc_fn=lambda x, dim: x)
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
        chunk_size = 40
        decoder_input.stream_apply(decoder_trsfm, sli_params, chunk_size=chunk_size, out_spec=decoder_out_spec)

        # Compare our cumsum outputs in stream vs non-streaming
        print("Max difference between streaming & sync cumsum output with the stateful cumsum:")
        for i, (call_in, _, call_out) in enumerate(interceptor.calls_in_out):
            end_idx = min((i + 1) * chunk_size * 2, ref_cumsum_out.shape[1])
            start_idx = end_idx - call_out.shape[1]
            abs_diff = ref_cumsum_out[:, start_idx:end_idx, :] - call_out
            print(f"Output chunk #{i} at indices [{start_idx}:{end_idx}] max abs diff: {abs_diff.abs().max().item()}")


# Now let's tackle instance norm. It's impossible to exactly stream a norm over the time dimension because it requires
# looking at the entire sequence to compute the mean and variance. We can however get a decent approximation with
# running estimates.
# There are multiple calls to InstanceNorm in the model, each with different learned parameters. The calls at each
# location need to maintain their own running stats.
def get_streaming_instance_norm():
    running_stats_per_instnorm = {}

    def streaming_instance_norm(instnorm_obj: InstanceNorm1d, x):
        # Instantiate fresh running stats for each unique InstanceNorm1d object
        if instnorm_obj not in running_stats_per_instnorm:
            running_stats_per_instnorm[instnorm_obj] = (
                # Mean
                torch.zeros(instnorm_obj.num_features, device=x.device, dtype=x.dtype),
                # Variance (NOTE: both could be initialized from the means & vars of a couple of inputs to be
                # more accurate)
                torch.ones(instnorm_obj.num_features, device=x.device, dtype=x.dtype),
            )
            # Use momentum=1.0 for the first call to just set the running stats to the stats of the first chunk
            momentum = 1.0
        else:
            momentum = max(instnorm_obj.momentum if instnorm_obj.momentum is not None else 0.0, 0.1)

        running_mean, running_var = running_stats_per_instnorm[instnorm_obj]

        # This is a lazy implementation: we do a first pass of instance_norm just to compute the running mean/var
        # (in place) and we discard the output
        # This implementation has quite some flaws: it uses the unbiased variance whereas the non streaming version
        # uses the biased one, and it abruptly shifts the running stats each chunk instead of smoothly updating them.
        # But this is totally sufficient for a heuristic
        F.instance_norm(
            x,
            running_mean,
            running_var,
            None,
            None,
            use_input_stats=True,
            momentum=momentum,
            eps=instnorm_obj.eps,
        )

        # Now we actually do the normalization
        return F.instance_norm(
            x,
            running_mean,
            running_var,
            instnorm_obj.weight,
            instnorm_obj.bias,
            use_input_stats=False,
            eps=instnorm_obj.eps,
        )

    return streaming_instance_norm


# Let's listen to see how we've improved
with intercept_calls("torch.cumsum", handler_fn=get_streaming_cumsum(), pass_original_fn=True):
    with intercept_calls(
        # NOTE: different target than before because we could not identify the calling instance with F.instance_norm()
        "torch.nn.modules.instancenorm.InstanceNorm1d.forward",
        get_streaming_instance_norm(),
    ):
        audio = decoder_input.stream_apply(decoder_trsfm, sli_params, chunk_size=40, out_spec=decoder_out_spec)
        sf.write("demo_audio_streamed_v3.wav", audio.data[0][0, 0].cpu().numpy(), 24000)
