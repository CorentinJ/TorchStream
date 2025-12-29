import inspect
import logging
import time
from textwrap import dedent

import numpy as np
import streamlit as st
import torch
from kokoro import KPipeline
from kokoro.model import KModel

from examples.streamlit_app import render_prev_next
from examples.utils.streamlit_worker import await_running_thread, run_managed_thread
from torchstream import intercept_calls
from torchstream.sliding_window.sliding_window_params import SlidingWindowParams
from torchstream.sliding_window.sliding_window_params_solver import find_sliding_window_params
from torchstream.sliding_window.sliding_window_stream import SlidingWindowStream

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


st.title("4. Streaming Kokoro TTS")

"""
In this example we will stream a full TTS pipeline from a static text input to a streaming audio output. We will be 
using the [open-source Kokoro-TTS model](https://huggingface.co/hexgrad/Kokoro-82M) by Hexgrad. Even though what 
we'll be doing is conceptually simple, this is a long read.

The challenges encountered in streaming this model are typical of what you might encounter in streaming other 
full fledged models. Hence if you get through this example, you should be well equipped to tackle streaming
other models of your own.

You can go through this demo either with CUDA or with CPU inference. These are very different performance profiles 
and both are worth considering in a project. In either case, **the point of streaming is to reduce latency**. For 
a TTS model, it means having a small and consistent Time To First Sound (TTFS), i.e. the time after which a user 
starts hearing audio playback when a request is made. 

An internal model will usually be deployed on a server with 
GPU(s), and streaming will help **shave off a couple hundred milliseconds from the TTFS**, leading to a more responsive 
experience. For models deployed on the user side, usually running on CPU, the goal is the same but **streaming can 
make the difference between a usable and an unusable experience**, as CPU inference times are often much higher.
"""


device = st.radio(
    "**Select device for inference:**",
    (
        "cuda" + (" (not available/enabled)" if not torch.cuda.is_available() else ""),
        "cpu",
    ),
    disabled=not torch.cuda.is_available(),
    index=0 if torch.cuda.is_available() else 1,
)
if not torch.cuda.is_available():
    device = "cpu"
device = torch.device(device)


st.code(
    """
from kokoro import KPipeline

pipeline = KPipeline(lang_code="en-us", repo_id="hexgrad/Kokoro-82M")
"""
)


with st.echo():
    sample_rate = 24_000

    text = (
        "[Kokoro](/kˈOkəɹO/) is an open-weight TTS model with 82 million parameters. "
        "Despite its lightweight architecture, it delivers comparable quality to "
        "larger models while being significantly faster and more cost-efficient. "
        "With Apache-licensed weights, [Kokoro](/kˈOkəɹO/) can be deployed "
        "anywhere from production environments to personal projects."
    )


# Cached as resource, but all results are disk-cached as data, so this should should not need to persist to RAM.
# TODO: clear at the end of the script execution? Add an env var to ensure this is not running?
@st.cache_resource()
def get_kokoro_pipeline(device):
    return KPipeline(lang_code="en-us", repo_id="hexgrad/Kokoro-82M", device=device)


@st.cache_data(show_time=True, persist=True)
def tts_infer(device_type: str):
    pipeline = get_kokoro_pipeline(device)

    # Do a single warmup run to get better benchmarks
    next(pipeline(text, voice="af_heart"))

    with st.echo():
        import time

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

It's not at all trivial to figure out what parts of the pipeline we want to stream here. It's easier when you've 
worked on the models you want to stream beforehand. However, with TorchStream it's possible to stream models 
with **little prior knowledge of them**, even **without modifying the source code**!

#### On to exploration!

Your best friend for snooping around neural networks is the **debugger**. By running through the major steps of the 
pipeline one by one you can figure out the ideal place to start streaming from. It also acts as a makeshift profiler¹: 
you get a sense of which parts are computationally intensive by stepping over each line. Keep in mind that the purpose 
of streaming is always to **reduce latency**, so we are looking for the hot spots in the pipeline.
"""

st.caption(
    "¹ Using a real profiler like [CProfile](https://docs.python.org/3/library/profile.html) would be ideal, "
    "but instrumenting it is out of scope for this example. TorchStream might evolve to include profiling utilities "
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
LSTMs so they need to see the full input sequence before producing any output.

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


@st.cache_data(show_time=True, persist=True)
def full_pipeline_bench(device_type: str):
    pipeline = get_kokoro_pipeline(device)

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

TorchStream comes with a little utility function for interrupting a deeply nested call immediately upon calling a 
certain function. That will let us benchmark up to the decoder call only without having to touch the source code.
"""


@st.cache_data(show_time=True, persist=True)
def partial_pipeline_bench(device_type: str):
    pipeline = get_kokoro_pipeline(device)

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
decoder_time_prop = (avg_full_time - avg_partial_time) / avg_full_time

f"""
On average, the full pipeline took **{avg_full_time:.2f}s** to run on **{device.type}**, with 
**{avg_partial_time:.2f}s** of that time being spent before the decoder. The steps after and including the decoder 
are thus responsible for **~{decoder_time_prop * 100:.0f}%** of the total inference time. 

Depending on your device, you might get a very different figure here. On **CPUs and low end GPUs** I found the decoder 
to be responsible for at least **80%** of the total inference time. In that case, if we can **stream the decoder 
alone**, we'll significantly reduce the major source of latency of this pipeline! End users will get a much shorter 
Time To First Sound (TTFS).

On a **high end GPU**, the decoder might only make up **20%** of the total inference time - which is 
already very low: around 100ms end-to-end. Kokoro TTS is a lightweight model after all! In that case, streaming the 
decoder would only lead to a small gain in latency. It won't make much of a difference to the user experience.
"""

with st.container(border=True):
    """
    You won't know the gains you'll get from streaming unless you design rigorous **benchmarks**. A superficial 
    performance analysis can be highly error inducing. Remember that results **do not translate from one device to 
    another**. 
    """

"""
#### Finding the decoder's sliding window params

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


@st.cache_data(show_time=True, persist=True)
def in_out_inspect(text: str, device_type: str):
    pipeline = get_kokoro_pipeline(device)

    with intercept_calls("kokoro.istftnet.Decoder.forward", store_in_out=True) as interceptor:
        *_, audio = next(pipeline(text, voice="af_heart"))
        (_, ref_asr, ref_f0_curve, ref_n, ref_s), _, ref_audio = interceptor.calls_in_out[0]

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

    return ref_asr, ref_f0_curve, ref_n, ref_s, ref_audio


"""
Let's try a "Hello world!" input
"""

in_out_inspect("Hello world!", device.type)


"""
For the second input let's put the longer original text (and let's store the results for later use)
"""

ref_asr, ref_f0_curve, ref_n, ref_s, ref_audio = in_out_inspect(text, device.type)
ref_audio = ref_audio.cpu().numpy().flatten()

"""
It looks like we have three sequential tensors: `asr`, `f0_curve`, `n`, and one constant tensor `s`. The sequential 
inputs have their last dimension as sequence dimensions, and both `f0_curve` and `n` have twice the time resolution 
of `asr`. We have audio as output. 

It is common when streaming to have heterogeneous inputs like this: some sequential, some constant, with different 
shapes and even different time resolutions. TorchStream handles this for you with the `SeqSpec` and `Sequence` 
classes that allow you to treat jointly these combined types as a single sequence.
"""

with st.echo():
    from functools import partial

    from torchstream import SeqSpec

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
        "Your transform outputs NaNs covering the entire output (in_size=300, out_size=180000). "
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

Again, if we don't know the model we've got to step in with the debugger to figure out why this is happening. I've 
found that the culprit is this core little module that is used several times in the model:
"""

from kokoro.istftnet import AdaIN1d

st.code(inspect.getsource(AdaIN1d), language="python")

"""
The `InstanceNorm1d` layer computes a mean and variance over the entire time dimension, then normalizes the input with 
the results. A **moving average approximation** will do the trick for streaming this layer.

But we won't implement it now, our priority is to know whether we can stream this model at all, the implementation 
can come later. So let's **patch this function to be a no-op** and return the unnormalized input as it is. The no-op
will propagate NaNs like an identity function.
"""

st.code(
    """
from torchstream.sliding_window.sliding_window_params_solver import find_sliding_window_params

with intercept_calls(
    "torch.nn.functional.instance_norm",
    handler_fn=lambda x, *args: x,
):    
    sli_params = find_sliding_window_params(
        decoder_trsfm,
        decoder_in_spec,
        decoder_out_spec,
    )[0]
"""
)
st.exception(
    RuntimeError(
        "Your transform outputs NaNs at the end of the output sequence (in_size=300, out_size=180000, "
        "out_nan_range=(73015, 180000)). This likely means that you have an autoregressive operation in your model "
        "(e.g. LSTM, cumsum, ...) that keeps producing NaNs onces it has seen one. These operations are usually "
        "trivially streamable, but you'll need to prevent their NaN propagation for the duration of the solver run, "
        "e.g. by patching them into identity functions using torchstream's intercept_calls context manager."
    )
)

"""
Strike two! This time we have an **autoregressive operation**. These are not as much of bad news as the previous one! 
Autoregressive operations are exactly streamable because their receptive field is causal: each output depends 
only on past inputs. They do propagate NaNs to all future outputs once they see one, so we will have to patch 
them out of the solver run as well. We'll build a correct stateful version of that operation later.

The culprit here is a single line in this function:
"""

from kokoro.istftnet import SineGen

source = inspect.getsource(SineGen._f02sine)
short_source = "\n".join(
    ["# function kokoro.istftnet.SineGen._f02sine", ""] + dedent(source).splitlines()[:17] + ["    ..."]
)
st.code(short_source, language="python")

"""
It's the `torch.cumsum()` call. It keeps on adding previously seen elements to a cumulated output, hence that makes 
it autoregressive. 

Let's patch it out of the solver run too. Also, this model produces large outputs for small inputs so let's bump 
up the maximum input/output size the solver can tolerate.
"""

st.code(
    """
from torchstream.sliding_window.sliding_window_params_solver import find_sliding_window_params

with intercept_calls(
    "torch.nn.functional.instance_norm",
    lambda x, *args: x,
):
    with intercept_calls("torch.cumsum", lambda x, dim: x):
        sli_params = find_sliding_window_params(
            decoder_trsfm,
            decoder_in_spec,
            decoder_out_spec,
            # We'll be dealing with long sequences, bump up the limit
            max_in_out_seq_size=1_000_000,
        )[0]
"""
)


def find_sli_params_and_print(*args, **kwargs):
    with intercept_calls("torch.nn.functional.instance_norm", lambda x, *args: x):
        with intercept_calls("torch.cumsum", lambda x, dim: x):
            sols = find_sliding_window_params(*args, **kwargs)

    logger.info("-----------------\n")
    for i, sol in enumerate(sols):
        logger.info(f"Solution #{i + 1}: {sol}")


run_managed_thread(
    func=find_sli_params_and_print,
    run_id="run1",
    job_id="kokoro_demo",
    func_kwargs=dict(
        trsfm=decoder_trsfm,
        in_spec=decoder_in_spec,
        out_spec=decoder_out_spec,
        max_in_out_seq_size=1_000_000,
    ),
    log_height=500,
)
# Await the thread here; the monkey patching needs to be undone properly.
await_running_thread()


"""
There we are! These parameters are similar to the BigVGAN ones we found earlier, and that makes sense because both 
models output audio from a low dimensional representation.

We made two operations into no-ops for the solver to run: instance norm and cumsum. Now that we have the sliding 
window parameters, we no longer need to make them no-ops, but before we get to implementing their streaming version 
we can already try out streaming as is. Sometimes, you get away without a proper implementation.
"""


@st.cache_data(show_time=True, persist=True)
def get_naive_streaming_audio():
    with st.echo():
        import math

        from torchstream import SlidingWindowParams, SlidingWindowStream

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

        decoder_input = decoder_in_spec.new_sequence_from_data(
            ref_asr,
            ref_f0_curve,
            ref_n,
        )
        stream = SlidingWindowStream(
            decoder_trsfm,
            sli_params,
            decoder_in_spec,
            decoder_out_spec,
        )

        # Let's do 10 steps of streaming, that's enough to hear boundary artifacts
        # if there are any
        n_steps = 10
        chunk_size = int(math.ceil(decoder_input.size / n_steps))
        audio_chunks = list(stream.forward_in_chunks_iter(decoder_input, chunk_size=chunk_size))
        chunk_boundaries_s = np.cumsum([chunk.size for chunk in audio_chunks]) / sample_rate
        audio = np.concatenate([chunk.data[0].cpu().numpy().flatten() for chunk in audio_chunks])

    return audio, chunk_boundaries_s


naive_streaming_audio, chunk_boundaries_s = get_naive_streaming_audio()
"""
The output:
"""
st.audio(naive_streaming_audio, sample_rate=sample_rate)
st.caption("Streaming the audio output with a completely stateless decoder")

"""
The output is decent, the streaming is working. But we can definitely hear **boundary artifacts**. They'll appear 
at some of the chunk boundaries. When you hear them, you'll see that they are at one of these timestamps:
"""
st.code(">>> chunk_boundaries_s\n" + str([round(float(t), 1) for t in chunk_boundaries_s]), language="python")

"""
We can do better than this.

### Stateful streaming
#### The autoregressive torch.cumsum()
It's not too hard to implement a stateful version of cumsum. A chunk comes in, you record the total cumulated 
sum, and when the next chunk comes in you start from that value.

That would be... if we had an implementation of streaming without redundant compute. The sliding window stream does 
buffer some of the past inputs as context, that is then fed again to the model (explained in example 1).

Let's see it:
"""
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
decoder_input = decoder_in_spec.new_sequence_from_data(ref_asr, ref_f0_curve, ref_n)


@st.cache_data(show_time=True, persist=True)
def ref_cumsum_in_out():
    with st.echo():
        with intercept_calls(
            "torch.nn.functional.instance_norm",
            lambda x, *args: x,
        ):
            with intercept_calls(
                "torch.cumsum",
                store_in_out=True,
            ) as interceptor:
                # Record the cumsum input when not streaming
                decoder_trsfm(*decoder_input.data)
                (ref_cumsum_in,), _, ref_cumsum_out = interceptor.calls_in_out[0]

                # And what it gets when streaming with chunks of size 100
                stream = SlidingWindowStream(
                    decoder_trsfm,
                    sli_params,
                    decoder_in_spec,
                    decoder_out_spec,
                )
                stream.forward_in_chunks(decoder_input, chunk_size=100)
                stream_cumsum_ins = [args[0] for args, _, _ in interceptor.calls_in_out[1:]]

                return ref_cumsum_in, ref_cumsum_out, stream_cumsum_ins


ref_cumsum_in, ref_cumsum_out, stream_cumsum_ins = ref_cumsum_in_out()

st.code(
    f"Non-streaming cumsum input shape: {tuple(ref_cumsum_in.shape)}\n"
    + "Streaming cumsum input shapes:\n    "
    + "\n    ".join(map(str, [tuple(x.shape) for x in stream_cumsum_ins]))
    + f"\n-> Concat shape of the input size seen in streaming: {tuple(torch.cat(stream_cumsum_ins, dim=1).shape)}"
)

f"""
The total amount of input seen by torch.cumsum() is _larger_ when streaming. 

The pattern of input sizes that you see is **typical of streaming a sliding window based transform**: 
- The first input chunk is provided as is (size 100 scaled by 2 due to the input resolution)
- All subsequent chunks (including the last) have a certain amount of past input as **prefix**. This amount is a 
function of the **input position and of the streaming context size**. It's often trivial to determine it.

-> In our case it's constant: `2 * sli_params.streaming_context_size(={sli_params.streaming_context_size})`

Let's verify our claims:
"""

with st.echo():
    from torchstream import Sequence

    cumsum_in_buff = Sequence(1, -1, 9, device=device)
    for i, x in enumerate(stream_cumsum_ins):
        if i == 0:
            cumsum_in_buff.feed(x)
        else:
            cumsum_in_buff.feed(x[:, sli_params.streaming_context_size * 2 :])
    max_abs_diff = torch.abs(cumsum_in_buff.data[0] - ref_cumsum_in).max().item()

st.code(f"Max difference between streaming & sync cumsum input: {max_abs_diff}")

"""
And now we can trivially implement a stateful function specific to this model's cumsum. There's only one cumsum 
call in the entire forward pass of the model, so **we don't need to disambiguate between multiple calls**. 
"""

with st.echo():

    def get_streaming_cumsum():
        # NOTE: notation abuse here, this variable will capture
        # a multidimensional tensor of values
        accum_value = 0.0

        def streaming_cumsum(x, dim, original_fn):
            nonlocal accum_value
            out = original_fn(x, dim=dim) + accum_value

            assert dim == 1
            accum_value = out[:, -2 * sli_params.streaming_context_size - 1, :]

            return out

        return streaming_cumsum


with st.container(border=True):
    """
    Manage your streaming state properly. I recommend:
    - 1 inference = 1 Stream instance = 1 state life cycle
    - Keeping your state **external** to your model (but ideally avoid this type of monkey patching in prod)
    """

"""
Let's verify that our cumsum outputs are the same when streaming. We're being very thorough here for the sake of 
the demonstration - rest assured that the final streaming implementation is very light.
"""

with st.echo():

    @st.cache_data(show_time=True, persist=True)
    def verify_cumsum_correctness(apply_cumsum_patch: bool):
        with intercept_calls(
            "torch.nn.functional.instance_norm",
            lambda x, *args: x,
        ):
            with intercept_calls(
                "torch.cumsum",
                handler_fn=get_streaming_cumsum() if apply_cumsum_patch else None,
                store_in_out=True,
                pass_original_fn=True if apply_cumsum_patch else False,
            ) as interceptor:
                chunk_size = 100
                stream = SlidingWindowStream(
                    decoder_trsfm,
                    sli_params,
                    decoder_in_spec,
                    decoder_out_spec,
                )
                stream.forward_in_chunks(decoder_input, chunk_size=chunk_size)

                diffs = []
                for i, (call_in, _, call_out) in enumerate(interceptor.calls_in_out):
                    end_idx = min((i + 1) * chunk_size * 2, ref_cumsum_out.shape[1])
                    start_idx = end_idx - call_out.shape[1]
                    abs_diff = ref_cumsum_out[:, start_idx:end_idx, :] - call_out
                    diffs.append((i, start_idx, end_idx, abs_diff.abs().max().item()))

        return diffs


@st.cache_data(show_time=True, persist=True)
def _verify_cumsum_correctness(apply_cumsum_patch: bool):
    diffs = verify_cumsum_correctness(apply_cumsum_patch)

    st.code(
        "Max differences between streaming & sync cumsum output:\n    "
        + "\n    ".join(
            f"Output chunk {i} slice[{start_idx}:{end_idx}]: {abs_diff}" for i, start_idx, end_idx, abs_diff in diffs
        )
    )


"""
Without the patch:
"""
_verify_cumsum_correctness(False)
"""
With the patch:
"""
_verify_cumsum_correctness(True)

"""
We're down to negligible differences due to numerical instability. One down, one to go!

### Streaming InstanceNorm1d

Now let's tackle instance norm. It's **impossible to exactly stream** a norm over the time dimension because it 
requires looking at the **entire sequence** to compute the mean and variance. We can however get a decent approximation 
with running estimates.

There are **multiple calls** to InstanceNorm in the model, each with **different learned parameters**. The calls at each
location need to maintain their own running stats. Because each call comes with its own InstanceNorm1d object, 
we can use them to identify which running stats to use.

Here too we should be accounting for redundant inputs, but because we're computing a running approximation of 
some statistics over the whole input, we can let it slide. Feel free to skip over the details of this implementation, 
we've already covered what's important.
"""

with st.echo():
    from torch.nn import functional as F
    from torch.nn.modules.instancenorm import InstanceNorm1d

    def get_streaming_instance_norm():
        running_stats_per_instnorm = {}

        def streaming_instance_norm(instnorm_obj: InstanceNorm1d, x):
            # Instantiate fresh running stats for each unique InstanceNorm1d object
            if instnorm_obj not in running_stats_per_instnorm:
                running_stats_per_instnorm[instnorm_obj] = (
                    # Mean
                    torch.zeros(
                        instnorm_obj.num_features,
                        device=x.device,
                        dtype=x.dtype,
                    ),
                    # Variance
                    torch.ones(
                        instnorm_obj.num_features,
                        device=x.device,
                        dtype=x.dtype,
                    ),
                )
                # Use momentum=1.0 for the first call to just set the
                # running stats to the stats of the first chunk
                momentum = 1.0
            else:
                momentum = max(
                    instnorm_obj.momentum if instnorm_obj.momentum is not None else 0.0,
                    0.1,
                )

            running_mean, running_var = running_stats_per_instnorm[instnorm_obj]

            # This is a lazy implementation exploiting instance_norm()'s behaviour.
            # With use_input_stats=True, it will update the mean and var in place
            # based on the input x. Then we discard the output.
            F.instance_norm(
                x,
                running_mean,
                running_var,
                use_input_stats=True,
                momentum=momentum,
            )

            # On the second call we do the normalization with the updated mean
            # and variance
            # This implementation has quite some flaws: it uses the unbiased
            # variance whereas the non streaming version uses the biased one, and
            # it abruptly shifts the running stats each chunk instead of smoothly
            # updating them. But this is totally sufficient for a heuristic
            return F.instance_norm(
                x,
                running_mean,
                running_var,
                instnorm_obj.weight,
                instnorm_obj.bias,
                use_input_stats=False,
            )

        return streaming_instance_norm


@st.cache_data(show_time=True, persist=True)
def get_streamed_audio():
    with st.echo():
        with intercept_calls(
            "torch.cumsum",
            handler_fn=get_streaming_cumsum(),
            pass_original_fn=True,
        ):
            with intercept_calls(
                # NOTE: different target so we can identify the
                # calls via the InstanceNorm1d objects
                "torch.nn.modules.instancenorm.InstanceNorm1d.forward",
                get_streaming_instance_norm(),
            ):
                stream = SlidingWindowStream(
                    decoder_trsfm,
                    sli_params,
                    decoder_in_spec,
                    decoder_out_spec,
                )
                streamed_audio = stream.forward_in_chunks(
                    decoder_input,
                    # Stream with an even lower chunk size to highlight any
                    # chunk boundary artifacts
                    chunk_size=40,
                )

    return streamed_audio.data[0].cpu().numpy().flatten()


"""
Now to stream it:
"""
streamed_audio = get_streamed_audio()


"""
The result:
"""
st.audio(streamed_audio, sample_rate=sample_rate)

"""
Pretty good! Maybe you can hear one or two remaining artifacts, but this is almost the original.

With only the sliding window parameters and two function patches, we've managed to stream a complex state of the art 
TTS model.

### Wrapping up

We haven't dug too deep into the topic of performance: how to write proper benchmarks, how to profile models 
quickly, how to pick the ideal chunk size for streaming, ... These topics will be covered in future examples, yet 
to be written.

For now this is the last TorchStream example. I hope you've enjoyed it. If you have questions or troubles streaming 
your own models, consider opening an issue or shouting me an email at corentin.jemine@gmail.com
"""
render_prev_next(__file__)
