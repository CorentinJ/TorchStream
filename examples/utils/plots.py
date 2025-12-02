import numpy as np
import torch


def plot_audio(
    ax,
    wave: np.ndarray,
    sample_rate: int,
    tick_label_offset_s: float = 0.0,
    with_end_tick: bool = True,
    as_scatter: bool = False,
):
    if as_scatter:
        xs = np.arange(len(wave))
        ax.scatter(xs, wave, s=6, color="tab:blue", alpha=0.7)
    else:
        ax.plot(wave)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Amplitude")
    ax.set_xlim(0, len(wave) - 1)
    ax.set_ylim(-1, 1)

    total_seconds = len(wave) / sample_rate
    if total_seconds >= 10:
        increment = 2.0
    elif total_seconds >= 5:
        increment = 1.0
    elif total_seconds >= 1:
        increment = 0.2
    else:
        increment = 0.1
    ticks_seconds = np.arange(0, total_seconds, increment)
    if with_end_tick:
        ticks_seconds = np.append(ticks_seconds, round(total_seconds, 1))
    ticks = (ticks_seconds * sample_rate).round().astype(int)
    ax.set_xticks(ticks)
    if total_seconds >= 5:
        ax.set_xticklabels([f"{ts + tick_label_offset_s:.0f}" for ts in ticks_seconds])
    else:
        ax.set_xticklabels([f"{ts + tick_label_offset_s:.1f}" for ts in ticks_seconds])


def plot_melspec(
    ax,
    spec,
    aspect="auto",
    is_log: bool = False,
    vmin: float | None = -20.0,
    vmax: float | None = 15.0,
):
    if torch.is_tensor(spec):
        spec = spec.cpu().numpy()
    if not is_log:
        spec = np.log2(spec + 1e-7)

    height, width = spec.shape
    ax.imshow(
        spec,
        aspect=aspect,
        origin="lower",
        extent=(0, width, 0, height),
        vmin=vmin,
        vmax=vmax,
    )
    ax.set_xlabel("Frames")
    ax.set_ylabel("Mel Bin")
    ax.set_xlim(0, width)
    ax.set_ylim(0, height)
