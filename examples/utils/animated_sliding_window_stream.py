from typing import Callable, Optional

from matplotlib.patches import ConnectionPatch, Rectangle

from torchstream import SeqSpec, Sequence, SlidingWindowParams, SlidingWindowStream
from torchstream.sliding_window.sliding_window_stream import IncorrectSlidingWindowParametersError
from torchstream.stream import NotEnoughInputError


class AnimatedSlidingWindowStream(SlidingWindowStream):
    def __init__(
        self,
        transform: Callable,
        sliding_window_params: SlidingWindowParams,
        in_spec: SeqSpec,
        out_spec: Optional[SeqSpec] = None,
    ):
        super().__init__(transform, sliding_window_params, in_spec, out_spec)

        self.step_history = []

    def _step(self, in_buff: Sequence, is_last_input: bool) -> Sequence:
        step_rec = {}
        step_rec["in_buff_start_pos"] = 0 if not self.step_history else self.step_history[-1]["in_buff_drop_pos"]
        step_rec["in_new_start_pos"] = 0 if not self.step_history else self.step_history[-1]["in_end_pos"]
        step_rec["in_end_pos"] = step_rec["in_buff_start_pos"] + in_buff.size

        # See where the output should be trimmed
        out_size, out_trim_start, out_trim_end = self.get_next_output_slice(in_buff.size, is_last_input)
        step_rec["out_start_pos"] = self.tsfm_out_pos
        step_rec.update(out_size=out_size, out_trim_start=out_trim_start, out_trim_end=out_trim_end)

        if not out_size:
            if is_last_input and self._prev_trimmed_output is not None:
                step_rec["untrimmed_output"] = self._prev_trimmed_output.copy()
                step_rec["in_buff_drop_pos"] = 0 if not self.step_history else self.step_history[-1]["in_buff_drop_pos"]
                self.step_history.append(step_rec)
                return self._prev_trimmed_output

            # TODO: breakdown current state & display how much more data is needed
            raise NotEnoughInputError(f"Input sequence of size {in_buff.size} is not enough to produce any output.")

        # Forward the input
        tsfm_out = in_buff.apply(self.transform, self.out_spec)
        if tsfm_out.size != out_size:
            raise IncorrectSlidingWindowParametersError(
                f"Sliding window parameters are not matching {self.transform}, got a {tsfm_out.size} sized "
                f"sequence instead of {out_size} for {in_buff.size} sized input."
            )
        step_rec["untrimmed_output"] = tsfm_out.copy()

        # Drop input that won't be necessary in the future. We retain only the context size rounded up to the nearest
        # multiple of the input stride.
        wins_to_drop = max(0, (in_buff.size - self.params.streaming_context_size) // self.params.stride_in)
        in_buff.drop(wins_to_drop * self.params.stride_in)
        step_rec["in_buff_drop_pos"] = step_rec["in_buff_start_pos"] + wins_to_drop * self.params.stride_in
        self.step_history.append(step_rec)

        # We've dropped past inputs, so the transform will now produce outputs starting further in the sequence
        self.tsfm_out_pos += wins_to_drop * self.params.stride_out

        # If we're trimming on the right, save the trim in case the stream closes before we can compute any
        # new sliding window output.
        self._prev_trimmed_output = tsfm_out[out_trim_end:] if out_trim_end < out_size else None

        return tsfm_out[out_trim_start:out_trim_end]

    def plot_step(self, step: int, in_ax, out_stream_ax, out_sync_ax, out_plot_fn: Callable):
        self._plot_step(
            in_ax,
            out_stream_ax,
            out_sync_ax,
            out_plot_fn,
            **self.step_history[step],
        )

    def _plot_step(
        self,
        in_ax,
        out_stream_ax,
        out_sync_ax,
        out_plot_fn: Callable,
        in_buff_start_pos: int,
        in_new_start_pos: int,
        in_buff_drop_pos: int,
        in_end_pos: int,
        out_start_pos: int,
        out_size: int,
        out_trim_start: int,
        out_trim_end: int,
        untrimmed_output: Sequence,
    ):
        fig = in_ax.get_figure()

        in_ax.axvline(in_buff_start_pos, color="purple", linestyle="-")
        in_ax.axvline(in_new_start_pos, color="blue", linestyle="--")
        in_ax.axvline(in_end_pos, color="purple", linestyle="-")

        if in_buff_start_pos != in_new_start_pos:
            x0, x1 = sorted((in_buff_start_pos, in_new_start_pos))
            y_min, y_max = in_ax.get_ylim()
            y_range = y_max - y_min if y_max != y_min else 1.0
            arrow_y = y_max - 0.05 * y_range
            in_ax.annotate(
                "",
                xy=(x0, arrow_y),
                xytext=(x1, arrow_y),
                arrowprops=dict(arrowstyle="<->", color="blue", linewidth=1.2),
                annotation_clip=False,
            )
            in_ax.text(
                (x0 + x1) / 2,
                y_max + 0.02 * y_range,
                f"overlap (={in_new_start_pos - in_buff_start_pos})",
                color="blue",
                ha="center",
                va="bottom",
                fontsize=11,
                clip_on=False,
            )

        out_plot_fn(out_stream_ax, *untrimmed_output.data)
        out_stream_ax.set_xlim(0, untrimmed_output.size)
        # Remove the ticks & border on the middle plot
        out_stream_ax.set_xticks([])
        out_stream_ax.set_yticks([])
        for spine in out_stream_ax.spines.values():
            spine.set_visible(False)
        out_stream_ax.set_xlabel("")
        out_stream_ax.set_ylabel("")

        # Align the middle plot with the input slice
        to_fig = fig.transFigure.inverted()
        start_fig_x = to_fig.transform(in_ax.transData.transform((in_buff_start_pos, 0)))[0]
        end_fig_x = to_fig.transform(in_ax.transData.transform((in_end_pos, 0)))[0]
        orig_pos = out_stream_ax.get_position()
        out_stream_ax.set_position([start_fig_x, orig_pos.y0, end_fig_x - start_fig_x, orig_pos.height])

        # Draw the trimming on the output
        trim_regions = []
        if out_size:

            def add_trim_rect(start_frac: float, end_frac: float):
                start = max(0.0, min(1.0, start_frac))
                end = max(0.0, min(1.0, end_frac))
                width = end - start
                if width <= 0:
                    return
                rect = Rectangle(
                    (start, 0),
                    width,
                    1,
                    transform=out_stream_ax.transAxes,
                    edgecolor="red",
                    facecolor=(1, 0, 0, 0.12),
                    hatch="///",
                    linewidth=1.2,
                    zorder=4,
                )
                out_stream_ax.add_patch(rect)
                trim_regions.append((start, end))

            add_trim_rect(0.0, out_trim_start / out_size)
            add_trim_rect(out_trim_end / out_size, 1.0)

            if trim_regions and not (out_trim_start == 0 and out_trim_end == out_size):
                labels = []
                if out_trim_start > 0:
                    leftmost = min(trim_regions, key=lambda region: region[0])
                    labels.append(
                        {
                            "x": leftmost[0] - 0.02,
                            "ha": "right",
                        }
                    )
                if out_trim_end < out_size:
                    rightmost = max(trim_regions, key=lambda region: region[1])
                    labels.append(
                        {
                            "x": rightmost[1] + 0.02,
                            "ha": "left",
                        }
                    )
                for label in labels:
                    text_x = max(-0.08, min(1.08, label["x"]))
                    out_stream_ax.text(
                        text_x,
                        0.5,
                        "output\nto discard",
                        transform=out_stream_ax.transAxes,
                        color="red",
                        fontsize=11,
                        ha=label["ha"],
                        va="center",
                        clip_on=False,
                    )

            if out_trim_end != out_size:
                delay_start = max(0.0, min(1.0, out_trim_end / out_size))
                delay_end = 1.0
                arrow_y = -0.11
                out_stream_ax.annotate(
                    "",
                    xy=(delay_start, arrow_y),
                    xytext=(delay_end, arrow_y),
                    xycoords=out_stream_ax.transAxes,
                    arrowprops=dict(arrowstyle="|-|", color="green", linewidth=1.2),
                    annotation_clip=False,
                )
                out_stream_ax.text(
                    1.02,
                    arrow_y + 0.025,
                    f"output delay (={out_size - out_trim_end})",
                    transform=out_stream_ax.transAxes,
                    color="green",
                    fontsize=11,
                    ha="left",
                    va="top",
                    clip_on=False,
                )

        # Connect the highlighted input boundaries to the top corners of the middle image
        input_bottom = in_ax.get_ylim()[0]
        connections = [
            (in_buff_start_pos, (0, 1)),
            (in_end_pos, (1, 1)),
        ]
        for x_value, corner in connections:
            fig.add_artist(
                ConnectionPatch(
                    xyA=(x_value, input_bottom),
                    xyB=corner,
                    coordsA="data",
                    coordsB="axes fraction",
                    axesA=in_ax,
                    axesB=out_stream_ax,
                    color="purple",
                    linestyle="--",
                )
            )
        if out_size:
            output_top = out_sync_ax.get_ylim()[1]
            connections = [
                (out_start_pos + out_trim_start, (out_trim_start / out_size, 0)),
                (out_start_pos + out_trim_end, (out_trim_end / out_size, 0)),
            ]
            for x_value, corner in connections:
                fig.add_artist(
                    ConnectionPatch(
                        xyA=(x_value, output_top),
                        xyB=corner,
                        coordsA="data",
                        coordsB="axes fraction",
                        axesA=out_sync_ax,
                        axesB=out_stream_ax,
                        color="red",
                        linestyle="--",
                    )
                )

        out_sync_ax.axvline(out_start_pos + out_trim_start, color="red", linestyle="-")
        out_sync_ax.axvline(out_start_pos + out_trim_end, color="red", linestyle="-")
