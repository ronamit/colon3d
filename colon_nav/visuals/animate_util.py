import io

import numpy as np
import plotly.graph_objects as go
from PIL import Image

from colon_nav.util.general_util import save_video_from_func
from colon_nav.util.rotations_util import get_identity_quaternion
from colon_nav.util.torch_util import np_func
from colon_nav.visuals.create_3d_obj import plot_xyz_axes

# --------------------------------------------------------------------------------------------------------------------


def frame_args(duration):
    # see https://github.com/plotly/plotly.js/blob/master/src/plots/animation_attributes.js
    return {
        "frame": {"duration": duration},
        "mode": "immediate",
        "fromcurrent": True,
        "transition": {"duration": duration, "easing": "linear"},
        "redraw": False,
    }


# --------------------------------------------------------------------------------------------------------------------


def create_interactive_3d_animation(
    axes_ranges,
    static_objs_list,
    per_step_objs_lists,
    title_per_step=None,
    step_duration_sec=0.05,
    layout_args=None,
):
    """
    Creates a 3D animation using plotly graph objects.
    Args:
        axes_ranges: dictionary with the axes ranges
        static_objs_list: list of plotly graph objects that are static
        per_step_objs_lists: list of lists of plotly graph objects, each list corresponds to a step
        title_per_step: list of titles for each step
        step_duration_sec: duration of each step in seconds
    Returns:
        fig: the figure object
    References:
        https://plotly.com/python/animations/
        https://plotly.com/python/visualizing-mri-volume-slices/
    """

    step_duration_ms = step_duration_sec * 1000  # [ms]

    # create the animation steps
    n_steps = len(per_step_objs_lists)

    # add the static objects to the per-step objects (put it first so the static objects will have the same indexes in all frames)
    per_step_objs_lists = [static_objs_list + objs_list for objs_list in per_step_objs_lists]

    if title_per_step is None:
        title_per_step = [f"Frame {i_step}" for i_step in range(n_steps)]

    frames = []
    for i_step in range(n_steps):
        frame_title = title_per_step[i_step]
        frame_layout = go.Layout(title_text=frame_title)
        frame_data = per_step_objs_lists[i_step]
        frame = go.Frame(data=frame_data, layout=frame_layout, name=str(i_step))
        #  note: you need to name the frame for the animation to behave properly
        frames.append(frame)

    # create the figure
    fig = go.Figure(data=per_step_objs_lists[0], frames=frames)
    if layout_args:
        fig.update_layout(**layout_args)

    # define the slider
    sliders = [
        {
            "steps": [
                {
                    "args": [[frame.name], frame_args(0)],
                    "label": str(i_step),
                    "method": "animate",
                }
                for i_step, frame in enumerate(fig.frames)
            ],
        },
    ]
    # Layout
    fig.update_layout(
        # define the size of the plot canvas
        width=600,
        height=600,
        # define fixed axes ranges, with equal aspect ratio
        xaxis={
            "range": axes_ranges["x"],
            "automargin": False,
        },
        yaxis={
            "range": axes_ranges["y"],
            "automargin": False,
        },
        scene={
            "xaxis": {"range": axes_ranges["x"]},
            "yaxis": {"range": axes_ranges["y"]},
            "zaxis": {"range": axes_ranges["z"]},
            "aspectratio": {
                "x": 0,
                "y": 0,
                "z": 0,
            },
            "aspectmode": "cube",
        },
        margin={"autoexpand": False},
        autosize=False,
        # define the animation buttons
        updatemenus=[
            {
                "buttons": [
                    {
                        "args": [None, frame_args(step_duration_ms)],
                        "label": "&#9654;",  # play symbol
                        "method": "animate",
                    },
                    {
                        "args": [[None], frame_args(0)],
                        "label": "\u23F8",  # pause symbol
                        "method": "animate",
                    },
                ],
                # define the position of the buttons
                "type": "buttons",
            },
        ],
        sliders=sliders,
    )

    return fig


# --------------------------------------------------------------------------------------------------------------------


def plotly_fig2array(fig, res_scale: float = 1.0) -> np.ndarray:
    # convert Plotly fig to  an array
    fig_bytes = fig.to_image(format="webp", scale=res_scale)
    buf = io.BytesIO(fig_bytes)
    img = Image.open(buf)
    return np.asarray(img)


# --------------------------------------------------------------------------------------------------------------------


def create_animation_video(
    axes_ranges,
    static_objs_list,
    per_step_objs_lists,
    title_per_step=None,
    step_duration_sec=0.05,
    layout_args=None,
    file_name="movie",
    save_path=None,
):
    # source: https://community.plotly.com/t/how-to-export-animation-and-save-it-in-a-video-format-like-mp4-mpeg-or/64621/2
    # create a moviepy clip from a list of plotly figure

    n_steps = len(per_step_objs_lists)

    def make_frame(t: float) -> np.ndarray:
        # converting a Plotly figure to an array
        i_step = int(t / step_duration_sec)
        if i_step >= n_steps:
            i_step = n_steps - 1

        # create the figure
        fig = go.Figure()
        if layout_args:
            fig.update_layout(**layout_args)
        # Layout
        fig.update_layout(
            # define the size of the plot canvas
            width=600,
            height=600,
            # define fixed axes ranges, with equal aspect ratio
            xaxis={
                "range": axes_ranges["x"],
                "automargin": True,
            },
            yaxis={
                "range": axes_ranges["y"],
                "automargin": True,
            },
            scene={
                "xaxis": {"range": axes_ranges["x"]},
                "yaxis": {"range": axes_ranges["y"]},
                "zaxis": {"range": axes_ranges["z"]},
                "aspectratio": {
                    "x": 0,
                    "y": 0,
                    "z": 0,
                },
                "aspectmode": "cube",
            },
            margin={"autoexpand": True},
            autosize=True,
        )
        fig.add_traces(static_objs_list + per_step_objs_lists[i_step])
        title = "Step " + str(i_step)
        if title_per_step is not None:
            title = title_per_step[i_step]
        fig.update_layout(title=title)
        fig_arr = plotly_fig2array(fig)
        del fig
        return fig_arr

    fps = 1 / step_duration_sec
    save_video_from_func(
        save_path=save_path / file_name,
        make_frame=make_frame,
        n_frames=n_steps,
        fps=fps,
    )


# --------------------------------------------------------------------------------------------------------------------
def main():
    # test the animation function

    # create the initial plot objects
    loc1 = np.array([0, 0, 0])
    rot1 = np_func(get_identity_quaternion)()

    loc2 = np.array([4, 0, 0])
    rot2 = np.array([1, 4, 3, 2])
    rot2 = rot2 / np.linalg.norm(rot2)
    obj_list2 = plot_xyz_axes(loc2, rot2)

    static_objs_list = obj_list2

    per_step_objs_lists = []
    for z in range(12):
        cur_objs_list = plot_xyz_axes(loc1 + np.array([0, 0, z]), rot1)
        per_step_objs_lists.append(cur_objs_list)

    # define the axes ranges
    axes_ranges = {"x": [-10, 10], "y": [-10, 10], "z": [-10, 10]}
    layout_args = {
        "scene": {"xaxis_title": "X [mm]", "yaxis_title": "Y [mm]", "zaxis_title": "Z [mm]"},
        "title": "World View",
        "hoverlabel": {"font_size": 12, "font_family": "Rockwell"},
    }
    fig = create_interactive_3d_animation(
        axes_ranges,
        static_objs_list,
        per_step_objs_lists,
        step_duration_sec=0.05,
        layout_args=layout_args,
    )

    # Show the plot
    fig.show()
# --------------------------------------------------------------------------------------------------------------------

if __name__ == "__main__":
    main()
# --------------------------------------------------------------------------------------------------------------------
