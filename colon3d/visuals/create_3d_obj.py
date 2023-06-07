import numpy as np
import plotly.graph_objects as go

from colon3d.rotations_util import get_identity_quaternion, invert_rotation, rotate_points
from colon3d.torch_util import np_func

# --------------------------------------------------------------------------------------------------------------------


def plot_3d_arrow(origin, vector, label, color, arrow_len: float = 1):
    vector = np.array(vector)
    origin = np.array(origin)
    vec_norm = np.linalg.norm(vector)
    unit_vec = vector / vec_norm
    # Compute the endpoint of the arrow
    endpoint = origin + unit_vec * arrow_len
    # Define the arrow as a Plotly scatter trace
    arrow_line = go.Scatter3d(
        x=[origin[0], endpoint[0]],
        y=[origin[1], endpoint[1]],
        z=[origin[2], endpoint[2]],
        mode="lines",
        name=label,
        line={"color": color, "width": 2},
        showlegend=False,
    )
    # Compute the cone coordinates
    cone_height_ratio = 0.1  # ratio of cone height to arrow length
    cone_base = origin + unit_vec * arrow_len * (1 - cone_height_ratio)
    arrow_tip = go.Cone(
        x=[cone_base[0]],
        y=[cone_base[1]],
        z=[cone_base[2]],
        u=[unit_vec[0] * arrow_len],
        v=[unit_vec[1] * arrow_len],
        w=[unit_vec[2] * arrow_len],
        colorscale=[[0, color], [1, color]],
        sizeref=cone_height_ratio * arrow_len,
        showscale=False,
        anchor="tail",
        name=label,
        showlegend=False,
    )
    arrow_objs = [arrow_line, arrow_tip]
    return arrow_objs


# --------------------------------------------------------------------------------------------------------------------
# --------------------------------------------------------------------------------------------------------------------


def plot_xyz_axes(origin_loc: np.array, rot_quat: np.array, arrow_len: float = 1, add_legend: bool = True):
    """Return plot objects for the XYZ axes at origin point with rotation according to a unit quaternion"""

    # arrows endpoints in camera coordinate system:
    axes_vecs = np.eye(3, dtype=np.float64) # 3 x 3 identity matrix
    
    # rotate the axes vectors according to the camera rotation.
    # (inverse rotation of the camera is applied to the axes vectors)
    inv_cam_rot = np_func(invert_rotation)(rot_quat)
    x_unit_vec = np_func(rotate_points)(axes_vecs[0], inv_cam_rot).squeeze()
    y_unit_vec = np_func(rotate_points)(axes_vecs[1], inv_cam_rot).squeeze()
    z_unit_vec = np_func(rotate_points)(axes_vecs[2], inv_cam_rot).squeeze()
    x_arrow_objs = plot_3d_arrow(origin_loc, x_unit_vec, "X", "red", arrow_len)
    y_arrow_objs = plot_3d_arrow(origin_loc, y_unit_vec, "Y", "green", arrow_len)
    z_arrow_objs = plot_3d_arrow(origin_loc, z_unit_vec, "Z", "blue", arrow_len)
    axes3d_objs = x_arrow_objs + y_arrow_objs + z_arrow_objs

    # add custom legend handles
    if add_legend:
        axes3d_legend_handles = [
            go.Scatter(
                x=[None],
                y=[None],
                mode="markers",
                name="X (Cam)",
                marker={"size": 7, "color": "red", "symbol": "triangle-up"},
            ),
            go.Scatter(
                x=[None],
                y=[None],
                mode="markers",
                name="Y (Cam)",
                marker={"size": 7, "color": "green", "symbol": "triangle-up"},
            ),
            go.Scatter(
                x=[None],
                y=[None],
                mode="markers",
                name="Z (Cam)",
                marker={"size": 7, "color": "blue", "symbol": "triangle-up"},
            ),
        ]
        axes3d_objs += axes3d_legend_handles

    return axes3d_objs


# --------------------------------------------------------------------------------------------------------------------
def plot_fov_cone(
    cone_height: float = 1000,
    fov_deg: float = 120,
    cam_loc=None,
    cam_rot=None,
    label="FOV",
    verbose=False,
    add_legend=True,
):
    """Returns a 3D objects that represents a camera's FOV cone
    by default, it sets the cone apex at (0,0,0) and looking at direction (0, 0, 1)."""
    n_circle_points = 90  # number of points on the circle
    # index zero is (0,0,0) the rest are on a circle at z=z_depth, with radius = z_depth * tan(FOV/2)
    points = np.zeros((n_circle_points + 1, 3))
    points[1:, 2] = cone_height  # set z values for all points besides the tip
    theta = np.linspace(0, 2 * np.pi, num=n_circle_points)
    fov_rad = np.deg2rad(fov_deg)
    radius = cone_height * np.tan(fov_rad / 2)
    # set x,y values for all points besides the tip
    points[1:, 0] = radius * np.cos(theta)
    points[1:, 1] = radius * np.sin(theta)
    if cam_loc is None:
        cam_loc = np.array([0, 0, 0])
    if cam_rot is None:
        cam_rot = np_func(get_identity_quaternion)()
    if verbose:
        cam_forward = np_func(rotate_points)(np.array([0, 0, 1], dtype=np.float64), cam_rot)
        print(
            f"Plotted camera location: {cam_loc} [mm], direction vector: {cam_forward}, FOV: {fov_deg:.1f} [deg]",
        )
    # translate and rotate the cone points:
    inv_cam_rot = np_func(invert_rotation)(cam_rot)
    points = np_func(rotate_points)(points, inv_cam_rot) + cam_loc
    cam_cone = go.Mesh3d(
        x=points[:, 0],
        y=points[:, 1],
        z=points[:, 2],
        i=np.zeros(n_circle_points),  # all triangles are connected to the apex (index 0)
        j=np.arange(1, n_circle_points + 1),  # triangle are connected to the next point in the circle
        k=np.roll(np.arange(1, n_circle_points + 1), -1),  # cyclic shift, so the last point is connected to the first
        opacity=0.3,
        color="lightpink",
        name=label,
        showlegend=False,
    )
    apex_point = go.Scatter3d(
        mode="markers",
        x=[points[0, 0]],
        y=[points[0, 1]],
        z=[points[0, 2]],
        marker={"color": "blue", "size": 1, "opacity": 0.6},
        showlegend=False,
    )
    axes3d_objs1 = plot_xyz_axes(origin_loc=cam_loc, rot_quat=cam_rot, arrow_len=5, add_legend=add_legend)
    axes3d_objs2 = plot_xyz_axes(origin_loc=cam_loc, rot_quat=cam_rot, arrow_len=5, add_legend=False)
    # Return the Plotly objects (I added the axes3d_objs twice, since they sometimes disappeared from the plot)
    fig_data = [*axes3d_objs1, cam_cone, apex_point, *axes3d_objs2]
    return fig_data


# --------------------------------------------------------------------------------------------------------------------


def plot_cam_and_point_cloud(
    points3d: np.ndarray,
    cam_pose: np.ndarray,
    cam_fov_deg: float = 120,
    verbose=False,
    show_fig=False,
    save_path=None,
):
    """Plot a 3D point cloud and a camera's FOV cone
    Args:
        points3d: the point cloud to plot  - Nx3 array of 3D points (in world coordinates)
        cam_pose: 1x7 array of camera pose (location and rotation quaternion) (in world coordinates)
        cam_fov_deg: camera's FOV in degrees
        verbose: print camera location and direction vector
        show_fig: show the figure
        save_path: save the figure to this path
    """
    if cam_pose is None:
        # if no camera pose is given, assume the camera is at (0,0,0) and looking at (0,0,1)
        cam_pose = np.concatenate((np.zeros(3), np_func(get_identity_quaternion)()))
    assert points3d.ndim == 2
    assert points3d.shape[1] == 3
    cam_loc = cam_pose[:3]
    cam_rot = cam_pose[3:]
    # get the z-depth of each point in the camera's coordinate system
    # inv_cam_rot = np_func(invert_rotation)(cam_rot[np.newaxis, :])
    point3_in_cam_sys = np_func(rotate_points)(points3d - cam_loc, cam_rot)
    points_z_depth = point3_in_cam_sys[:, 2]
    # plot the point cloud (in world system)
    fig_data = [
        go.Scatter3d(
            x=points3d[:, 0],
            y=points3d[:, 1],
            z=points3d[:, 2],
            mode="markers",
            marker={
                "size": 1,
                "color": points_z_depth,  # set color to an array/list of desired values
                "colorscale": "Viridis",  # choose a colorscale
                "opacity": 0.8,
            },
        ),
    ]
    fov_cone = plot_fov_cone(
        cone_height=np.max(points_z_depth),
        fov_deg=cam_fov_deg,
        cam_loc=cam_loc,
        cam_rot=cam_rot,
        label="Alg. FOV",
        verbose=verbose,
        add_legend=True,
    )
    fig_data += fov_cone
    fig = go.Figure(
        data=fig_data,
    )
    # tight layout
    fig.update_layout(
        margin={"l": 0, "r": 0, "b": 0, "t": 0},
        scene={"xaxis_title": "X [mm]", "yaxis_title": "Y [mm]", "zaxis_title": "Z [mm]"},
    )
    if save_path:
        fig.write_html(save_path)
        print("Saved figure at ", save_path)
    if show_fig:
        fig.show()


# --------------------------------------------------------------------------------------------------------------------
# --------------------------------------------------------------------------------------------------------------------

if __name__ == "__main__":
    # Define the origin and vector of the arrow
    origin_example = np.array([0, 0, 0])
    rot_quat_example = np_func(get_identity_quaternion)()

    # Plot the arrows
    objs = plot_xyz_axes(origin_example, rot_quat_example)

    # Create a Plotly figure containing the arrow and the cone
    fig = go.Figure(data=objs)

    # Set the axis labels and ranges
    fig.update_layout(
        scene={
            "xaxis_title": "X",
            "yaxis_title": "Y",
            "zaxis_title": "Z",
        },
    )

    # Show the plot
    fig.show()
# --------------------------------------------------------------------------------------------------------------------
