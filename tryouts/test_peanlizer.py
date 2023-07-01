import matplotlib.pyplot as plt
import numpy as np
import torch

from colon3d.utils.torch_util import SoftConstraints

# --------------------------------------------------------------------------------------------------------------------

if __name__ == "__main__":
    # Test code:
    penalizer = SoftConstraints(
        {
            "my_up_bound": {"lim_type": "upper", "lim": 100, "margin_ratio": 0.1, "barrier_jump": 30},
            "my_low_bound": {
                "lim_type": "lower",
                "lim": 10,
                "barrier_jump": 20,
            },
        },
    )
    n_points = 1000
    grid = np.linspace(start=0, stop=110, num=n_points)
    grid = torch.from_numpy(grid).float().to(penalizer.device)
    up_pen = torch.zeros_like(grid)
    low_pen = torch.zeros_like(grid)
    for i in range(n_points):
        up_pen[i] = penalizer.get_penalty(constraint_name="my_up_bound", val=grid[i])
        low_pen[i] = penalizer.get_penalty(constraint_name="my_low_bound", val=grid[i])
    up_pen = up_pen.cpu().numpy()
    low_pen = low_pen.cpu().numpy()
    grid = grid.cpu().numpy()
    fig = plt.figure()
    plt.plot(grid, up_pen, color="r", label="Upper bound penalty")
    plt.plot(grid, low_pen, color="b", label="Lower bound penalty")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.legend()
    plt.show()
