import numpy as np

cam1_loc = np.array([-0.433244650321488, -0.76937862321043482,  35.558685994656145])
cam2_loc = np.array([0.63307721790875437, 0.18222879384473417,  37.110857957986447])

cam1_axis_angle = np.array([0.938046385240379, -0.3274053377383993, -2.9111391384306611])
cam2_axis_angle = np.array([0.93947917724749652, 0.35651524613547175, 2.9015056387299745])

cams_dist = np.linalg.norm(cam1_loc - cam2_loc)

cam1_axis = cam1_axis_angle / np.linalg.norm(cam1_axis_angle)
print(cam1_axis)

cam2_axis = cam2_axis_angle / np.linalg.norm(cam2_axis_angle)
print(cam2_axis)

print(cams_dist)


# fish-eye camera