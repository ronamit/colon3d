import os
from pathlib import Path

import numpy as np
import torch
from imageio import imread, imwrite
from tqdm import tqdm

from sc_depth.config import get_opts, get_training_size
from sc_depth.datasets import custom_transforms
from sc_depth.sc_depth import SC_Depth
from sc_depth.sc_depth_v2 import SC_DepthV2
from sc_depth.sc_depth_v3 import SC_DepthV3
from sc_depth.visualization import visualize_depth


@torch.no_grad()
def main():
    hparams = get_opts()

    if hparams.model_version == "v1":
        system = SC_Depth(hparams)
    elif hparams.model_version == "v2":
        system = SC_DepthV2(hparams)
    elif hparams.model_version == "v3":
        system = SC_DepthV3(hparams)
    else:
        raise ValueError(f"Unknown model version: {hparams.model_version}")

    system = system.load_from_checkpoint(hparams.ckpt_path, strict=False)

    model = system.depth_net
    model.cuda()
    model.eval()

    # training size
    training_size = get_training_size(hparams.dataset_name)

    # normalization
    inference_transform = custom_transforms.Compose(
        [custom_transforms.RescaleTo(training_size), custom_transforms.ArrayToTensor(), custom_transforms.Normalize()],
    )

    input_dir = Path(hparams.input_dir)
    output_dir = Path(hparams.output_dir) / f"model_{hparams.model_version}"
    output_dir.makedirs_p()

    if hparams.save_vis:
        (output_dir / "vis").makedirs_p()

    if hparams.save_depth:
        (output_dir / "depth").makedirs_p()

    image_files = sum([(input_dir).files(f"*.{ext}") for ext in ["jpg", "png"]], [])
    image_files = sorted(image_files)

    print(f"{len(image_files)} images for inference")

    for _i, img_file in enumerate(tqdm(image_files)):
        filename = os.path.splitext(os.path.basename(img_file))[0]

        img = imread(img_file).astype(np.float32)
        tensor_img = inference_transform([img])[0][0].unsqueeze(0).cuda()
        pred_depth = model(tensor_img)

        if hparams.save_vis:
            vis = visualize_depth(pred_depth[0, 0]).permute(1, 2, 0).numpy() * 255
            imwrite(output_dir / f"vis/{filename}.jpg", vis.astype(np.uint8))

        if hparams.save_depth:
            depth = pred_depth[0, 0].cpu().numpy()
            np.save(output_dir / f"depth/{filename}.npy", depth)


if __name__ == "__main__":
    main()
