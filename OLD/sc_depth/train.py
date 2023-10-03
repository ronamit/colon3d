from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger

from sc_depth.config import get_opts
from sc_depth.data_modules import VideosDataModule
from sc_depth.sc_depth import SC_Depth
from sc_depth.sc_depth_v2 import SC_DepthV2
from sc_depth.sc_depth_v3 import SC_DepthV3

if __name__ == "__main__":
    hparams = get_opts()

    # pl model
    if hparams.model_version == "v1":
        system = SC_Depth(hparams)
    elif hparams.model_version == "v2":
        system = SC_DepthV2(hparams)
    elif hparams.model_version == "v3":
        system = SC_DepthV3(hparams)

    # pl data module
    dm = VideosDataModule(hparams)

    # pl logger
    logger = TensorBoardLogger(
        save_dir="ckpts",
        name=hparams.exp_name,
    )

    # save checkpoints
    ckpt_dir = f"ckpts/{hparams.exp_name}/version_{logger.version:d}"
    checkpoint_callback = ModelCheckpoint(
        dirpath=ckpt_dir,
        filename="{epoch}-{val_loss:.4f}",
        monitor="val_loss",
        mode="min",
        save_last=True,
        save_weights_only=True,
        save_top_k=3,
    )

    # restore from previous checkpoints
    if hparams.ckpt_path is not None:
        print(f"load pre-trained model from {hparams.ckpt_path}")
        system = system.load_from_checkpoint(hparams.ckpt_path, strict=False, hparams=hparams)

    # set up trainer
    trainer = Trainer(
        accelerator="gpu",
        max_epochs=hparams.num_epochs,
        limit_train_batches=hparams.epoch_size,
        limit_val_batches=200 if hparams.val_mode == "photo" else 1.0,
        num_sanity_val_steps=5,
        callbacks=[checkpoint_callback],
        logger=logger,
        benchmark=True,
    )

    trainer.fit(system, dm)
