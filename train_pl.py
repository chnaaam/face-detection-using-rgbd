import numpy as np

import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint

from config import load_config
from face_detection_data_module import FaceDetectionDataModule
from face_detectior_trainer import FaceDetectorTrainer

def fix_seeds():
    torch.manual_seed(0)
    np.random.seed(0)


def main():
    fix_seeds()

    config = load_config("./config/train.cfg")

    dataset_path = config.path.dataset
    save_model_path = config.path.trained_model

    epochs = config.train.epochs
    batch_size = config.train.batch_size
    lr = float(config.train.lr)
    weight_decay = config.train.weight_decay

    gpu = config.train.gpu
    precision = config.train.precision

    with_depth_info = config.train.with_depth_info

    face_detection_data_module = FaceDetectionDataModule(
        dataset_path=dataset_path,
        batch_size=batch_size,
        with_depth_info=with_depth_info)

    face_detector_trainer = FaceDetectorTrainer(
        lr=lr,
        weight_decay=weight_decay,
        score_threshold=0.6,
        iou_threshold=0.4,
        with_depth_info=with_depth_info)

    checkpoint_callback = ModelCheckpoint(
        monitor="loss",
        dirpath=save_model_path,
        filename="retinaface-{loss:0.3f}",
        save_top_k=1,
        mode="min"
    )

    trainer = pl.Trainer(
        max_epochs=epochs,
        gpus=gpu,
        # accelerator="dp",
        # precision=precision,
        callbacks=[checkpoint_callback])

    trainer.fit(face_detector_trainer, face_detection_data_module)

if __name__ == "__main__":
    main()