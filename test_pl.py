import numpy as np

import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from face_detection_data_module import FaceDetectionDataModule
from face_detectior_trainer import FaceDetectorTrainer

def main():
    torch.manual_seed(0)
    np.random.seed(0)

    dataset_path = "../dataset/widerface"
    batch_size = 12
    lr = 1e-3
    weight_decay = 0.001
    save_model_path = "./model"
    epochs = 300
    with_depth_info = True

    face_detection_data_module = FaceDetectionDataModule(
        dataset_path=dataset_path,
        batch_size=batch_size,
        with_depth_info=with_depth_info)

    face_detector_trainer = FaceDetectorTrainer(
        lr=lr,
        weight_decay=weight_decay,
        score_threshold=0.5,
        iou_threshold=0.5,
        with_depth_info=with_depth_info)

    path = "model/retinaface-depth-loss=1.252_backup.ckpt"
    check_point = torch.load(path)
    face_detector_trainer.load_state_dict(check_point["state_dict"])

    checkpoint_callback = ModelCheckpoint(
        monitor="loss",
        dirpath=save_model_path,
        filename=f"retinaface-" if not with_depth_info else "retinaface-depth-" + "{loss:0.3f}",
        save_top_k=2,
        mode="min"
    )

    trainer = pl.Trainer(
        max_epochs=epochs,
        gpus=-1,
        accelerator="dp",
        precision=16,
        callbacks=[checkpoint_callback])

    trainer.test(face_detector_trainer, face_detection_data_module.val_dataloader)

if __name__ == "__main__":
    main()