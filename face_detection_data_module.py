from torch.utils.data import DataLoader

import pytorch_lightning as pl
from torchvision.transforms import transforms

from wider_dataset import WiderDataset, collate_fn

from utils.transforms import Resize, PadToSquare

class FaceDetectionDataModule(pl.LightningDataModule):
    def __init__(self, dataset_path, batch_size, with_depth_info=False):

        super().__init__()
        self.batch_size = batch_size
        self.train_dataset = WiderDataset(
            path=dataset_path,
            mode="train",
            transforms=transforms.Compose([Resize(), PadToSquare()]),
            with_depth_info=with_depth_info)

        self.valid_dataset = WiderDataset(
            path=dataset_path,
            mode="val",
            transforms=transforms.Compose([Resize(), PadToSquare()]),
            with_depth_info=with_depth_info)

    def train_dataloader(self):
        return DataLoader(
            dataset=self.train_dataset,
            batch_size=self.batch_size,
            num_workers=8,
            collate_fn=collate_fn,
            shuffle=True,
            pin_memory=True)

    def val_dataloader(self):
        return DataLoader(
            dataset=self.valid_dataset,
            batch_size=self.batch_size,
            num_workers=8,
            collate_fn=collate_fn,
            shuffle=False,
            pin_memory=True)