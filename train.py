import os
from tqdm import tqdm

import torch
import torch.optim as optim

from torchvision.transforms import transforms
from torch.utils.data import DataLoader

from wider_dataset import WiderDataset, collate_fn
from nets.retina_face import RetinaFace

from multi_task_loss import MultiTaskLoss

from utils.transforms import Resize, PadToSquare

dataset_path = "../dataset/widerface"
epochs = 300
batch_size = 6
device = "cuda:1"
lr = 1e-3

train_dataset = WiderDataset(path=dataset_path, mode="train", transforms=transforms.Compose([Resize(), PadToSquare()]))
train_data_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, collate_fn=collate_fn, shuffle=True)

model = RetinaFace()
model = model.to(device)

optimizer = optim.Adam(model.parameters(), lr=lr)

criterion = MultiTaskLoss()

model.train()

for epoch in tqdm(range(epochs)):
    model.train()

    for idx, data in enumerate(train_data_loader):
        optimizer.zero_grad()

        batched_image = data[0]
        batched_targets = data[1]

        batched_image = batched_image.to(device)
        # batched_targets = batched_targets.to(device)

        cls_loss, bboxes_loss = model(batched_image, batched_targets)

        cls_loss = cls_loss.mean()
        bboxes_loss = bboxes_loss.mean()

        loss = cls_loss + bboxes_loss

        if idx % 10 == 0:
            print(f"Loss: {loss}\t Cls Loss: {cls_loss}\t Bboxes Loss: {bboxes_loss}")
            torch.save(model.state_dict(), os.path.join("model", "retinaface.pt"))

        loss.backward()
        optimizer.step()
