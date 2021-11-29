import os
import numpy as np
import torch
from PIL import Image
import skimage.io
from torch.utils.data import Dataset

def collate_fn(data):


    if len(data) == 2:
        batched_img = []
        batched_annotations = []

        for img, annotations in data:
            batched_img.append(img.permute(2, 0, 1).float())
            batched_annotations.append(annotations.float())

        return torch.stack(batched_img), batched_annotations

    else:
        batched_rgb_img = []
        batched_depth_img = []
        batched_annotations = []

        for rgb_img, depth_img, annotations in data:
            batched_rgb_img.append(rgb_img.permute(2, 0, 1).float())

            if depth_img:
                batched_depth_img.append(depth_img.permute(2, 0, 1).float())

            batched_annotations.append(annotations.float())

        if batched_depth_img:
            return torch.stack(batched_rgb_img), torch.stack(batched_depth_img), batched_annotations
        else:
            return torch.stack(batched_rgb_img), None, batched_annotations

class WiderDataset(Dataset):
    def __init__(self, path, mode="val", transforms=None, with_depth_info=False):
        self.dataset_path = os.path.join(path, mode)
        self.depth_dataset_path = os.path.join(path + "_depth", mode)
        self.transforms = transforms
        self.img_name_list, self.bboxes_list = self.load_dataset()
        self.with_depth_info = with_depth_info

    def load_dataset(self):
        img_name_list, bboxes_list = [], []

        annotations = []

        with open(os.path.join(self.dataset_path, "label.txt"), "r") as fp:
            for line in fp.readlines():
                line = line.replace('\n', '')

                if line.startswith("#"):
                    if annotations:
                        img_name_list.append(file_name)
                        bboxes_list.append(annotations)

                        annotations = []

                    file_name = line.split(" ")[1]

                    continue

                points = line.split(" ")[:4]

                cx = float(points[0])
                cy = float(points[1])
                width = float(points[2])
                height = float(points[3])

                annotations.append((cx, cy, width, height))

        return img_name_list, bboxes_list

    def __len__(self):
        return len(self.img_name_list)

    def __getitem__(self, idx, debug=False):
        # For debugging
        # idx = 0

        img_name = self.img_name_list[idx]
        # print(img_name)

        if not self.with_depth_info:
            img = skimage.io.imread(os.path.join(self.dataset_path, "images", img_name))
            bboxes = self.bboxes_list[idx]

            targets = []
            for bbox in bboxes:
                x1 = bbox[0]
                y1 = bbox[1]
                x2 = bbox[0] + bbox[2]
                y2 = bbox[1] + bbox[3]

                targets.append([x1, y1, x2, y2])

            # For debugging
            if debug:
                print("Image name: ", img_name)

                self.show_image(img, targets)
                return

            if self.transforms:
                img, targets = self.transforms((img, np.array(targets)))

            return img, None, targets

        else:
            rgb_img = skimage.io.imread(os.path.join(self.dataset_path, "images", img_name))
            depth_img = np.load(os.path.join(self.depth_dataset_path, "images", img_name[:-4] + ".npy"))
            bboxes = self.bboxes_list[idx]

            targets = []
            for bbox in bboxes:
                x1 = bbox[0]
                y1 = bbox[1]
                x2 = bbox[0] + bbox[2]
                y2 = bbox[1] + bbox[3]

                targets.append([x1, y1, x2, y2])

            if self.transforms:
                rgb_img, depth_img, targets = self.transforms((rgb_img, depth_img, np.array(targets)))

            return rgb_img, depth_img, targets

    # For debugging methods
    def show_image(self, img, targets):
        import numpy as np
        from PIL import ImageDraw
        import matplotlib.pyplot as plt

        draw = ImageDraw.Draw(img)

        for x1, y1, x2, y2 in targets:
            print(f"[{x1}, {y1}, {x2}, {y2}],")
            draw.rectangle(((x1, y1), (x2, y2)), outline=(0, 0, 255), width=4)

        numpy_img = np.array(img)
        plt.imshow(numpy_img)
        plt.show()

if __name__ == "__main__":
    from torchvision import transforms
    from utils.transforms import Resize, PadToSquare

    wider_dataset = WiderDataset(
        path="../dataset/widerface",
        transforms=transforms.Compose([Resize(), PadToSquare()]),
        with_depth_info=True)

    print(wider_dataset.__len__())
    # wider_dataset.__getitem__(7, debug=True)