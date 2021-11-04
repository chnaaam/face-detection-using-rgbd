import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import skimage
import skimage.transform
from skimage import io
from PIL import Image
import cv2
import torchvision
import model
import os

import torchvision.ops as ops
import copy

def get_detections(img_batch, model, score_threshold=0.5, iou_threshold=0.7):
    model.eval()
    with torch.no_grad():
        classifications, bboxes = model(img_batch)
        batch_size = classifications.shape[0]
        picked_boxes = []
        picked_scores = []

        for i in range(batch_size):
            classification = torch.exp(classifications[i, :, :])
            bbox = bboxes[i, :, :]

            # choose positive and scores > score_threshold
            scores, argmax = torch.max(classification, dim=1)
            argmax_indice = argmax == 0
            scores_indice = scores > score_threshold
            positive_indices = argmax_indice & scores_indice

            scores = scores[positive_indices]

            if scores.shape[0] == 0:
                picked_boxes.append(None)
                picked_scores.append(None)
                continue

            bbox = bbox[positive_indices]

            keep = ops.boxes.nms(bbox, scores, iou_threshold)
            keep_boxes = bbox[keep]
            keep_scores = scores[keep]
            keep_scores.unsqueeze_(1)
            picked_boxes.append(keep_boxes)
            picked_scores.append(keep_scores)

        return picked_boxes, picked_scores

def pad_to_square(img, pad_value):
    _, h, w = img.shape
    dim_diff = np.abs(h - w)
    # (upper / left) padding and (lower / right) padding
    pad1, pad2 = dim_diff // 2, dim_diff - dim_diff // 2
    # Determine padding
    pad = (0, 0, pad1, pad2) if h <= w else (pad1, pad2, 0, 0)
    # Add padding
    img = F.pad(img, pad, "constant", value=pad_value)

    return img, pad


def resize(image, image_size):
    height, width, _ = image.shape
    long_side = max(height, width)
    scale = image_size / long_side

    # image = image.resize((int(height * image_size / long_side), int(width * image_size / long_side)))
    image = skimage.transform.resize(image, (int(height * image_size / long_side), int(width * image_size / long_side)))
    image = image * 255

    return image, scale

def get_args():
    parser = argparse.ArgumentParser(description="Detect program for retinaface.")
    # path = "../dataset/widerface/train/images/0--Parade/0_Parade_marchingband_1_849.jpg"
    path = "../dataset/widerface/val/images/0--Parade/0_Parade_Parade_0_628.jpg"
    parser.add_argument('--image_path', type=str, default=path, help='Path for image to detect')
    parser.add_argument('--model_path', type=str, help='Path for model')
    parser.add_argument('--save_path', type=str, default='./out', help='Path for result image')
    parser.add_argument('--depth', help='Resnet depth, must be one of 18, 34, 50, 101, 152', type=int, default=50)
    parser.add_argument('--scale', type=float, default=1.0, help='Image resize scale', )
    args = parser.parse_args()

    return args

def main():
    args = get_args()
    # Create torchvision model

    device = "cuda:1"

    # from face_detectior_trainer import FaceDetectorTrainer
    # model = FaceDetectorTrainer(lr=0, weight_decay=0, score_threshold=0, iou_threshold=0)
    #
    # path = "./model/retinaface.ckpt"
    # check_point = torch.load(path)
    # model.load_state_dict(check_point["state_dict"])
    # torch.save(model.model.state_dict(), "./model/retinaface.model")
    # return
    from nets.retina_face import RetinaFace
    model = RetinaFace()
    model_path = "./model/retinaface.model"
    model.load_state_dict(torch.load(model_path))

    model = model.to(device)
    model.eval()

    # Read image
    img = skimage.io.imread(args.image_path)
    origin_img = copy.deepcopy(img)
    img, resized_factor = resize(img, 640)

    img = torch.from_numpy(img)
    img = img.permute(2, 0, 1)

    input_img = img.unsqueeze(0).float().to(device)

    picked_boxes, picked_scores = get_detections(input_img, model, score_threshold=0.5, iou_threshold=0.3)

    # np_img = resized_img.cpu().permute(1,2,0).numpy()
    np_img = img.cpu().permute(1,2,0).numpy()
    np_img.astype(int)
    img = cv2.cvtColor(np_img.astype(np.uint8),cv2.COLOR_BGR2RGB)

    font = cv2.FONT_HERSHEY_SIMPLEX

    for j, boxes in enumerate(picked_boxes):
        if boxes is not None:
            for box, score in zip(boxes, picked_scores[j]):
                x1 = box[0] / resized_factor
                y1 = box[1] / resized_factor
                x2 = box[2] / resized_factor
                y2 = box[3] / resized_factor

                width = int((x2 - x1) + 0.5)
                height = int((y2 - y1) + 0.5)

                cx = int(x1.item())
                cy = int(y1.item())

                # cx = int(x1 + width / 2 + 0.5)
                # cy = int(y1 + height / 2 + 0.5)

                print(cx, cy, width, height, origin_img.shape)

                cv2.rectangle(origin_img, (x1, y1), (x2, y2), (0, 0, 255), thickness=2)
                cv2.putText(
                    origin_img,
                    text=str(score.item())[:5], org=(x1, y1),
                    fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.5,
                    thickness=1,
                    lineType=cv2.LINE_AA,
                    color=(255, 255, 255))
                cv2.rectangle(img,(box[0],box[1]),(box[2],box[3]),(0,0,255),thickness=2)

                cv2.putText(img, text=str(score.item())[:5], org=(box[0],box[1]), fontFace=font, fontScale=0.5,
                            thickness=1, lineType=cv2.LINE_AA, color=(255, 255, 255))

    image_name = args.image_path.split('/')[-1]
    save_path = os.path.join(args.save_path,image_name)
    cv2.imwrite(save_path, img)
    cv2.imshow('RetinaFace-Pytorch',img)
    cv2.waitKey()

if __name__=='__main__':
    main()
