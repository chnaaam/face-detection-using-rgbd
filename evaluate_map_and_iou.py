import cv2
import numpy as np
import time
import os
from tqdm import tqdm
import time
import argparse

import torch
from torchvision.ops.boxes import box_iou

def get_iou(boxA, boxB):
    """
	Calculate the Intersection over Union (IoU) of two bounding boxes.

	Parameters
	----------
	boxA = np.array( [ xmin,ymin,xmax,ymax ] )
	boxB = np.array( [ xmin,ymin,xmax,ymax ] )

	Returns
	-------
	float
		in [0, 1]
	"""

    bb1 = dict()
    bb1['x1'] = boxA[0]
    bb1['y1'] = boxA[1]
    bb1['x2'] = boxA[2]
    bb1['y2'] = boxA[3]

    bb2 = dict()
    bb2['x1'] = boxB[0]
    bb2['y1'] = boxB[1]
    bb2['x2'] = boxB[2]
    bb2['y2'] = boxB[3]

    # Determine the coordinates of the intersection rectangle
    x_left = max(bb1['x1'], bb2['x1'])
    y_top = max(bb1['y1'], bb2['y1'])
    x_right = min(bb1['x2'], bb2['x2'])
    y_bottom = min(bb1['y2'], bb2['y2'])

    if x_right < x_left or y_bottom < y_top:
        return 0.0

    # The intersection of two axis-aligned bounding boxes is always an
    # axis-aligned bounding box
    intersection_area = (x_right - x_left) * (y_bottom - y_top)

    # Compute the area of both bounding boxes area
    bb1_area = (bb1['x2'] - bb1['x1']) * (bb1['y2'] - bb1['y1'])
    bb2_area = (bb2['x2'] - bb2['x1']) * (bb2['y2'] - bb2['y1'])

    # Compute the intersection over union by taking the intersection
    # area and dividing it by the sum of prediction + ground-truth
    # areas - the intersection area
    iou = intersection_area / float(bb1_area + bb2_area - intersection_area)

    assert iou >= 0.0
    assert iou <= 1.0

    return iou


def get_gt_and_pred_data(gt_path, pred_path):
    ground_truth_bboxes = {}
    predicted_bboxes = {}

    with open(os.path.join(gt_path, f"wider_face_val_bbx_gt.txt"), "r") as fp:
        for line in fp.readlines():
            line = line.replace("\n", "")

            if line.endswith(".jpg"):
                image_file_name = line[:-4] + ".txt"

                ground_truth_bboxes.setdefault(image_file_name, [])
                continue

            datas = line.split(" ")

            if len(datas) == 1:
                continue

            bbox = [int(d) for d in datas[:4]]

            if bbox[2] > 15 and bbox[3] > 15:

                x1 = bbox[0]
                y1 = bbox[1]
                x2 = bbox[2] + bbox[0]
                y2 = bbox[3] + bbox[1]

                ground_truth_bboxes[image_file_name].append(np.array([x1, y1, x2, y2]))

        for fn, gt_bboxes in ground_truth_bboxes.items():
            if not gt_bboxes:
                continue

            with open(os.path.join(pred_path, fn), "r") as fp:
                for line in fp.readlines():
                    line = line.replace("\n", "")

                    bboxes = line.split(" ")

                    if len(bboxes) <= 1:
                        continue

                    bboxes = [int(d) for d in bboxes[:4]]

                    x1 = bboxes[0]
                    y1 = bboxes[1]
                    x2 = bboxes[2] + bboxes[0]
                    y2 = bboxes[3] + bboxes[1]

                    if not fn in predicted_bboxes:
                        predicted_bboxes.setdefault(fn, [np.array([x1, y1, x2, y2])])
                    else:
                        predicted_bboxes[fn].append(np.array([x1, y1, x2, y2]))

    return ground_truth_bboxes, predicted_bboxes

def evaluate(ground_truth_bboxes, predicted_bboxes, IoU_threshold=0.5):

    total_recall, total_precision = 0., 0.
    idx = 0
    for fn, gt_bboxes in ground_truth_bboxes.items():
        if not gt_bboxes:
            continue

        gt_bboxes = torch.tensor(gt_bboxes)

        if not fn in predicted_bboxes:
            pred_bboxes = torch.tensor([])
        else:
            pred_bboxes = torch.tensor(predicted_bboxes[fn])

            overlap = box_iou(gt_bboxes, pred_bboxes)

            # compute recall
            max_overlap, _ = torch.max(overlap, dim=1)
            mask = max_overlap > IoU_threshold
            detected_num = mask.sum().item()
            total_recall += detected_num / gt_bboxes.shape[0]

            # compute precision
            max_overlap, _ = torch.max(overlap, dim=0)
            mask = max_overlap > IoU_threshold
            true_positives = mask.sum().item()
            total_precision += true_positives / pred_bboxes.shape[0]

        idx += 1

    return total_recall / idx, total_precision / idx

def main():

    for filter in [0.01, 0.1, 0.2, 0.3, 0.5]:
        print("Filter : ", filter)

        gt_path = "../../wider_evaluation/ground_truth"
        # pred_path = "../../wider_evaluation/noise-rgb-f-" + str(filter)
        # pred_path = "../../wider_evaluation/opencv-haar"
        pred_path = "../../wider_evaluation/sp-fg-rgb-f-" + str(filter)
        # '../../wider_evaluation/evaluate_result_rgb_img\\10--People_Marching/10_People_Marching_People_Marching_10_People_Marching_People_Marching_10_People_Marching_People_Marching_10_674.txt'
        # test_path = "../../wider_evaluation/evaluate_result_rgb_img/10--People_Marching/10_People_Marching_People_Marching_10_People_Marching_People_Marching_10_People_Marching_People_Marching_10_674.txt"
        gt_bboxes, pred_bboxes = get_gt_and_pred_data(gt_path, pred_path)

        average_recall, average_precision = evaluate(ground_truth_bboxes=gt_bboxes, predicted_bboxes=pred_bboxes)

        print("Average recall: ", average_recall)
        print("Average precision: ", average_precision)

    # path = '../../wider_evaluation/evaluate_result/10--People_Marching/10_People_Marching_People_Marching_10_People_Marching_People_Marching_10_People_Marching_People_Marching_10_674.txt'
    # t_path = "../../wider_evaluation/evaluate_result/10--People_Marching/10_People_Marching_People_Marching_10_People_Marching_People_Marching_10_People_Marching_People_Marching_10_674.txt"
    # with open(
    #         t_path, 'r') as fp:
    #     for line in fp.readlines():
    #         print(line)

if __name__ == '__main__':
    main()
