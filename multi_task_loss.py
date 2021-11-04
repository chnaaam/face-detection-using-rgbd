import torch
import torch.nn as nn
import torch.nn.functional as F

# from utils.box_util import calc_iou

def calc_iou(a, b):
    area = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])

    iw = torch.min(torch.unsqueeze(a[:, 2], dim=1), b[:, 2]) - torch.max(torch.unsqueeze(a[:, 0], 1), b[:, 0])
    ih = torch.min(torch.unsqueeze(a[:, 3], dim=1), b[:, 3]) - torch.max(torch.unsqueeze(a[:, 1], 1), b[:, 1])

    iw = torch.clamp(iw, min=0)
    ih = torch.clamp(ih, min=0)

    ua = torch.unsqueeze((a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1]), dim=1) + area - iw * ih

    ua = torch.clamp(ua, min=1e-8)

    intersection = iw * ih

    IoU = intersection / ua

    return IoU

class MultiTaskLoss(nn.Module):
    def __init__(self):
        super(MultiTaskLoss, self).__init__()
        self.smoothl1 = nn.SmoothL1Loss()

    def forward(self, classifications, bbox_regressions, anchors, annotations):
        anchors = anchors.to(classifications.device)
        
        batch_size = classifications.shape[0]
        classification_losses = []
        bbox_regression_losses = []

        anchor = anchors[0, :, :]
        anchor_widths = anchor[:, 2] - anchor[:, 0]
        anchor_heights = anchor[:, 3] - anchor[:, 1]
        anchor_ctr_x = anchor[:, 0] + 0.5 * anchor_widths
        anchor_ctr_y = anchor[:, 1] + 0.5 * anchor_heights

        # temp
        positive_indices_list = []

        for j in range(batch_size):
            classification = classifications[j, :, :]
            bbox_regression = bbox_regressions[j, :, :]

            annotation = annotations[j]
            # annotation = annotation[annotation[:,0] != -1]
            annotation = annotation[annotation[:, 0] > 0]
            bbox_annotation = annotation[:, :4]

            if bbox_annotation.shape[0] == 0:
                bbox_regression_losses.append(torch.tensor(0., requires_grad=True).to(classifications.device))
                classification_losses.append(torch.tensor(0., requires_grad=True).to(classifications.device))

                # temp
                positive_indices_list.append([])
                continue

            IoU = calc_iou(anchors[0, :, :], bbox_annotation[:, :4])
            # IoU, filt_iou = filt_IoU(anchors[0, :, :], bbox_annotation, ldm_annotation)

            IoU_max, IoU_argmax = torch.max(IoU, dim=1)

            # those whose iou<0.3 have no object
            negative_indices = torch.lt(IoU_max, 0.3)

            # those whose iou>0.5 have object
            positive_indices = torch.ge(IoU_max, 0.5)

            # temp
            positive_indices_list.append(positive_indices)

            num_positive_anchors = positive_indices.sum()

            # keep positive and negative ratios with 1:3
            keep_negative_anchors = num_positive_anchors * 3

            bbox_assigned_annotations = bbox_annotation[IoU_argmax, :]

            # OHEM
            negative_losses = classification[negative_indices, 1] * -1
            sorted_losses, _ = torch.sort(negative_losses, descending=True)
            if sorted_losses.numel() > keep_negative_anchors:
                sorted_losses = sorted_losses[:keep_negative_anchors]
            positive_losses = classification[positive_indices, 0] * -1

            if positive_indices.sum() > 0:
                classification_losses.append(positive_losses.mean() + sorted_losses.mean())
            else:
                classification_losses.append(torch.tensor(0., requires_grad=True).to(classifications.device))

            # compute bboxes loss
            if positive_indices.sum() > 0:
                # bbox
                bbox_assigned_annotations = bbox_assigned_annotations[positive_indices, :]

                anchor_widths_pi = anchor_widths[positive_indices]
                anchor_heights_pi = anchor_heights[positive_indices]
                anchor_ctr_x_pi = anchor_ctr_x[positive_indices]
                anchor_ctr_y_pi = anchor_ctr_y[positive_indices]

                gt_widths = bbox_assigned_annotations[:, 2] - bbox_assigned_annotations[:, 0]
                gt_heights = bbox_assigned_annotations[:, 3] - bbox_assigned_annotations[:, 1]
                gt_ctr_x = bbox_assigned_annotations[:, 0] + 0.5 * gt_widths
                gt_ctr_y = bbox_assigned_annotations[:, 1] + 0.5 * gt_heights

                targets_dx = (gt_ctr_x - anchor_ctr_x_pi) / (anchor_widths_pi + 1e-14)
                targets_dy = (gt_ctr_y - anchor_ctr_y_pi) / (anchor_heights_pi + 1e-14)
                targets_dw = torch.log(gt_widths / anchor_widths_pi)
                targets_dh = torch.log(gt_heights / anchor_heights_pi)

                bbox_targets = torch.stack((targets_dx, targets_dy, targets_dw, targets_dh))
                bbox_targets = bbox_targets.t()
                bbox_targets = bbox_targets.to(classifications.device)

                # Rescale
                bbox_targets = bbox_targets / torch.Tensor([[0.1, 0.1, 0.2, 0.2]]).to(classifications.device)

                # smooth L1
                # box losses
                bbox_regression_loss = self.smoothl1(bbox_targets, bbox_regression[positive_indices, :])
                bbox_regression_losses.append(bbox_regression_loss)
            else:
                bbox_regression_losses.append(torch.tensor(0., requires_grad=True).to(classifications.device))

        return torch.stack(classification_losses), torch.stack(bbox_regression_losses)

    # def forward(self, pred_cls_list, pred_bboxes_list, prior_boxes, true_bboxes_list):
    #
    #     prior_boxes = prior_boxes.to(pred_cls_list.device)
    #
    #     anchors = prior_boxes[0, :, :]
    #     anchor_widths = anchors[:, 2] - anchors[:, 0]
    #     anchor_heights = anchors[:, 3] - anchors[:, 1]
    #     anchor_ctr_x = anchors[:, 0] + 0.5 * anchor_widths
    #     anchor_ctr_y = anchors[:, 1] + 0.5 * anchor_heights
    #
    #     batch_size = pred_cls_list.shape[0]
    #     classification_loss = []
    #     bboxes_regression_loss = []
    #
    #     for batch_idx in range(batch_size):
    #         pred_cls = pred_cls_list[batch_idx]
    #         # pred_cls = F.log_softmax(pred_cls, dim=-1)
    #
    #         pred_bboxes = pred_bboxes_list[batch_idx]
    #         true_bboxes = true_bboxes_list[batch_idx].to(pred_cls_list.device)
    #
    #         IoU = calc_iou(prior_boxes[0, :, :], true_bboxes)
    #
    #         # Anchors are matched to a ground-truth box when IoU is larger than 0.7,
    #         # and to the background when IoU is less than 0.3
    #
    #         # 1. 각 ground truth box에 대해 가장 IoU가 높은 prior box를 뽑음
    #         max_IoU_score, max_IoU_indexes = torch.max(IoU, dim=1)
    #
    #
    #         # 2. 뽑은 prior box에서 0.7, 0.3 기준으로 positive / negative anchor을 뽑음
    #         positive_anchor_indexes = torch.ge(max_IoU_score, 0.7)
    #         negative_anchor_indexes = torch.lt(max_IoU_score, 0.3)
    #
    #         # 4.1 - Classification loss
    #         # L(p_i, pp_i)
    #         # p_i: anchor i에서 예측한 확률 값
    #         # pp_i: positive anchor = 1, negative anchor = 0
    #         num_positive_anchors = positive_anchor_indexes.sum()
    #
    #         # keep positive and negative ratios with 1:3
    #         keep_negative_anchors = num_positive_anchors * 3
    #
    #         # # OHEM
    #         negative_losses = pred_cls[negative_anchor_indexes, 1] * -1
    #         sorted_losses, _ = torch.sort(negative_losses, descending=True)
    #         if sorted_losses.numel() > keep_negative_anchors:
    #             sorted_losses = sorted_losses[:keep_negative_anchors]
    #         positive_losses = pred_cls[positive_anchor_indexes, 0] * -1
    #
    #         if positive_anchor_indexes.sum() > 0:
    #             classification_loss.append(positive_losses.mean() + sorted_losses.mean())
    #         else:
    #             classification_loss.append(torch.tensor(0., requires_grad=True).to(pred_cls.device))
    #
    #         # 4.2 Bbox loss
    #         bbox_assigned_annotations = true_bboxes[max_IoU_indexes, :]
    #
    #         # compute bboxes loss
    #         if positive_anchor_indexes.sum() > 0:
    #             # bbox
    #             bbox_assigned_annotations = bbox_assigned_annotations[positive_anchor_indexes, :]
    #
    #             anchor_widths_pi = anchor_widths[positive_anchor_indexes]
    #             anchor_heights_pi = anchor_heights[positive_anchor_indexes]
    #             anchor_ctr_x_pi = anchor_ctr_x[positive_anchor_indexes]
    #             anchor_ctr_y_pi = anchor_ctr_y[positive_anchor_indexes]
    #
    #             gt_widths = bbox_assigned_annotations[:, 2] - bbox_assigned_annotations[:, 0]
    #             gt_heights = bbox_assigned_annotations[:, 3] - bbox_assigned_annotations[:, 1]
    #             gt_ctr_x = bbox_assigned_annotations[:, 0] + 0.5 * gt_widths
    #             gt_ctr_y = bbox_assigned_annotations[:, 1] + 0.5 * gt_heights
    #
    #             targets_dx = (gt_ctr_x - anchor_ctr_x_pi) / (anchor_widths_pi + 1e-14)
    #             targets_dy = (gt_ctr_y - anchor_ctr_y_pi) / (anchor_heights_pi + 1e-14)
    #             targets_dw = torch.log(gt_widths / anchor_widths_pi)
    #             targets_dh = torch.log(gt_heights / anchor_heights_pi)
    #
    #             bbox_targets = torch.stack((targets_dx, targets_dy, targets_dw, targets_dh))
    #             bbox_targets = bbox_targets.t().to(pred_cls.device)
    #
    #             # Rescale
    #             bbox_targets = bbox_targets / torch.Tensor([[0.1, 0.1, 0.2, 0.2]]).to(pred_cls.device)
    #
    #             # smooth L1
    #             # box losses
    #             bbox_regression_loss = F.smooth_l1_loss(pred_bboxes[positive_anchor_indexes, :], bbox_targets)
    #             bboxes_regression_loss.append(bbox_regression_loss)
    #         else:
    #             bboxes_regression_loss.append(torch.tensor(0., requires_grad=True).to(pred_cls.device))
    #
    #     return torch.stack(classification_loss), torch.stack(bboxes_regression_loss)