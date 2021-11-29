import torch
import torch.optim as optim

import torchvision.ops as ops

import pytorch_lightning as pl

from nets.retina_face import RetinaFace
from eval_widerface import get_detections

class FaceDetectorTrainer(pl.LightningModule):
    def __init__(self, lr, weight_decay, score_threshold, iou_threshold, with_depth_info=False):
        super().__init__()

        self.lr = lr
        self.weight_decay = weight_decay
        self.score_threshold = score_threshold
        self.iou_threshold = iou_threshold
        self.with_depth_info = with_depth_info

        self.model = RetinaFace(with_depth_info=with_depth_info)

    def forward(self, rgb_imgs, depth_imgs=None, annotations=None):
        return self.model(rgb_imgs, depth_imgs, annotations)

    def training_step(self, batch, batch_idx):

        if not self.with_depth_info:
            rgb_imgs, _, annotations = batch

            total_classification_loss, total_bbox_reg_loss = self.forward(
                rgb_imgs=rgb_imgs,
                annotations=annotations)
        else:
            rgb_imgs, depth_imgs, annotations = batch

            total_classification_loss, total_bbox_reg_loss = self.forward(
                rgb_imgs=rgb_imgs,
                depth_imgs=depth_imgs,
                annotations=annotations)

        avg_total_classification_loss = total_classification_loss.mean()
        avg_total_bbox_reg_loss = total_bbox_reg_loss.mean()

        loss = avg_total_classification_loss + avg_total_bbox_reg_loss

        return loss

    def validation_step(self, batch, batch_idx):
        if not self.with_depth_info:
            rgb_imgs, _, annotations = batch

            total_classification_loss, total_bbox_reg_loss = self.forward(
                rgb_imgs=rgb_imgs,
                annotations=annotations)
        else:
            rgb_images, depth_imgs, annotations = batch

            total_classification_loss, total_bbox_reg_loss = self.forward(
                rgb_imgs=rgb_images,
                depth_imgs=depth_imgs,
                annotations=annotations)

            # classifications, bboxes = self.forward(
            #     rgb_imgs=rgb_images,
            #     depth_imgs=depth_imgs)
            #
            # batch_size = classifications.shape[0]
            # picked_boxes = []
            # picked_scores = []
            #
            # for i in range(batch_size):
            #     classification = torch.exp(classifications[i, :, :])
            #     bbox = bboxes[i, :, :]
            #
            #     # choose positive and scores > score_threshold
            #     scores, argmax = torch.max(classification, dim=1)
            #     argmax_indice = argmax == 0
            #     scores_indice = scores > 0.6
            #     positive_indices = argmax_indice & scores_indice
            #
            #     scores = scores[positive_indices]
            #
            #     if scores.shape[0] == 0:
            #         picked_boxes.append(None)
            #         picked_scores.append(None)
            #         continue
            #
            #     bbox = bbox[positive_indices]
            #
            #     keep = ops.boxes.nms(bbox, scores, 0.4)
            #     keep_boxes = bbox[keep]
            #     keep_scores = scores[keep]
            #     keep_scores.unsqueeze_(1)
            #     picked_boxes.append(keep_boxes)
            #     picked_scores.append(keep_scores)


        avg_total_classification_loss = total_classification_loss.mean()
        avg_total_bbox_reg_loss = total_bbox_reg_loss.mean()

        loss = avg_total_classification_loss + avg_total_bbox_reg_loss

        self.log("loss", loss)
        # return loss

        # images, annotations = batch
        #
        # cls_outs, bboxes = self.forward(img=images)
        #
        # batch_size = cls_outs.shape[0]
        # picked_boxes = []
        # picked_scores = []
        #
        # for i in range(batch_size):
        #     classification = torch.exp(cls_outs[i, :, :])
        #     bbox = bboxes[i, :, :]
        #
        #     # choose positive and scores > score_threshold
        #     scores, argmax = torch.max(classification, dim=1)
        #     argmax_indice = argmax == 0
        #     scores_indice = scores > self.score_threshold
        #     positive_indices = argmax_indice & scores_indice
        #
        #     scores = scores[positive_indices]
        #
        #     if scores.shape[0] == 0:
        #         picked_boxes.append(None)
        #         picked_scores.append(None)
        #         continue
        #
        #     bbox = bbox[positive_indices]
        #
        #     keep = ops.boxes.nms(bbox, scores, self.iou_threshold)
        #     keep_boxes = bbox[keep]
        #     keep_scores = scores[keep]
        #     keep_scores.unsqueeze_(1)
        #     picked_boxes.append(keep_boxes)
        #     picked_scores.append(keep_scores)
        #
        # recall_iter = 0.
        # precision_iter = 0.
        #
        # for j, boxes in enumerate(picked_boxes):
        #     annot_boxes = annotations[j]
        #     annot_boxes = annot_boxes[annot_boxes[:, 0] != -1]
        #
        #     if boxes is None and annot_boxes.shape[0] == 0:
        #         continue
        #     elif boxes is None and annot_boxes.shape[0] != 0:
        #         recall_iter += 0.
        #         precision_iter += 1.
        #         continue
        #     elif boxes is not None and annot_boxes.shape[0] == 0:
        #         recall_iter += 1.
        #         precision_iter += 0.
        #         continue
        #
        #     overlap = ops.boxes.box_iou(annot_boxes, boxes)
        #
        #     # compute recall
        #     max_overlap, _ = torch.max(overlap, dim=1)
        #     mask = max_overlap > 0.5
        #     detected_num = mask.sum().item()
        #     recall_iter += detected_num / annot_boxes.shape[0]
        #
        #     # compute precision
        #     max_overlap, _ = torch.max(overlap, dim=0)
        #     mask = max_overlap > 0.5
        #     true_positives = mask.sum().item()
        #     precision_iter += true_positives / boxes.shape[0]
        #
        # print(f"recall: {recall_iter / len(picked_boxes)}\t precision: {precision_iter / len(picked_boxes)}")
        # return {
        #     "recall": recall_iter / len(picked_boxes),
        #     "precision": precision_iter / len(picked_boxes)
        # }

    # def validation_step_end(self, outputs):
    #     print("DEBUG: ", outputs)
    #     tensorboard_logs = {"loss": outputs}
    #
    #     return {"val_loss": outputs, "log": tensorboard_logs}

    def test_step(self, batch, batch_idx):
        images, annotations = batch

        cls_outs, bboxes = self.forward(img=images)

        batch_size = cls_outs.shape[0]
        picked_boxes = []
        picked_scores = []

        for i in range(batch_size):
            classification = torch.exp(cls_outs[i, :, :])
            bbox = bboxes[i, :, :]

            # choose positive and scores > score_threshold
            scores, argmax = torch.max(classification, dim=1)
            argmax_indice = argmax == 0
            scores_indice = scores > self.score_threshold
            positive_indices = argmax_indice & scores_indice

            scores = scores[positive_indices]

            if scores.shape[0] == 0:
                picked_boxes.append(None)
                picked_scores.append(None)
                continue

            bbox = bbox[positive_indices]

            keep = ops.boxes.nms(bbox, scores, self.iou_threshold)
            keep_boxes = bbox[keep]
            keep_scores = scores[keep]
            keep_scores.unsqueeze_(1)
            picked_boxes.append(keep_boxes)
            picked_scores.append(keep_scores)

        recall_iter = 0.
        precision_iter = 0.

        for j, boxes in enumerate(picked_boxes):
            annot_boxes = annotations[j]
            annot_boxes = annot_boxes[annot_boxes[:, 0] != -1]

            if boxes is None and annot_boxes.shape[0] == 0:
                continue
            elif boxes is None and annot_boxes.shape[0] != 0:
                recall_iter += 0.
                precision_iter += 1.
                continue
            elif boxes is not None and annot_boxes.shape[0] == 0:
                recall_iter += 1.
                precision_iter += 0.
                continue

            overlap = ops.boxes.box_iou(annot_boxes, boxes)

            # compute recall
            max_overlap, _ = torch.max(overlap, dim=1)
            mask = max_overlap > 0.5
            detected_num = mask.sum().item()
            recall_iter += detected_num / annot_boxes.shape[0]

            # compute precision
            max_overlap, _ = torch.max(overlap, dim=0)
            mask = max_overlap > 0.5
            true_positives = mask.sum().item()
            precision_iter += true_positives / boxes.shape[0]

        print(f"recall: {recall_iter / len(picked_boxes)}\t precision: {precision_iter / len(picked_boxes)}")
        return {
            "recall": recall_iter / len(picked_boxes),
            "precision": precision_iter / len(picked_boxes)
        }

    def configure_optimizers(self):
        optimizer = optim.Adam(
            self.model.parameters(),
            lr=self.lr,
            weight_decay=self.weight_decay)

        # return optimizer

        def lr_lambda(epoch):
            if epoch < 3:
                return 0.7
            elif epoch < 7:
                return 0.01
            elif epoch < 12:
                return 0.05
            else:
                return 0.001

        scheduler = optim.lr_scheduler.LambdaLR(
            optimizer=optimizer,
            lr_lambda=lr_lambda)

        return [optimizer], [scheduler]