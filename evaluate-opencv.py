import os
import numpy as np
import cv2
import skimage.io
from face_detection.face_detector import FaceDetector
from tqdm import tqdm
import torch
from face_detectior_trainer import FaceDetectorTrainer

if __name__ == "__main__":

    test_image_path = "../dataset/widerface/val/images"
    test_depth_image_path = "../dataset/widerface_depth/val/images"

    path = "../../venv/Lib/site-packages/cv2/data/haarcascade_frontalface_default.xml"
    face_detector = cv2.CascadeClassifier(path)

    evaluate_result_path = f"../../wider_evaluation/opencv-haar"

    if not os.path.isdir(evaluate_result_path):
        os.mkdir(evaluate_result_path)

    for group in os.listdir(test_image_path):
        test_image_group_path = os.path.join(test_image_path, group)

        evaluate_result_group_path = os.path.join(evaluate_result_path, group)

        if not os.path.isdir(evaluate_result_group_path):
            os.mkdir(evaluate_result_group_path)

        for rgb_image_fn in tqdm(os.listdir(test_image_group_path)):
            rgb_image_fn_path = os.path.join(test_image_group_path, rgb_image_fn)

            rgb_img = cv2.imread(rgb_image_fn_path)
            gray = cv2.cvtColor(rgb_img, cv2.COLOR_BGR2GRAY)

            boxes = face_detector.detectMultiScale(gray, 1.3, 5)

            evaluate_result_fn = os.path.join(evaluate_result_group_path, rgb_image_fn[:-4] + ".txt")
            with open(evaluate_result_fn, "w") as fp:

                file_name = os.path.basename(evaluate_result_fn)[:-4] + "\n"

                fp.write(file_name)

                if boxes is not None:
                    bboxe_num = str(len(boxes)) + "\n"

                    fp.write(bboxe_num)

                    for (cx, cy, width, height) in boxes:

                        line = f"{cx} {cy} {width} {height}\n"
                        fp.write(line)
                else:
                    bboxe_num = "0\n"
                    fp.write(bboxe_num)

            # for (x, y, w, h) in boxes:
            #     cv2.rectangle(rgb_img, (x, y), (x + w, y + h), (0, 0, 255), thickness=2)
            #
            # cv2.imshow("Hello", rgb_img)
            # cv2.waitKey(0)

