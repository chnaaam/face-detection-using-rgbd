import os
import numpy as np
import cv2
import skimage.io
from face_detection.face_detector import FaceDetector
from tqdm import tqdm
import torch
from face_detectior_trainer import FaceDetectorTrainer

def noisy(amount,image):
    row,col,ch = image.shape
    s_vs_p = 0.5

    out = np.copy(image)
    # Salt mode
    num_salt = np.ceil(amount * image.size * s_vs_p)
    coords = [np.random.randint(0, i - 1, int(num_salt))
      for i in image.shape]
    out[coords] = 1

    # Pepper mode
    num_pepper = np.ceil(amount* image.size * (1. - s_vs_p))
    coords = [np.random.randint(0, i - 1, int(num_pepper))
      for i in image.shape]
    out[coords] = 0
    return out


if __name__ == "__main__":

    test_image_path = "../dataset/widerface/val/images"
    test_depth_image_path = "../dataset/widerface_depth/val/images"

    face_detector = FaceDetector(model_path="../retinaface/model/retinaface.model", with_depth_info=False)

    # for filter in tqdm([1, 3, 15]):
    for filter in tqdm([0.01, 0.1, 0.2, 0.3, 0.5]):
        evaluate_result_path = f"../../wider_evaluation/sp-fg-rgb-f-{filter}"

        if not os.path.isdir(evaluate_result_path):
            os.mkdir(evaluate_result_path)

        for group in os.listdir(test_image_path):
            test_image_group_path = os.path.join(test_image_path, group)

            evaluate_result_group_path = os.path.join(evaluate_result_path, group)

            if not os.path.isdir(evaluate_result_group_path):
                os.mkdir(evaluate_result_group_path)

            for rgb_image_fn in tqdm(os.listdir(test_image_group_path)):
                rgb_image_fn_path = os.path.join(test_image_group_path, rgb_image_fn)

                rgb_img = skimage.io.imread(rgb_image_fn_path)

                # Blurring
                # rgb_img = cv2.blur(rgb_img, (filter, filter))
                rgb_img = noisy(filter, rgb_img)

                rgb_img = cv2.medianBlur(rgb_img, 5)
                # rgb_img = cv2.GaussianBlur(rgb_img, (5,5), 0)

                # face_detector.predict(
                #     rgb_img,
                #     depth_img=None,
                #     evaluate=True,
                #     img_name=rgb_image_fn,
                #     evaluate_result_path=evaluate_result_group_path)
                cv2.imshow("h", rgb_img)
                cv2.waitKey(0)
                break
            break
        break