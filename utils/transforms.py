import numpy as np
import torch
import torch.nn.functional as F
import skimage.transform
from scipy import stats

class Resize(object):
    def __call__(self, data, image_size=640, debug=False):
        is_with_depth = False
        if len(data) == 3:
            is_with_depth = True

        if not is_with_depth:
            image = data[0]
            annotations = data[1]
        else:
            image = data[0]
            depth_image = data[1]
            annotations = data[2]

        height, width, _ = image.shape
        long_side = max(height, width)
        scale = image_size / long_side

        # image = image.resize((int(height * image_size / long_side), int(width * image_size / long_side)))
        image = skimage.transform.resize(image, (int(height * image_size / long_side), int(width * image_size / long_side)))
        image = image * 255

        if is_with_depth:
            depth_image = skimage.transform.resize(
                depth_image,
                (
                    int(height * image_size / long_side), int(width * image_size / long_side)
                ))

            # depth_image = np.nan_to_num(depth_image)

            depth_image[np.isnan(depth_image)] = 0#np.nanmean(depth_image)
            depth_image[depth_image < 0] = 0

            depth_image = 1 - np.log2(depth_image + 1)

            depth_image = (depth_image - np.nanmean(depth_image)) / np.nanstd(depth_image)

            depth_image_max, depth_image_min = depth_image.max(), depth_image.min()
            depth_image = (depth_image - depth_image_min) / (depth_image_max - depth_image_min)

            # depth_image = (depth_image - np.nanmean(depth_image)) / np.nanstd(depth_image)

            new_depth_image = np.zeros(shape=(depth_image.shape[0], depth_image.shape[1], 3))
            new_depth_image[:, :, 0] = depth_image
            new_depth_image[:, :, 1] = depth_image
            new_depth_image[:, :, 2] = depth_image

        annotations = annotations * scale

        # if debug:
        #     from PIL import ImageDraw
        #     import matplotlib.pyplot as plt
        #
        #     draw = ImageDraw.Draw(image)
        #     draw.rectangle(((annotations[0][0], annotations[0][1]), (annotations[0][2], annotations[0][3])), outline=(0, 0, 255), width=4)
        #     numpy_img = np.array(image)
        #     plt.imshow(numpy_img)
        #     plt.show()

        if not is_with_depth:
            return (image, annotations)
        else:
            return (image, new_depth_image, annotations)

class PadToSquare(object):
    def __call__(self, data, image_size=640, debug=False):
        is_with_depth = False
        if len(data) == 3:
            is_with_depth = True

        if not is_with_depth:
            image = data[0]
            annotations = data[1]
        else:
            image = data[0]
            depth_image = data[1]
            annotations = data[2]

        height, width, _ = image.shape
        dim_diff = np.abs(height - width)
        
        if width == image_size:
            diff = image_size - height
            annotations[:, 1] += diff / 2
            annotations[:, 3] += diff / 2
        elif height == image_size:
            diff = image_size - width
            annotations[:, 0] += diff / 2
            annotations[:, 2] += diff / 2

        annotations = torch.from_numpy(annotations)

        img = torch.from_numpy(np.array(image))
        img = img.permute(2, 0, 1)
        pad1, pad2 = dim_diff // 2, dim_diff - dim_diff // 2
        pad = (0, 0, pad1, pad2) if height <= width else (pad1, pad2, 0, 0)

        padded_img = F.pad(img, pad, "constant", value=0)
        padded_img = padded_img.permute(1, 2, 0)

        if is_with_depth:
            depth_image = torch.from_numpy(np.array(depth_image))
            depth_image = depth_image.permute(2, 0, 1)
            pad1, pad2 = dim_diff // 2, dim_diff - dim_diff // 2
            pad = (0, 0, pad1, pad2) if height <= width else (pad1, pad2, 0, 0)

            padded_depth_image = F.pad(depth_image, pad, "constant", value=0)
            padded_depth_image = padded_depth_image.permute(1, 2, 0)

        # if debug:
        #     from PIL import Image
        #     from PIL import ImageDraw
        #     import matplotlib.pyplot as plt
        #
        #     pil_image = Image.fromarray(padded_img.numpy())
        #
        #     draw = ImageDraw.Draw(pil_image)
        #     draw.rectangle((
        #         (annotations[0][0], annotations[0][1]),
        #         (annotations[0][2], annotations[0][3])), outline=(0, 0, 255), width=4)
        #     numpy_img = np.array(pil_image)
        #     plt.imshow(numpy_img)
        #     plt.show()

        if not is_with_depth:
            return (padded_img, annotations)
        else:
            return (padded_img, padded_depth_image, annotations)


class ToTensor(object):
    def __call__(self, image, target):
        image = F.to_tensor(image)
        return image, target