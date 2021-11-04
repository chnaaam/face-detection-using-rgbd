# import torch
#
# # For debugging
# from tqdm import tqdm
# import matplotlib.pyplot as plt
# import matplotlib.patches as patches
#
# def get_prior_box(image_size, strides, scale_step, aspect_ratio, debug=False):
#     prior_box = []
#     feature_levels = [2 ** x for x in [2, 3, 4, 5, 6]]
#
#     for idx, level in enumerate(feature_levels):
#         feature_size_H = image_size[0] // level
#         feature_size_W = image_size[1] // level
#
#         for f_h in range(feature_size_H):
#             for f_w in range(feature_size_W):
#                 for scale in scale_step:
#                     for ratio in aspect_ratio:
#
#                         cy = (f_h + 0.5) * (2 ** strides[idx])
#                         cx = (f_w + 0.5) * (2 ** strides[idx])
#
#                         w = h = 2 * (2 ** (strides[idx] + 1)) * scale * ratio
#
#                         x1 = cx - w/2
#                         y1 = cy - h/2
#                         x2 = cx + w/2
#                         y2 = cy + h/2
#
#                         prior_box.append((x1, y1, x2, y2))
#
#         break
#
#     if debug:
#         print(len(prior_box))
#         fig, ax = plt.subplots()
#         ax.plot([-640, 640], [-640, 640])
#
#         for x1, y1, x2, y2 in tqdm(prior_box):
#             cx = (x1 + x2) / 2
#             cy = (y1 + y2) / 2
#             width = x2 - x1
#             height = y2 - y1
#
#             ax.add_patch(
#                 patches.Rectangle(
#                     (cx, cy),
#                     width,
#                     height,
#                     edgecolor='red',
#                     fill=False))
#
#         plt.show()
#
#     return torch.tensor(prior_box)
#
# if __name__ == "__main__":
#     get_prior_box(
#         image_size = (640, 640),
#         strides = [2, 3, 4, 5, 6],
#         scale_step = [2 ** (x/3) for x in range(3)],
#         aspect_ratio = [1],
#         debug=True)

import numpy as np
import torch
import torch.nn as nn


class Anchors(nn.Module):
    def __init__(self, pyramid_levels=None, strides=None, sizes=None, ratios=None, scales=None):
        super(Anchors, self).__init__()

        if pyramid_levels is None:
            self.pyramid_levels = [2, 3, 4, 5, 6]
            # self.pyramid_levels = [3, 4, 5]
        if strides is None:
            self.strides = [2 ** x for x in self.pyramid_levels]
        if sizes is None:
            self.sizes = [2 ** (x + 2) for x in self.pyramid_levels]
            # self.sizes = [2 ** 4.0, 2 ** 6.0, 2 ** 8.0]
        if ratios is None:
            self.ratios = np.array([1, 1, 1])
        if scales is None:
            self.scales = np.array([2 ** 0, 2 ** (1.0 / 3.0), 2 ** (2.0 / 3.0)])
            # self.scales = np.array([2 ** 0, 2 ** (1 / 2.0), 2 ** 1.0])

    def forward(self, image):

        image_shape = image.shape[2:]
        image_shape = np.array(image_shape)
        image_shapes = [(image_shape + 2 ** x - 1) // (2 ** x) for x in self.pyramid_levels]

        # compute anchors over all pyramid levels
        all_anchors = np.zeros((0, 4)).astype(np.float32)

        for idx, p in enumerate(self.pyramid_levels):
            anchors = generate_anchors(base_size=self.sizes[idx], ratios=self.ratios, scales=self.scales)
            shifted_anchors = shift(image_shapes[idx], self.strides[idx], anchors)
            all_anchors = np.append(all_anchors, shifted_anchors, axis=0)

        all_anchors = np.expand_dims(all_anchors, axis=0)

        return torch.from_numpy(all_anchors.astype(np.float32))


def generate_anchors(base_size=16, ratios=None, scales=None):
    """
    Generate anchor (reference) windows by enumerating aspect ratios X
    scales w.r.t. a reference window.
    """

    if ratios is None:
        ratios = np.array([1, 1, 1])

    if scales is None:
        scales = np.array([2 ** 0, 2 ** (1.0 / 3.0), 2 ** (2.0 / 3.0)])

    num_anchors = len(scales)

    # initialize output anchors
    anchors = np.zeros((num_anchors, 4))

    # scale base_size
    anchors[:, 2:] = base_size * np.tile(scales, (2, 1)).T

    # transform from (x_ctr, y_ctr, w, h) -> (x1, y1, x2, y2)
    anchors[:, 0::2] -= np.tile(anchors[:, 2] * 0.5, (2, 1)).T
    anchors[:, 1::2] -= np.tile(anchors[:, 3] * 0.5, (2, 1)).T

    return anchors


def shift(shape, stride, anchors):
    shift_x = (np.arange(0, shape[1]) + 0.5) * stride
    shift_y = (np.arange(0, shape[0]) + 0.5) * stride

    shift_x, shift_y = np.meshgrid(shift_x, shift_y)

    shifts = np.vstack((
        shift_x.ravel(), shift_y.ravel(),
        shift_x.ravel(), shift_y.ravel()
    )).transpose()

    # add A anchors (1, A, 4) to
    # cell K shifts (K, 1, 4) to get
    # shift anchors (K, A, 4)
    # reshape to (K * A, 4) shifted anchors
    A = anchors.shape[0]
    K = shifts.shape[0]
    all_anchors = (anchors.reshape((1, A, 4)) + shifts.reshape((1, K, 4)).transpose((1, 0, 2)))
    all_anchors = all_anchors.reshape((K * A, 4))

    return all_anchors

if __name__ == "__main__":
    anchors = Anchors()
    prior_box = anchors.forward(image=torch.ones(size=(1, 3, 640, 640)))

    print(prior_box.shape)

    import matplotlib.pyplot as plt
    import matplotlib.patches as patches
    from tqdm import tqdm

    fig, ax = plt.subplots()
    ax.plot([-640, 640], [-640, 640])

    for x1, y1, x2, y2 in tqdm(prior_box[0, :, :].numpy()):
        cx = (x1 + x2) / 2
        cy = (y1 + y2) / 2
        width = x2 - x1
        height = y2 - y1

        ax.add_patch(
            patches.Rectangle(
                (cx, cy),
                width,
                height,
                edgecolor='red',
                fill=False))

    plt.show()