import numpy as np


def grid_images(images, H=20, W=10):
    rows = []
    for l in range(0, H * W, W):
        rows.append(np.concatenate(images[l : l + W], axis=1))
    return np.concatenate(rows, axis=0)


def grid_images_filter(images, labels, cur_label, H=20, W=10):
    images, _ = zip(*filter(lambda il: il[1] == cur_label, zip(images, labels)))
    return grid_images(images, H, W)
