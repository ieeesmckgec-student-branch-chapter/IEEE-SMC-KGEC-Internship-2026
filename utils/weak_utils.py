import numpy as np
from PIL import Image
from skimage.morphology import skeletonize
import random


def create_scribble_mask(mask_pil):

    # PIL -> numpy
    mask = np.array(mask_pil)
    # binary mask
    binary = (mask > 128).astype(np.uint8)
    # skeletonize foreground
    scribble_fg = skeletonize(binary)
    # initialize ignore mask
    weak_mask = np.ones_like(binary, dtype=np.uint8) * 255
    # foreground scribbles
    weak_mask[scribble_fg == 1] = 1
    # random background points
    bg_indices = np.where(binary == 0)
    if len(bg_indices[0]) > 0:
        num_bg = min(500, len(bg_indices[0]))
        selected = np.random.choice(
            len(bg_indices[0]),
            num_bg,
            replace=False
        )
        weak_mask[
            bg_indices[0][selected],
            bg_indices[1][selected]
        ] = 0

    return Image.fromarray(weak_mask)




