from typing import Optional, List

import numpy as np

def crop_np_image(self, loc: List[int, int, int, int], input_image: np.ndarray, leeway: Optional[int] = None):

    h, w = input_image.shape[:2]
    if leeway is None:
        leeway = 0
    x1 = max(
        loc[3] - leeway,
        0,
    )
    y1 = max(
        loc[0] - leeway,
        0,
    )
    x2 = min(
        loc[1] + leeway,
        w,
    )
    y2 = min(
        loc[2] + leeway,
        h,
    )
    return input_image[y1:y2, x1:x2]
