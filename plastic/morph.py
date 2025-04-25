from matplotlib import pyplot as plt
import numpy as np
import cv2


def get_boundary_points(shape):
    h, w = shape[:2]
    boundary_pts = [
        (1, 1),
        (w - 1, 1),
        (1, h - 1),
        (w - 1, h - 1),
        ((w - 1) // 2, 1),
        (1, (h - 1) // 2),
        ((w - 1) // 2, h - 1),
        ((w - 1) // 2, (h - 1) // 2),
    ]
    return np.array(boundary_pts)


def crop_to_face(im, landmarks):
    # Get the bounding box of the landmarks
    x_min = int(np.min(landmarks[:, 0]))
    x_max = int(np.max(landmarks[:, 0]))
    y_min = int(np.min(landmarks[:, 1]))
    y_max = int(np.max(landmarks[:, 1]))

    # Add some padding to the bounding box
    padding = 0.1
    padding_x = int((x_max - x_min) * padding)
    padding_y = int((y_max - y_min) * padding)
    x_min = max(0, x_min - padding_x)
    x_max = min(im.shape[1], x_max + padding_x)
    y_min = max(0, y_min - padding_y)
    y_max = min(im.shape[0], y_max + padding_y)

    # Crop the image to the bounding box
    cropped_face = im[y_min:y_max, x_min:x_max]

    return cropped_face


def warp_box(img, src_points: np.ndarray, dst_points, debug: bool = False):
    mask = np.zeros_like(img)
    cv2.fillPoly(mask, [dst_points.astype(np.int32)], (255, 255, 255))
    transform_matrix = cv2.getAffineTransform(
        src_points[:3].astype(np.float32), dst_points[:3]
    )
    transformed = cv2.warpAffine(img, transform_matrix, (img.shape[1], img.shape[0]))
    mask = mask.astype(bool)
    if debug:
        _, ax = plt.subplots(1, 2, figsize=(10, 5))
        debug_before = np.zeros_like(img)
        debug_after = np.zeros_like(img)
        debug_before_mask = np.zeros_like(img)
        debug_after_mask = np.zeros_like(img)
        cv2.fillPoly(debug_before_mask, [src_points.astype(np.int32)], (255, 255, 255))
        cv2.fillPoly(debug_after_mask, [dst_points.astype(np.int32)], (255, 255, 255))
        debug_before_mask = debug_before_mask.astype(bool)
        debug_after_mask = debug_after_mask.astype(bool)
        debug_before[debug_before_mask] = img[debug_before_mask]
        debug_after[debug_after_mask] = transformed[debug_after_mask]
        ax[0].set_title("Before")
        ax[0].imshow(debug_before)
        ax[1].set_title("After")
        ax[1].imshow(debug_after)
        plt.show()
    img[mask] = transformed[mask]
