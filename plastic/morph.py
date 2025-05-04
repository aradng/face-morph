from subprocess import DEVNULL, PIPE, Popen
import numpy as np
import cv2
from PIL import Image
from pathlib import Path
from tqdm import tqdm

from scipy.spatial import Delaunay


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


def warp_im(im, src_landmarks, dst_landmarks, dst_triangulation):
    # im_out = np.zeros_like(im)
    im_out = im.copy()

    for i in range(len(dst_triangulation)):
        src_tri = src_landmarks[dst_triangulation[i]]
        dst_tri = dst_landmarks[dst_triangulation[i]]
        morph_triangle(im, im_out, src_tri, dst_tri)

    return im_out


def draw_triangulation(img, landmarks, triangulation):
    import matplotlib.pyplot as plt

    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    plt.triplot(
        landmarks[:, 0],
        landmarks[:, 1],
        triangulation,
        color="blue",
        linewidth=1,
    )
    plt.axis("off")
    plt.show()


def affine_transform(src, src_tri, dst_tri, size):
    M = cv2.getAffineTransform(np.float32(src_tri), np.float32(dst_tri))
    # BORDER_REFLECT_101 is good for hiding seems
    dst = cv2.warpAffine(src, M, size, borderMode=cv2.BORDER_REFLECT_101)
    return dst


def morph_triangle(im, im_out, src_tri, dst_tri):
    # For efficiency, we crop out a rectangular region containing the triangles
    # to warp only that small part of the image.

    # Get bounding boxes around triangles
    sr = cv2.boundingRect(np.float32([src_tri]))
    dr = cv2.boundingRect(np.float32([dst_tri]))

    # Get new triangle coordinates reflecting their location in bounding box
    cropped_src_tri = [
        (src_tri[i][0] - sr[0], src_tri[i][1] - sr[1]) for i in range(3)
    ]
    cropped_dst_tri = [
        (dst_tri[i][0] - dr[0], dst_tri[i][1] - dr[1]) for i in range(3)
    ]

    # Create mask for destination triangle
    mask = np.zeros((dr[3], dr[2], 3), dtype=np.float32)
    cv2.fillConvexPoly(mask, np.int32(cropped_dst_tri), (1.0, 1.0, 1.0), 16, 0)

    # Crop input image to corresponding bounding box
    cropped_im = im[sr[1] : sr[1] + sr[3], sr[0] : sr[0] + sr[2]]

    size = (dr[2], dr[3])
    warpImage1 = affine_transform(
        cropped_im, cropped_src_tri, cropped_dst_tri, size
    )

    # Copy triangular region of the cropped patch to the output image
    im_out[dr[1] : dr[1] + dr[3], dr[0] : dr[0] + dr[2]] = (
        im_out[dr[1] : dr[1] + dr[3], dr[0] : dr[0] + dr[2]] * (1 - mask)
        + warpImage1 * mask
    )


def morph_seq(
    total_frames, img, src_landmarks, dst_landmarks, triangulation, stream
):

    img = np.float32(img)

    for j in tqdm(range(total_frames)):
        alpha = j / (total_frames - 1)
        weighted_landmarks = (
            1.0 - alpha
        ) * src_landmarks + alpha * dst_landmarks

        warped_img = warp_im(
            img, src_landmarks, weighted_landmarks, triangulation
        )

        res = Image.fromarray(np.uint8(warped_img))
        if stream is not None:
            res.save(stream.stdin, "JPEG")

    return res


def morph_pair(
    frames: int,
    src_landmarks: np.ndarray,
    dst_landmarks: np.ndarray,
    img: np.ndarray,
    debug: bool = False,
    output_name: str = "output",
    fps: int = 30,
    gif: bool = False,
):
    """
    For a pair of images, produce a morph sequence with the given duration
    and fps to be written to the provided output stream.
    """
    h = max(img.shape[:2])
    w = min(img.shape[:2])

    p = None
    output_name = Path(str(output_name).split(".")[0])
    # path.mkdir(parents=True, exist_ok=True)
    # output_name = str((path / path.name).absolute())
    if gif:
        command = [
            "ffmpeg",
            "-y",
            "-f",
            "image2pipe",
            "-r",
            str(fps),
            "-s",
            f"{h}x{w}",
            "-i",
            "-",
            "-vf",
            "scale=trunc(iw/2)*2:trunc(ih/2)*2",
            f"{output_name}.gif",
        ]

        p = Popen(command, stdin=PIPE, stdout=DEVNULL, stderr=DEVNULL)

    average_landmarks = (src_landmarks + dst_landmarks) / 2
    average_landmarks = dst_landmarks

    triangulation = Delaunay(average_landmarks).simplices
    if debug:
        draw_triangulation(img, average_landmarks, triangulation)
    return morph_seq(
        frames,
        img,
        src_landmarks,
        dst_landmarks,
        triangulation.tolist(),
        p,
    )
