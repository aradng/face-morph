from pathlib import Path
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import numpy as np
import matplotlib.pyplot as plt
import mediapipe as mp
import cv2

from landmarks import (
    draw_fifths,
    draw_thids,
    get_axis,
    get_fifths_landmarks,
    get_ratios,
    get_thirds_landmarks,
    landmark_to_point,
    project_to_image,
)
from morph import warp_box


def draw_landmarks(img, detection_result):
    face_landmarks_list = detection_result.face_landmarks
    facial_transfrom_matrix_list = detection_result.facial_transformation_matrixes
    annotated_image = np.copy(img)

    for idx in range(len(face_landmarks_list)):
        landmarks_projection = [
            project_to_image(landmark_to_point(landmark), img)
            for landmark in face_landmarks_list[idx]
        ]
        transfrom_matrix = facial_transfrom_matrix_list[idx]
        x_axis, y_axis = get_axis(transfrom_matrix[:3, :3])

        # thirds = get_thirds_landmarks(landmarks_projection)

        fifths = get_fifths_landmarks(landmarks_projection)
        thirds_ratios, fifths_ratios = get_ratios(
            landmarks_projection, transfrom_matrix[:3, :3]
        )

        # draw_fifths(annotated_image, fifths, y_axis)
        # draw_thids(annotated_image, thirds, x_axis)

        left_boundary = fifths[2]  # left eye end
        right_boundary = fifths[3]  # right eye start
        region_height = int(img.shape[0])  # Use 50% of image height

        width = np.linalg.norm(right_boundary - left_boundary)
        adjusted_width = width * 20 / fifths_ratios[2]
        offset = (width - adjusted_width) / 2

        middle_top_left = (left_boundary - y_axis * region_height).astype(int)
        middle_bottom_left = (left_boundary + y_axis * region_height).astype(int)
        middle_top_right = (right_boundary - y_axis * region_height).astype(int)
        middle_bottom_right = (right_boundary + y_axis * region_height).astype(int)

        right_top_left = middle_top_right
        right_bottom_left = middle_bottom_right
        right_top_right = np.array([annotated_image.shape[0], 0]).astype(int)
        right_bottom_right = np.array(
            [annotated_image.shape[0], annotated_image.shape[1]]
        ).astype(int)

        middle_src_points = np.array(
            [
                middle_top_left,
                middle_bottom_left,
                middle_bottom_right,
                middle_top_right,
            ],
            np.int32,
        )
        middle_dst_points = np.float32(
            [
                middle_top_left,
                middle_bottom_left,
                middle_bottom_right - x_axis * offset * 2,
                middle_top_right - x_axis * offset * 2,
            ]
        )

        right_src_points = np.array(
            [
                right_top_left,
                right_bottom_left,
                right_bottom_right,
                right_top_right,
            ],
            np.int32,
        )

        right_dst_points = np.float32(
            [
                right_top_left - x_axis * offset * 2,
                right_bottom_left - x_axis * offset * 2,
                right_bottom_right - x_axis * offset * 2,
                right_top_right - x_axis * offset * 2,
            ]
        )

        warp_box(annotated_image, middle_src_points, middle_dst_points)
        warp_box(annotated_image, right_src_points, right_dst_points)

        annotated_image = annotated_image[:, : -int((x_axis * offset * 2)[0]), :]

        # for i, landmark in enumerate(landmarks_projection):
        #     cv2.circle(annotated_image, landmark, 1, (255, 0, 0), -1)
        #     cv2.putText(
        #         annotated_image,
        #         str(i),
        #         landmark,
        #         cv2.FONT_HERSHEY_SIMPLEX,
        #         0.3,
        #         (255, 0, 0),
        #         1,
        #     )
    return annotated_image


fn = Path("./images/ilya.jpg")
img = cv2.imread(fn)
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
base_options = python.BaseOptions(
    model_asset_path=Path("./face_landmarker.task"),
    # delegate=python.BaseOptions.Delegate.GPU,
)
options = vision.FaceLandmarkerOptions(
    base_options=base_options,
    output_face_blendshapes=True,
    output_facial_transformation_matrixes=True,
    num_faces=1,
)
detector = vision.FaceLandmarker.create_from_options(options)

# STEP 3: Load the input image.
image = mp.Image(image_format=mp.ImageFormat.SRGB, data=img)

# STEP 4: Detect face landmarks from the input image.
detection_result = detector.detect(image)

# STEP 5: Process the detection result. In this case, visualize it.
annotated_image = draw_landmarks(image.numpy_view(), detection_result)
# cv2.imshow("Image", annotated_image)
plt.figure("Image")
plt.title("Image")
plt.imshow(annotated_image)
plt.imsave(f"./images/{fn.name.split('.')[0]}_landmarks.png", annotated_image)
plt.show()
