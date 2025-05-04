from pathlib import Path
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import numpy as np
import matplotlib.pyplot as plt
import mediapipe as mp
import cv2

from landmarks import (
    adjust_fifths,
    get_axis,
    get_fifths_landmarks,
    get_ratios,
    get_thirds_landmarks,
    landmark_to_point,
    project_to_image,
)
from morph import morph_pair


def draw(img, detection_result, output_name):
    face_landmarks_list = detection_result.face_landmarks
    facial_transfrom_matrix_list = (
        detection_result.facial_transformation_matrixes
    )
    annotated_image = np.copy(img)

    for idx in range(len(face_landmarks_list)):
        landmarks_projection = [
            project_to_image(landmark_to_point(landmark), img)
            for landmark in face_landmarks_list[idx]
        ]
        transfrom_matrix = facial_transfrom_matrix_list[idx]
        x_axis, y_axis = get_axis(transfrom_matrix[:3, :3])

        thirds = get_thirds_landmarks(landmarks_projection)

        fifths = get_fifths_landmarks(landmarks_projection)
        thirds_ratios, fifths_ratios = get_ratios(
            landmarks_projection, transfrom_matrix[:3, :3]
        )
        print(thirds, fifths)
        print(thirds_ratios, fifths_ratios)

        # draw_fifths(annotated_image, fifths, y_axis)
        # draw_thids(annotated_image, thirds, x_axis)

        adjusted_landmarks = adjust_fifths(
            landmarks_projection, fifths_ratios, x_axis
        )
        annotated_image = morph_pair(
            # 2,
            30,
            np.array(landmarks_projection),
            np.array(adjusted_landmarks),
            annotated_image,
            fps=10,
            output_name=output_name,
            gif=True,
        )

        # draw_landmarks(
        #     annotated_image,
        #     landmarks_projection,
        #     color=(0, 255, 0),
        #     draw_text=False,
        #     filter_index=NOSE_LANDMARKS,
        #     size=3,
        # )
        # draw_landmarks(
        #     annotated_image,
        #     adjusted_landmarks,
        #     draw_text=False,
        #     filter_index=NOSE_LANDMARKS,
        #     size=1,
        # )

    return annotated_image


fn = Path("./images/pink.jpg")
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
annotated_image = draw(
    image.numpy_view(), detection_result, output_name=fn.absolute()
)
plt.figure("Image")
plt.title("Image")
plt.imsave(
    f"./images/{fn.name.split('.')[0]}_landmarks.jpg",
    np.array(annotated_image),
)
plt.imshow(annotated_image)
plt.show()
