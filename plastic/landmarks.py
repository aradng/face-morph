import numpy as np
import cv2


def get_axis(rotation_matrix: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    inv_rot = np.linalg.inv(rotation_matrix)
    # return normalized axis vectors
    return (
        np.dot(inv_rot, np.array([1, 0, 0]))[:2]
        / np.linalg.norm(np.dot(inv_rot, np.array([1, 0, 0]))[:2]),
        np.dot(inv_rot, np.array([0, 1, 0]))[:2]
        / np.linalg.norm(np.dot(inv_rot, np.array([0, 1, 0]))[:2]),
    )


def project_to_image(p, img):
    return np.array([int(p[0] * img.shape[1]), int(p[1] * img.shape[0])])


def landmark_to_point(landmark):
    return np.array([landmark.x, landmark.y])


def get_face_boundaries(landmarks) -> tuple[np.ndarray, np.ndarray]:
    x_min = np.min(landmarks[:, 0])
    x_max = np.max(landmarks[:, 0])
    y_min = np.min(landmarks[:, 1])
    y_max = np.max(landmarks[:, 1])
    return np.array([x_min, y_min]), np.array([x_max, y_max])


def get_face_center_axis(landmarks, x_slope, y_slope) -> np.ndarray:
    center = landmarks[0]
    bound_x, bound_y = get_face_boundaries(landmarks)
    x_start = center + x_slope * (bound_x - center)
    x_end = center - x_slope * (bound_x - center)
    y_start = center + y_slope * (bound_y - center)
    y_end = center - y_slope * (bound_y - center)
    return np.array([x_start, x_end]), np.array([y_start, y_end])


def get_thirds_landmarks(face_landmarks) -> list[np.ndarray]:
    hairline = face_landmarks[10]
    nose_top = face_landmarks[8]
    nose_bot = face_landmarks[1]
    chin = face_landmarks[152]

    return [hairline, nose_top, nose_bot, chin]


def get_fifths_landmarks(face_landmarks) -> list[np.ndarray]:
    face_bound_left = face_landmarks[234]
    left_eye_start = face_landmarks[130]
    left_eye_end = face_landmarks[133]
    right_eye_start = face_landmarks[362]
    right_eye_end = face_landmarks[359]
    face_bound_right = face_landmarks[454]

    return [
        face_bound_left,
        left_eye_start,
        left_eye_end,
        right_eye_start,
        right_eye_end,
        face_bound_right,
    ]


def draw_thids(
    img: np.ndarray,
    thirds: list[np.ndarray],
    x_axis: np.ndarray,
):
    for i in thirds:
        x1 = (i + x_axis * img.shape[0]).astype(int)
        x2 = (i - x_axis * img.shape[0]).astype(int)
        cv2.line(
            img,
            x1,
            x2,
            (0, 0, 0),
            img.shape[0] // 200,
        )

    return img


def draw_fifths(img: np.ndarray, fifths: list[np.ndarray], y_axis: np.ndarray):
    for i in fifths:
        y1 = (i + y_axis * img.shape[0]).astype(int)
        y2 = (i - y_axis * img.shape[0]).astype(int)
        cv2.line(
            img,
            y1,
            y2,
            (0, 0, 0),
            img.shape[0] // 200,
        )


def get_ratios(
    landmarks: list[np.ndarray], rotation_matrix: np.ndarray
) -> tuple[list[float], list[float]]:
    thirds = get_thirds_landmarks(landmarks)
    fifths = get_fifths_landmarks(landmarks)
    x_axis, y_axis = get_axis(rotation_matrix)

    thids_total = abs(
        np.dot(np.array(thirds[-1]) - np.array(thirds[0]), y_axis)
    )
    fifths_total = abs(
        np.dot(np.array(fifths[-1]) - np.array(fifths[0]), x_axis)
    )

    # Project distances onto y-axis for thirds
    thirds_ratios = []
    for p1, p2 in zip(thirds[:-1], thirds[1:]):
        projected_dist = abs(np.dot(np.array(p2) - np.array(p1), y_axis))
        thirds_ratios.append(projected_dist / thids_total * 100)

    # Project distances onto x-axis for fifths
    fifths_ratios = []
    for p1, p2 in zip(fifths[:-1], fifths[1:]):
        projected_dist = abs(np.dot(np.array(p2) - np.array(p1), x_axis))
        fifths_ratios.append(projected_dist / fifths_total * 100)

    return thirds_ratios, fifths_ratios


def adjust_fifths(
    landmarks: list[np.ndarray],
    fifths_ratios: list[np.ndarray],
    y_axis: np.ndarray,
) -> list[np.ndarray]:
    # change middle section (nose) to match average ratio

    adjustment_ratio = 1 - (20 / fifths_ratios[2])

    adjusted_landmarks = [
        (
            (
                landmark
                + (np.dot(landmarks[1] - landmark, y_axis) * y_axis)
                * adjustment_ratio
            ).astype(int)
            if i in NOSE_LANDMARKS
            else landmark
        )
        for i, landmark in enumerate(landmarks)
    ]
    return adjusted_landmarks


def draw_landmarks(
    img: np.ndarray,
    landmarks: list[np.ndarray],
    color=(255, 0, 0),
    size=1,
    draw_text: bool = False,
    filter_index: list[int] | None = None,
) -> np.ndarray:
    for i, landmark in enumerate(landmarks):
        if filter_index is not None and i not in filter_index:
            continue
        cv2.circle(img, landmark, size, color, -1)
        if draw_text:
            cv2.putText(
                img,
                str(i),
                landmark,
                cv2.FONT_HERSHEY_SIMPLEX,
                1.3,
                color,
                1,
            )


NOSE_LANDMARKS = [
    8,  #
    193,
    168,
    417,  #
    245,
    122,
    6,
    55,
    285,
    351,
    465,  #
    188,
    412,  #
    126,
    217,
    174,
    196,
    197,
    419,
    399,
    437,
    355,  #
    209,
    198,
    236,
    363,
    20,
    3,
    195,
    248,
    456,
    420,
    60,
    429,  #
    102,
    49,
    131,
    134,
    51,
    5,
    281,
    360,
    279,  # might be more at the right down end
    48,
    115,
    220,
    45,
    4,
    275,
    440,
    344,
    278,  #
    64,
    219,
    218,
    79,
    237,
    239,
    166,
    235,
    59,
    98,
    240,
    75,
    99,
    97,  #
    44,
    1,
    274,
    238,
    241,
    125,
    19,
    354,
    461,
    458,
    242,
    141,
    94,
    370,
    462,
    250,
    2,  #
    457,
    459,
    290,
    328,
    326,
    309,
    305,
    438,
    392,
    289,
    305,
    439,
    455,
    294,
    331,
]
