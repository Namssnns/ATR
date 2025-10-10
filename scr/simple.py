from typing import List, Tuple, Union
import cv2
import numpy as np
from odr import ODR


def find_contours(image: np.ndarray) -> List[np.ndarray]:
    if image is None or image.size == 0:
        return []
    contours, _ = cv2.findContours(image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    return list(contours)


def get_contour_center(contour: np.ndarray) -> Tuple[int, int]:
    if contour is None or len(contour) == 0:
        return (0, 0)
    M = cv2.moments(contour)
    if M["m00"] == 0:
        return (0, 0)
    center_X = int(M["m10"] / M["m00"])
    center_Y = int(M["m01"] / M["m00"])
    return (center_X, center_Y)


def extract_contour(image: np.ndarray, contour: np.ndarray) -> np.ndarray:
    if image is None or image.size == 0 or contour is None or len(contour) == 0:
        return np.zeros((1, 1, image.shape[2] if image.ndim == 3 else 1), dtype=np.uint8)

    X, Y, W, H = cv2.boundingRect(contour)
    if W <= 0 or H <= 0:
        return np.zeros((1, 1, image.shape[2] if image.ndim == 3 else 1), dtype=np.uint8)

    mask = np.zeros((H, W), np.uint8)
    cv2.drawContours(mask, [contour], -1, 255, -1, offset=(-X, -Y))
    points = np.where(mask)

    if len(points[0]) == 0:
        return np.zeros((H, W, image.shape[2] if image.ndim == 3 else 1), dtype=np.uint8)

    cutout = np.zeros((H, W) + (() if image.ndim == 2 else (image.shape[2],)), np.uint8)
    try:
        cutout[points] = image[points[0] + Y, points[1] + X]
    except Exception:
        pass
    return cutout


def crop_image(image: np.ndarray, top=True, bottom=True, left=True, right=True) -> np.ndarray:
    if image is None or image.size == 0:
        return np.zeros((1, 1) + (() if image.ndim == 2 else (image.shape[2],)), dtype=np.uint8)

    H, W = image.shape[:2]
    if H == 0 or W == 0:
        return np.zeros((1, 1) + (() if image.ndim == 2 else (image.shape[2],)), dtype=np.uint8)

    non_empty_columns = np.where(image.max(axis=0) > 0)[0]
    non_empty_rows = np.where(image.max(axis=1) > 0)[0]

    if len(non_empty_rows) == 0 or len(non_empty_columns) == 0:
        return np.zeros((1, 1) + (() if image.ndim == 2 else (image.shape[2],)), dtype=np.uint8)

    crop_box = (
        min(non_empty_rows) if top else 0,
        max(non_empty_rows) + 1 if bottom else H,
        min(non_empty_columns) if left else 0,
        max(non_empty_columns) + 1 if right else W,
    )
    return image[crop_box[0]:crop_box[1], crop_box[2]:crop_box[3]]


def mask_transparent(image: np.ndarray, mask: np.ndarray) -> np.ndarray:
    if image is None or mask is None or image.shape[:2] != mask.shape[:2]:
        return np.zeros((1, 1, 4), dtype=np.uint8)

    transparent_image = cv2.cvtColor(image, cv2.COLOR_BGR2BGRA)
    transparent_image[mask == 0] = [0] * 4
    return transparent_image


def upscale(image: np.ndarray, coef: float):
    if image is None or image.size == 0:
        return image
    if coef <= 0:
        return image

    new_width = round(image.shape[1] * coef)
    new_height = round(image.shape[0] * coef)
    if new_width <= 0 or new_height <= 0:
        return image

    return cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_CUBIC)


def mask_color(image: np.ndarray, color: np.ndarray) -> np.ndarray:
    if image is None or image.size == 0 or color is None:
        return np.zeros(image.shape[:2], dtype=np.uint8)
    return np.all(image == color, axis=len(image.shape) - 1).astype(np.uint8) * 255


def has_color(image: np.ndarray, color: np.ndarray) -> bool:
    if image is None or image.size == 0 or color is None:
        return False
    return bool(mask_color(image, color).any())


def spread_hist(image: np.ndarray) -> np.ndarray:
    if image is None or image.size == 0 or len(image.shape) != 2:
        return np.zeros((1, 1), dtype=np.uint8)
    min_val, max_val = np.min(image), np.max(image)
    if max_val == min_val:
        return np.zeros_like(image, dtype=np.uint8)
    return ((image - min_val) * (255 / (max_val - min_val))).astype(np.uint8)


def split_characters(img: np.ndarray) -> None:
    if img is None or img.size == 0:
        return
    cropped = crop_image(img)
    characters_contours = find_contours(cropped)
    if not characters_contours:
        return

    for contour in characters_contours:
        hull = cv2.convexHull(contour, returnPoints=False)
        if hull is None or len(hull) == 0:
            continue
        defects = cv2.convexityDefects(contour, hull)
        if defects is None:
            continue

        try:
            short_lines_filter = filter(
                lambda e: np.linalg.norm(
                    np.array(contour[e[0][0]][0]) - np.array(contour[e[0][1]][0])
                ) > cropped.shape[0] // 2,
                defects
            )
            vertical_lines_filter = filter(
                lambda e: abs(
                    np.arctan2(*(np.array(contour[e[0][0]][0]) -
                                 np.array(contour[e[0][1]][0]))[::-1]) % np.pi - (np.pi / 2)
                ) > np.pi / 4,
                short_lines_filter
            )
            sorted_defects = sorted(vertical_lines_filter, key=lambda e: np.array(contour[e[0][2]][0])[0])

            for i in range(0, 2 * (len(sorted_defects) // 2), 2):
                s1, e1, f1, d1 = sorted_defects[i][0]
                s2, e2, f2, d2 = sorted_defects[i + 1][0]
                far1 = tuple(contour[f1][0])
                far2 = tuple(contour[f2][0])
                cv2.line(cropped, far1, far2, 0, 10)
        except Exception:
            continue


def read_number(odr: ODR, img: np.ndarray, dtype: int | float) -> int | float:
    if img is None or img.size == 0:
        return dtype(0)

    characters_contours = find_contours(img)
    if not characters_contours:
        return dtype(0)

    characters_contours.sort(key=lambda e: get_contour_center(e)[0], reverse=True)

    number = ""
    for character_contour in characters_contours:
        character_img = extract_contour(img, character_contour)
        if character_img.size == 0:
            continue

        try:
            if character_img.shape[0] / img.shape[0] < 0.5:
                if "." not in number and dtype == float:
                    number += "."
                continue

            result = odr.detect(character_img)
            if result is None:
                continue
            number += str(result)
        except Exception:
            continue

    if number.strip() == "" or number == ".":
        return dtype(0)

    try:
        return dtype(number[::-1])
    except Exception:
        return dtype(0)
