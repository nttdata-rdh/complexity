import numpy as np
import cv2

img_extension = [".jpg", ".png", ".jpeg", ".bmp"]


def image_normalize_to_float(x: np.ndarray) -> np.ndarray:
    if x.dtype.kind == "i" or x.dtype.kind == "u":
        return np.clip((x / 255.0).astype("float64"), 0, 1.0).astype("float32")
    if x.dtype.kind == "b":
        return np.clip(x.astype("float64"), 0, 1.0).astype("float32")
    return np.clip(x, 0, 1.0)


def image_normalize_to_int(x: np.ndarray) -> np.ndarray:
    if x.dtype.kind == "f":
        return np.clip((x * 255).astype("int"), 0, 255).astype("uint8")
    if x.dtype.kind == "b":
        return np.clip(x.astype("int"), 0, 255).astype("uint8")
    return np.clip(x, 0, 255)


def image_contrast(x: np.ndarray, contrast: float) -> np.ndarray:
    return_type = x.dtype
    x = image_normalize_to_float(x)
    res = np.clip(cv2.multiply(x, np.array(contrast)), 0, 1.0)
    return (
        image_normalize_to_float(res)
        if return_type.kind == "f"
        else image_normalize_to_int(res)
    )


def image_brightness(x: np.ndarray, brightness: float) -> np.ndarray:
    return_type = x.dtype
    x = image_normalize_to_float(x)
    res = np.clip(cv2.add(x, np.array(brightness - 1)), 0, 1.0)
    return (
        image_normalize_to_float(res)
        if return_type.kind == "f"
        else image_normalize_to_int(res)
    )
    # x = image_normalize_to_int(x)
    # x_hsv = cv2.cvtColor(x, cv2.COLOR_RGB2HSV)
    # x_hsv[:, :, 2] = np.clip(x_hsv[:, :, 2] * brightness, 0, 255)
    # return image_normalize_to_float(cv2.cvtColor(x_hsv, cv2.COLOR_HSV2RGB))


def image_saturation(x: np.ndarray, saturation: float) -> np.ndarray:
    return_type = x.dtype
    x = image_normalize_to_float(x)
    x_hsv = cv2.cvtColor(x, cv2.COLOR_RGB2HSV)
    x_hsv[:, :, 1] = x_hsv[:, :, 1] * saturation
    res = cv2.cvtColor(x_hsv, cv2.COLOR_HSV2RGB)
    return (
        image_normalize_to_float(res)
        if return_type.kind == "f"
        else image_normalize_to_int(res)
    )
