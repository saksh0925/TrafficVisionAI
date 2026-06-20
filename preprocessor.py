import cv2
import numpy as np
from PIL import Image

def preprocess_image(image_input):
    """
    Takes a PIL Image or numpy array.
    Returns enhanced numpy array (BGR) ready for detection.
    """
    if isinstance(image_input, Image.Image):
        img = np.array(image_input.convert("RGB"))
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    else:
        img = image_input.copy()

    # Step 1: Denoise
    img = cv2.fastNlMeansDenoisingColored(img, None, 10, 10, 7, 21)

    # Step 2: CLAHE for low-light enhancement (on L channel)
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
    l = clahe.apply(l)
    lab = cv2.merge((l, a, b))
    img = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

    # Step 3: Sharpen to handle motion blur
    kernel = np.array([[0, -1, 0],
                       [-1, 5, -1],
                       [0, -1, 0]])
    img = cv2.filter2D(img, -1, kernel)

    # Step 4: Normalize brightness
    img = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX)

    return img


def resize_for_model(img, size=640):
    """Resize image to model input size keeping aspect ratio."""
    h, w = img.shape[:2]
    scale = size / max(h, w)
    new_w, new_h = int(w * scale), int(h * scale)
    resized = cv2.resize(img, (new_w, new_h))
    return resized
