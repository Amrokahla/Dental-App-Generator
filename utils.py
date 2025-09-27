# src/utils.py
import logging
import cv2
import numpy as np
import os
from dotenv import load_dotenv

# load .env early
load_dotenv()

# Configure logging once
logging.basicConfig(
    level=logging.DEBUG,  # set to INFO or WARNING in production
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("smilegen")


def get_api_key() -> str:
    """
    Return IMAGINE_KEY from environment or raise.
    """
    key = os.getenv("IMAGINE_KEY")
    if not key:
        logger.error("IMAGINE_KEY not found in environment.")
        raise RuntimeError("Missing IMAGINE_KEY environment variable")
    logger.debug("Loaded IMAGINE_KEY successfully.")
    return key


def ensure_dir(path: str):
    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)
        logger.debug("Created directory: %s", path)


def correct_face_orientation(img: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    # Load Haar cascades
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_eye.xml")

    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)
    if len(faces) == 0:
        return img  # no face found, return original

    for (x, y, w, h) in faces:
        roi_gray = gray[y:y+h, x:x+w]
        eyes = eye_cascade.detectMultiScale(roi_gray)
        if len(eyes) >= 2:
            # Eye centers
            eye1 = eyes[0]
            eye2 = eyes[1]
            eye_center1 = (x + eye1[0] + eye1[2] // 2, y + eye1[1] + eye1[3] // 2)
            eye_center2 = (x + eye2[0] + eye2[2] // 2, y + eye2[1] + eye2[3] // 2)

            dx = eye_center2[0] - eye_center1[0]
            dy = eye_center2[1] - eye_center1[1]
            angle = np.degrees(np.arctan2(dy, dx))

            # Special-case: if nearly 90 or 180, use cv2.rotate
            if abs(abs(angle) - 90) < 10:
                return cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE if angle > 0 else cv2.ROTATE_90_COUNTERCLOCKWISE)
            elif abs(abs(angle) - 180) < 10:
                return cv2.rotate(img, cv2.ROTATE_180)

            # Otherwise: safe affine rotation with border
            h_img, w_img = img.shape[:2]
            center = (w_img // 2, h_img // 2)
            rot_mat = cv2.getRotationMatrix2D(center, angle, 1.0)

            # Add border before rotation to prevent cutoff
            expanded = cv2.copyMakeBorder(img, 100, 100, 100, 100, cv2.BORDER_REPLICATE)
            h_exp, w_exp = expanded.shape[:2]
            center_exp = (w_exp // 2, h_exp // 2)
            rot_mat = cv2.getRotationMatrix2D(center_exp, angle, 1.0)
            rotated = cv2.warpAffine(expanded, rot_mat, (w_exp, h_exp), flags=cv2.INTER_LINEAR)

            # Crop back to original size
            rotated = rotated[100:h_exp-100, 100:w_exp-100]
            return rotated

    return img  # fallback

