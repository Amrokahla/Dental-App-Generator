# src/detector.py
import cv2
import dlib
import numpy as np
from pathlib import Path
from utils import logger, ensure_dir, correct_face_orientation
import os, sys

def resource_path(relative_path):
    """Get absolute path for PyInstaller and dev mode"""
    if hasattr(sys, "_MEIPASS"):
        return os.path.join(sys._MEIPASS, relative_path)
    return os.path.join(os.path.abspath("."), relative_path)


PREDICTOR_PATH = resource_path("assets/shape_predictor_68_face_landmarks.dat")

logger.info("Initializing dlib face detector and shape predictor...")
DETECTOR = dlib.get_frontal_face_detector()
try:
    PREDICTOR = dlib.shape_predictor(PREDICTOR_PATH)
    logger.info("Loaded shape predictor model.")
except Exception as e:
    logger.exception("Could not load predictor model from %s: %s", PREDICTOR_PATH, e)
    PREDICTOR = None


def load_image(image_path: str) -> np.ndarray:
    logger.debug("Loading image from %s", image_path)
    img = cv2.imread(image_path)
    if img is None:
        logger.error("Failed to load image at %s", image_path)
        raise FileNotFoundError(f"Image not found: {image_path}")
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def _predict_on_image(img: np.ndarray):
    """
    Run dlib detection+predictor on img (RGB numpy).
    Returns landmarks array Nx2 on success, otherwise None.
    """
    if PREDICTOR is None:
        raise RuntimeError("Dlib predictor not loaded.")
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    rects = DETECTOR(gray, 1)
    if rects:
        # Use the first face found
        shape = PREDICTOR(gray, rects[0])
        landmarks = np.array([(p.x, p.y) for p in shape.parts()], dtype=np.int32)
        return landmarks
    return None


def detect_landmarks(img: np.ndarray) -> tuple:
    """
    Try multiple orientations (0, 90, 180, 270) and return tuple:
        (landmarks: np.ndarray (Nx2), aligned_img: np.ndarray (RGB))
    aligned_img is the rotated image on which detection succeeded (or original if 0° succeeded).
    Raises RuntimeError if no face detected in any orientation.
    """
    logger.debug("Running face detection (orientation-robust)...")

    # 1) Try as-is
    landmarks = _predict_on_image(img)
    if landmarks is not None:
        logger.debug("Detected face without rotation.")
        return landmarks, img

    # 2) Try corrected orientation using the lighter correct_face_orientation (optional)
    #    (This can help with small tilt, but we will still test hard 90-degree rotations next.)
    try:
        img_corrected = correct_face_orientation(img)
    except Exception:
        img_corrected = None

    if img_corrected is not None:
        landmarks = _predict_on_image(img_corrected)
        if landmarks is not None:
            logger.debug("Detected face after small-angle correction.")
            return landmarks, img_corrected

    # 3) Hard rotations (90, 180, 270)
    rotations = [
        ("90°", cv2.ROTATE_90_CLOCKWISE),
        ("180°", cv2.ROTATE_180),
        ("270°", cv2.ROTATE_90_COUNTERCLOCKWISE),
    ]
    for angle_label, flag in rotations:
        rotated = cv2.rotate(img, flag)
        landmarks = _predict_on_image(rotated)
        if landmarks is not None:
            logger.debug("Detected face after %s rotation.", angle_label)
            # Return landmarks found on the rotated image and the rotated image itself.
            return landmarks, rotated

    # All attempts failed
    logger.error("No face detected after 0/90/180/270 attempts.")
    raise RuntimeError("No face detected in image.")


def extract_mouth_region(img: np.ndarray, landmarks: np.ndarray) -> np.ndarray:
    """
    Return mouth crop (RGB numpy) using dlib 48:68 mouth points.
    """
    logger.debug("Extracting mouth region...")
    mouth_pts = landmarks[48:68]
    x, y, w, h = cv2.boundingRect(mouth_pts)
    # clamp
    h_img, w_img = img.shape[:2]
    x = max(0, x); y = max(0, y); w = min(w, w_img - x); h = min(h, h_img - y)
    mouth_crop = img[y:y+h, x:x+w].copy()
    logger.debug("Mouth crop shape: %s", str(mouth_crop.shape))
    return mouth_crop


def run_detection_and_save_previews(src_image_path: str):
    """
    Runs detection and returns dict with in-memory arrays.
    Uses the aligned (upright) image for everything (forgets the original).
    """
    img = load_image(src_image_path)

    # Robust landmark detection that also returns aligned image
    landmarks, aligned_img = detect_landmarks(img)

    # Extract mouth crop from the aligned image
    mouth = extract_mouth_region(aligned_img, landmarks)

    return {
        "image_rgb": aligned_img,   # aligned/upright image
        "landmarks": landmarks,
        "mouth": mouth,   # numpy array (RGB)
    }
