# src/mask_utils.py
import cv2
import numpy as np
from utils import logger

def create_mouth_mask(image_shape, landmarks):
    """
    image_shape: (H, W, C) or (H, W)
    landmarks: Nx2 numpy array
    returns single-channel uint8 mask same (H,W) with 255 inside mouth polygon.
    """
    logger.debug("Creating binary mask for mouth region...")
    h = image_shape[0]
    w = image_shape[1]
    mask = np.zeros((h, w), dtype=np.uint8)
    mouth_points = landmarks[48:68].astype(np.int32)
    cv2.fillPoly(mask, [mouth_points], 255)
    logger.debug("Mask created successfully.")
    return mask
