# src/pipeline.py
import cv2
from utils import logger
from detector import load_image, detect_landmarks, extract_mouth_region
from mask_utils import create_mouth_mask
from generator import generate_variants


def run_pipeline(image_path: str,
                 doctor_recommendations: str = "",
                 patient_needs: str = "",
                 num_outputs: int = 4) -> list[dict]:
    """
    Orchestrator: load -> detect (0/90/180/270) -> mask -> call generator.
    Returns list of dicts that generator produces (each dict contains 'image', 'mask', 'prompt').
    The image passed to generator is the aligned (upright) image — original orientation is forgotten.
    """
    # Load
    img = load_image(image_path)  # RGB numpy

    # Detect landmarks and get the aligned image (the function tries 0/90/180/270)
    landmarks, aligned_img = detect_landmarks(img)

    # Create mask on the aligned image
    mask = create_mouth_mask(aligned_img.shape, landmarks)

    # Prompt
    prompt = f"enhanced smile, {doctor_recommendations}, {patient_needs}".strip(", ")

    # Call generator with the aligned image & mask
    outputs = generate_variants(aligned_img, mask, prompt, num_variants=num_outputs)

    return outputs
