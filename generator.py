import requests
import cv2
import numpy as np
from utils import get_api_key, logger, correct_face_orientation
import io
from PIL import Image
from PyQt5.QtGui import QImage, QPixmap

# Import your detection functions
from detector import detect_landmarks, extract_mouth_region

IMAGINE_URL = "https://api.vyro.ai/v2/image/edits/generative-fill"


def numpy_to_qpixmap(arr: np.ndarray) -> QPixmap:
    """
    Convert RGB numpy array -> QPixmap.
    """
    h, w, ch = arr.shape
    bytes_per_line = ch * w
    qimg = QImage(arr.data, w, h, bytes_per_line, QImage.Format_RGB888)
    return QPixmap.fromImage(qimg.copy())  # copy() avoids memory ownership issues


def call_imagine_api(image_array: np.ndarray, mask_array: np.ndarray,
                     prompt: str, timeout: int = 60) -> np.ndarray:
    """
    Send numpy arrays to Imagine API, return generated RGB numpy array.
    """
    api_key = get_api_key()
    headers = {"Authorization": f"Bearer {api_key}"}

    # Encode numpy arrays to PNG in memory
    _, img_encoded = cv2.imencode(".png", cv2.cvtColor(image_array, cv2.COLOR_RGB2BGR))
    _, mask_encoded = cv2.imencode(".png", mask_array)

    files = {
        "image": ("image.png", io.BytesIO(img_encoded.tobytes()), "image/png"),
        "mask": ("mask.png", io.BytesIO(mask_encoded.tobytes()), "image/png"),
    }
    data = {"prompt": prompt}

    resp = requests.post(IMAGINE_URL, headers=headers, files=files, data=data, timeout=timeout)
    if resp.status_code != 200:
        raise RuntimeError(f"API error: {resp.status_code} - {resp.text}")

    # Response -> numpy array
    img_bytes = io.BytesIO(resp.content)
    pil_img = Image.open(img_bytes).convert("RGB")
    return np.array(pil_img)

def generate_variants(image_array: np.ndarray, mask_array: np.ndarray,
                      base_prompt: str, num_variants: int = 4) -> list[dict]:
    """
    Generate multiple variants.
    Each output is a dict: { "image": np.ndarray, "mask": np.ndarray, "prompt": str }
    """
    outputs = []
    for i in range(num_variants):
        prompt = f"{base_prompt} | variation {i+1}"
        try:
            out_img = call_imagine_api(image_array, mask_array, prompt)
            outputs.append({
                "image": out_img,
                "mask": mask_array,
                "prompt": prompt
            })
        except Exception as e:
            logger.error("Variant %d failed: %s", i+1, e)
    return outputs
