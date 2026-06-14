import cv2
import numpy as np
import asyncio
import pywt
import yaml
import onnxruntime as ort
from huggingface_hub import hf_hub_download
from concurrent.futures import ThreadPoolExecutor

executor = ThreadPoolExecutor()

# load config for threshold and denoising settings
with open("config.yaml", "r") as f:
    config = yaml.safe_load(f)

THRESHOLD = config["validation"]["noise_threshold"]

class DnCNNDenoiser:
    def __init__(self, repo_id: str, filename: str):
        self.repo_id = repo_id
        self.filename = filename
        self._session = None

    def get_session(self):
        if self._session is None:
            model_path = hf_hub_download(repo_id=self.repo_id, filename=self.filename)
            self._session = ort.InferenceSession(model_path)
        return self._session

def apply_clahe(img: np.ndarray) -> np.ndarray:  # CLAHE -> 
    """applies CLAHE enhancement to the image."""
    # convert to LAB to apply CLAHE on the L channel (lightness)
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    cl = clahe.apply(l)
    limg = cv2.merge((cl, a, b))
    return cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)

def apply_dncnn(img: np.ndarray, denoiser: DnCNNDenoiser) -> np.ndarray:
    """applies DnCNN denoising using ONNX Runtime."""
    session = denoiser.get_session()
    
    # preprocessing for DnCNN (usually expects [1, C, H, W] and float32)
    # this is a generic implementation DnCNN variant
    img_float = img.astype(np.float32) / 255.0
    img_input = np.transpose(img_float, (2, 0, 1)) # HWC to CHW
    img_input = np.expand_dims(img_input, axis=0) # add batch dimension
    
    input_name = session.get_inputs()[0].name
    output = session.run(None, {input_name: img_input})[0]
    
    # postprocessing
    output = np.squeeze(output, axis=0)
    output = np.transpose(output, (1, 2, 0)) # CHW to HWC
    output = np.clip(output * 255.0, 0, 255).astype(np.uint8)
    return output

def estimate_noise(img: np.ndarray) -> float:
    """Estimates the noise level in an image using Wavelet Transform (MAD)."""
    if len(img.shape) == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray = img
        
    coeffs = pywt.dwt2(gray, 'haar')
    LL, (LH, HL, HH) = coeffs
    sigma = np.median(np.abs(HH)) / 0.6745
    return float(sigma)

def _preprocess_sync(contents: bytes, denoiser: DnCNNDenoiser) -> tuple[np.ndarray, float, bool]:
    """synchronous preprocessing: decode, estimate noise, conditionally denoise, and enhance."""
    nparr = np.frombuffer(contents, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    
    if img is None:
        raise ValueError("could not decode image")

    noise_level = estimate_noise(img)
    denoised = False

    # conditional denoising
    if noise_level > THRESHOLD:
        try:
            img = apply_dncnn(img, denoiser)
            denoised = True
        except Exception as e:
            # fallback to original if DnCNN fails (e.g. model mismatch)
            print(f"DnCNN failed: {e}")
    
    # always apply CLAHE
    img = apply_clahe(img)
        
    img_resized = cv2.resize(img, (630, 630), interpolation=cv2.INTER_AREA)
    img_resized = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
    img_final = np.expand_dims(img_resized, axis=0) / 255.0
    return img_final, noise_level, denoised

async def preprocess_image(contents: bytes, denoiser: DnCNNDenoiser) -> tuple[np.ndarray, float, bool]:
    """asynchronous wrapper for image preprocessing."""
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(executor, _preprocess_sync, contents, denoiser)
