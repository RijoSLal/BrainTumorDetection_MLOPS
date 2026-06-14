from fastapi import FastAPI, UploadFile, File
from fastapi.exceptions import HTTPException
from fastapi.responses import HTMLResponse
import uvicorn
import yaml
from preprocessing_pipeline import preprocess_image, DnCNNDenoiser
from inference_pipeline import ModelRunner
from logs.logging_setup import logger

# configuration
with open("config.yaml", "r") as f:
    config = yaml.safe_load(f)

app = FastAPI()

MODEL = config["model"]["name"]
VERSION = config["model"]["version"]
TRACKING_URI = config["mlflow"]["tracking_uri"]
ALLOWED_IMAGE_TYPES = set(config["validation"]["allowed_image_types"])
MAX_FILE_SIZE = config["validation"]["max_file_size"]

# instantiate the model runner once. the model is loaded in __init__.
try:
    runner = ModelRunner(MODEL, VERSION, TRACKING_URI)
except Exception as e:
    logger.error(f"critical error: failed to initialize model runner. {e}")
    runner = None

# instantiate the denoiser once
denoiser = DnCNNDenoiser(
    repo_id=config["denoising"]["repo_id"],
    filename=config["denoising"]["filename"]
)


async def validate_and_read_file(
    file: UploadFile,
    max_size: int = MAX_FILE_SIZE,
    chunk_size: int = 1024 * 1024,
) -> bytes:
    if file.content_type not in ALLOWED_IMAGE_TYPES:
            raise HTTPException(
                status_code=400,
                detail="only PNG and JPEG images are allowed",
            )

    size = 0
    chunks = []

    while chunk := await file.read(chunk_size):
        size += len(chunk)

        if size > max_size:
            raise HTTPException(
                status_code=413,
                detail="file too large",
            )

        chunks.append(chunk)

    return b"".join(chunks)

@app.get("/", response_class=HTMLResponse)
async def landing_page():
    html_content = """
    <!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>BrainTumorDetection API</title>
    <style>
        body {
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
            line-height: 1.6;
            margin: 2rem auto;
            max-width: 800px;
            color: #333;
            padding: 0 1rem;
        }
        h1, h2, h3 {
            color: #222;
        }
        h3 {
            margin-top: 2rem;
            border-bottom: 1px solid #eaeaea;
            padding-bottom: 0.3rem;
        }
        pre {
            background-color: #f6f8fa;
            padding: 1rem;
            border: 1px solid #d0d7de;
            border-radius: 6px;
            overflow-x: auto;
        }
        code {
            font-family: ui-monospace, SFMono-Regular, SF Mono, Menlo, Consolas, Liberation Mono, monospace;
            font-size: 0.9em;
        }
    </style>
</head>
<body>

    <h1>BrainTumorDetection API</h1>
    
    <p>BrainTumorDetection is a diagnostic API service designed to classify potential tumors in X-ray imaging. Powered by Vision Transformer (ViT) architecture, the processing pipeline utilizes built-in, clinically approved denoising model and image enhancement to ensure high-accuracy classification.</p>

    <h2>Documentation</h2>

    <h3>cURL</h3>
<pre><code>curl -X 'POST' \
    'http://0.0.0.0:8000/' \
    -H 'accept: application/json' \
    -H 'Content-Type: multipart/form-data' \
    -F 'file=@No19.jpg;type=image/jpeg'</code></pre>

    <h3>Python [requests]</h3>
<pre><code>import requests

def detect_tumor(file_path: str) -> dict:
    
    url = "http://0.0.0.0:8000/"
    headers = {
        "accept": "application/json"
    }

    with open(file_path, "rb") as image_file:
        files = {
            "file": ("img", image_file, "image/jpeg")
        }
        response = requests.post(url, headers=headers, files=files)
    
    response.raise_for_status()
    return response.json()

# result = detect_tumor("brain_xray_image.jpeg")</code></pre>

</body>
</html>
    """
    return HTMLResponse(content=html_content)

@app.post("/")
async def upload_medical_image_file(file: UploadFile = File(...)):
    contents = await validate_and_read_file(file)

    # asynchronous preprocessing (includes decoding, noise estimation, and conditional denoising)
    img, noise_level, denoised = await preprocess_image(contents, denoiser)

    # inference logic
    if runner is None:
        raise HTTPException(status_code=503, detail="Model runner not initialized")
        
    prediction = runner.model_prediction(img)
    
    if "error" in prediction:
        raise HTTPException(status_code=500, detail=prediction["error"])

    # add diagnostic information to response
    prediction["noise_level"] = noise_level
    prediction["denoise"] = denoised

    return prediction


# if __name__ == "__main__":
#     uvicorn.run("server:app", host="0.0.0.0", port=8000)
