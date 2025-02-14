from fastapi import FastAPI, UploadFile, File
from fastapi.exceptions import HTTPException
import uvicorn
import mlflow
import cv2
import numpy as np
import filetype
app = FastAPI()

MODEL="tumor_model"
VERSION="1"

class Model_runner:
    def __init__(self,model_name,model_version):
        self.model_name = model_name
        self.model_version=model_version

    def image_preprocessing(self,image):
        img_resized = cv2.resize(image, (630, 630), interpolation=cv2.INTER_AREA)
        img_resized = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
        img = np.expand_dims(img_resized, axis=0) / 255
        return img


    def model_prediction(self,img, ml_model):
        prediction = ml_model.predict(img)
        prediction = np.argmax(prediction)
        text = (
            "Possible signs of a brain tumor identified. Clinical diagnosis required"
            if prediction == 1
            else "No abnormal growth detected. However, consult a doctor for confirmation"
        )
        return {"Prediction": text}

    def loading_model_from_cloud(self):

        mlflow.set_tracking_uri("https://dagshub.com/slalrijo2005/BTD.mlflow")
        model_uri = f"models:/{self.model_name}/{self.model_version}"
        model = mlflow.tensorflow.load_model(model_uri)

        return model

def full_cycle(img,model_name,version):
    initialize=Model_runner(model_name,version)
    loading_model=initialize.loading_model_from_cloud()
    preprocessed=initialize.image_preprocessing(img)
    prediction=initialize.model_prediction(preprocessed,loading_model)
    return prediction
    




@app.post("/")
async def upload_medical_image_file(file: UploadFile = File(...)):
    


    contents = await file.read()
    kind = filetype.guess(contents)
    if kind is None or not kind.mime.startswith("image/"):
        raise HTTPException(status_code=400, detail="Invalid file type. Please upload an image.")

    nparr = np.frombuffer(contents, np.uint8)

    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    prediction=full_cycle(img,MODEL,VERSION)

    return prediction


if __name__ == "__main__":
    

    uvicorn.run("server:app", host="0.0.0.0", port=8000)
