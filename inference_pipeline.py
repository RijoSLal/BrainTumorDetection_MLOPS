import mlflow
import numpy as np

class ModelRunner:
    def __init__(self, model_name: str, model_version: str, tracking_uri: str):
        self.model_name = model_name
        self.model_version = model_version
        self.tracking_uri = tracking_uri
        
        try:
            mlflow.set_tracking_uri(self.tracking_uri)
            model_uri = f"models:/{self.model_name}/{self.model_version}"
            self._model = mlflow.tensorflow.load_model(model_uri)
        except Exception as e:
            raise RuntimeError(f"failed to load model from {self.tracking_uri}: {e}")

    def model_prediction(self, img: np.ndarray):
        try:
            prediction = self._model.predict(img)
            prediction_index = int(np.argmax(prediction))
            
            return {"prediction": prediction_index}
        except Exception as e:
            return {"error": f"Prediction failed: {e}"}

