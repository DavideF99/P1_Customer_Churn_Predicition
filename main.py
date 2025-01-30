import pickle
import numpy as np
from fastapi import FastAPI
from pydantic import BaseModel

# Load all models from the "pickle_models" folder
model_paths = {
    "svm_rbf": "pickle_models/svm_rbf_model.pkl",
    "random_forest": "pickle_models/rfc_model.pkl",
    "logistic_regression": "pickle_models/logreg_model.pkl",
    "mlp": "pickle_models/ann_model.pkl",
    "xgboost": "pickle_models/xgb_model.pkl"
}

# Load models into a dictionary
models = {}
for model_name, path in model_paths.items():
    with open(path, "rb") as f:
        models[model_name] = pickle.load(f)

# Initialize FastAPI app
app = FastAPI()

# Define input data structure
class InputData(BaseModel):
    feature1: float
    feature2: float
    feature3: float
    feature4: float

# Define prediction endpoint with model selection
@app.post("/predict")
def predict(data: InputData, model_name: str):
    """
    Predict using the specified model.

    Parameters:
        - data: JSON input containing feature values
        - model_name: Name of the model to use (e.g., "xgboost", "random_forest")
    
    Returns:
        - Prediction result
    """
    if model_name not in models:
        return {"error": "Invalid model name. Choose from: " + ", ".join(models.keys())}
    
    # Convert input data into a NumPy array
    input_features = np.array([[data.feature1, data.feature2, data.feature3, data.feature4]])
    
    # Make a prediction
    model = models[model_name]
    prediction = model.predict(input_features)
    
    return {"model_used": model_name, "prediction": int(prediction[0])}
