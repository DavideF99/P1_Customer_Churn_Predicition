import pickle
import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, create_model
from typing import Dict

# Load models
model_paths = {
    "svm_rbf": "pickle_models/svm_rbf_model.pkl",
    "random_forest": "pickle_models/rfc_model.pkl",
    "logistic_regression": "pickle_models/logreg_model.pkl",
    "mlp": "pickle_models/ann_model.pkl",
    "xgboost": "pickle_models/xgb_model.pkl"
}

models = {}
model_feature_names = {}

for model_name, path in model_paths.items():
    with open(path, "rb") as f:
        model = pickle.load(f)
        models[model_name] = model

        # Extract feature names if available
        if hasattr(model, "feature_names_in_"):  # Works for models trained with pandas DataFrame
            feature_names = list(model.feature_names_in_)
        elif hasattr(model, "n_features_in_"):  # Fallback to generic naming if names aren't available
            feature_names = [f"feature_{i+1}" for i in range(model.n_features_in_)]
        else:
            feature_names = [f"feature_{i+1}" for i in range(40)]  # Default if unknown

        model_feature_names[model_name] = feature_names

# Initialize FastAPI app
app = FastAPI()

# Dynamically create Pydantic models with actual feature names
input_models = {}
for model_name, feature_names in model_feature_names.items():
    input_models[model_name] = create_model(
        f"{model_name}_InputModel",
        **{name: (float, ...) for name in feature_names}  # Use actual feature names
    )

# Define prediction endpoint using the dynamically created models
@app.post("/predict/{model_name}")
def predict(model_name: str, data: input_models[model_name]):  
    """
    Predict using the specified model.

    Parameters:
    - model_name: The model to use (e.g., "xgboost", "random_forest").
    - data: JSON input with required feature values.

    Returns:
    - Prediction result.
    """
    if model_name not in models:
        raise HTTPException(status_code=400, detail=f"Invalid model name. Choose from: {', '.join(models.keys())}")

    # Convert input data to NumPy array (use feature names dynamically)
    input_features = np.array([[getattr(data, name) for name in model_feature_names[model_name]]])

    # Make prediction
    model = models[model_name]
    prediction = model.predict(input_features)

    return {"model_used": model_name, "prediction": int(prediction[0])}