"""
Deploy Tourism Package Prediction to Hugging Face

This script is executed automatically by GitHub Actions.
"""

import os

from huggingface_hub import login
from huggingface_hub import upload_file
from huggingface_hub import upload_folder


# ==========================================================
# Hugging Face Configuration
# ==========================================================

HF_TOKEN = os.environ["HF_TOKEN"]

HF_MODEL_REPO = "RajnishMohan/tourism-package-prediction-model"

HF_SPACE_REPO = "RajnishMohan/tourism-package-prediction"

MODEL_FILE = "tourism_package_prediction_model.joblib"


# ==========================================================
# Login
# ==========================================================

login(token=HF_TOKEN)

print("Successfully authenticated with Hugging Face.")


# ==========================================================
# Upload Model
# ==========================================================

MODEL_PATH = os.path.join(
    "models",
    MODEL_FILE
)

upload_file(

    path_or_fileobj=MODEL_PATH,

    path_in_repo=MODEL_FILE,

    repo_id=HF_MODEL_REPO,

    repo_type="model"

)

print("Model uploaded successfully.")


# ==========================================================
# Upload Streamlit Application
# ==========================================================

upload_folder(

    repo_id=HF_SPACE_REPO,

    repo_type="space",

    folder_path="deployment",

    commit_message="Automatic deployment from GitHub Actions"

)

print("Deployment completed successfully.")