"""
Upload deployment files to Hugging Face Space
"""

import os

from huggingface_hub import upload_folder

HF_SPACE_REPO = "RajnishMohan/tourism-package-prediction"

# Get the folder where this script is located
DEPLOYMENT_FOLDER = os.path.dirname(os.path.abspath(__file__))

upload_folder(

    repo_id=HF_SPACE_REPO,

    repo_type="space",

    folder_path=DEPLOYMENT_FOLDER,

    commit_message="Deploy Streamlit Application"

)

print("Deployment completed successfully.")